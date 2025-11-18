import argparse, math, time
from pathlib import Path
from typing import Tuple, Optional, List

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_msssim import ssim as ssim_torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms import InterpolationMode

# твоя модель
from unet2d import UNet2D, UNetConfig

# --------------- утилиты .mat ---------------

PREF_KEYS = ("LR","HR","l1","l2","l3","l4","tomo","data","vol")

def get_dataset_key(f: h5py.File) -> str:
    for k in PREF_KEYS:
        if k in f and isinstance(f[k], h5py.Dataset):
            return k
    # первый dataset верхнего уровня
    for k, v in f.items():
        if isinstance(v, h5py.Dataset):
            return k
    # рекурсивно
    def walk(g):
        for kk, vv in g.items():
            if isinstance(vv, h5py.Dataset):
                return (g.name.strip("/") + "/" + kk) if g.name != "/" else kk
            if isinstance(vv, h5py.Group):
                r = walk(vv)
                if r: return r
        return None
    k = walk(f)
    if not k:
        raise KeyError("В файле не найден dataset")
    return k

def guess_axes_zyx(shape: Tuple[int,int,int]) -> Tuple[str,int]:
    """Возвращает ('ZYX',0) если Z первая ось, иначе ('YXZ',2) если Z последняя."""
    if shape[0] < shape[1] and shape[0] < shape[2]:
        return ("ZYX", 0)
    if shape[2] < shape[0] and shape[2] < shape[1]:
        return ("YXZ", 2)
    return ("ZYX", 0)  # дефолт

def read_slice(dset: h5py.Dataset, z: int, z_axis: int) -> np.ndarray:
    if z_axis == 0:   sl = dset[z, :, :]
    elif z_axis == 2: sl = dset[:, :, z]
    else: raise ValueError("Ожидался z_axis 0 или 2")
    return np.array(sl)

def robust_window(x: np.ndarray, lo=1, hi=99) -> Tuple[float, float]:
    vmin, vmax = np.percentile(x, (lo, hi))
    if vmax <= vmin: vmax = vmin + 1.0
    return float(vmin), float(vmax)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    vmin, vmax = robust_window(x, 1, 99)
    x = np.clip(x, vmin, vmax)
    x = (x - x.min()) / max(np.ptp(x), 1e-6)
    return x

def upsample_to(shape_hw: Tuple[int,int], img: np.ndarray) -> np.ndarray:
    """билинейный апскейл numpy [H,W] -> out_shape"""
    H, W = shape_hw
    ten = torch.from_numpy(img[None, None].astype(np.float32))
    ten_up = torch.nn.functional.interpolate(ten, size=(H, W), mode="bilinear", align_corners=False)
    return ten_up.numpy().squeeze(0).squeeze(0)

# --------------- метрики ---------------

def psnr_torch(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((x - y) ** 2, dim=[1,2,3]).clamp_min(1e-10)
    val = 20.0 * torch.log10(max_val / torch.sqrt(mse))
    return float(val.mean())

# --------------- Dataset ---------------

class MRCCM_ILSpairs(Dataset):
    """
    Пары 2D (LR_up, HR) из ILS_LR и ILS1 с принудительным 4× по Z:
      - HR берётся как среднее 4 подряд слоёв из ILS1
      - LR берётся только из диапазона, покрываемого этой HR-частью
      - LR апскейлится до размера HR (bicubic)
    Важно: h5py файлы открываются лениво внутри воркера (__getitem__), поэтому совместимо с DataLoader.
    """
    def __init__(self,
                 hr_mat: str, lr_mat: str,
                 hr_key: Optional[str] = None, lr_key: Optional[str] = None,
                 limit_pairs: Optional[int] = 100, sample_seed: int = 42,
                 hr_z_offset: int = 0,   # =0 для ILS1, 888 для ILS2, 1776 для ILS3, 2664 для ILS4
                 force_4x: bool = True):
        # --- только пути и метаданные, без открытых h5py.File ---
        self.hr_mat_path = str(hr_mat)
        self.lr_mat_path = str(lr_mat)
        self.hr_f = None; self.lr_f = None
        self.HR = None;    self.LR = None

        # Определим ключи и формы аккуратно (временное открытие)
        with h5py.File(self.hr_mat_path, "r") as fhr:
            self.hr_key = hr_key if (hr_key and hr_key in fhr) else get_dataset_key(fhr)
            hr_shape = fhr[self.hr_key].shape
        with h5py.File(self.lr_mat_path, "r") as flr:
            self.lr_key = lr_key if (lr_key and lr_key in flr) else get_dataset_key(flr)
            lr_shape = flr[self.lr_key].shape

        self.hr_axes, self.hr_z = guess_axes_zyx(hr_shape)
        self.lr_axes, self.lr_z = guess_axes_zyx(lr_shape)

        # размеры
        if self.hr_z == 0: Z_hr, H_hr, W_hr = hr_shape
        else:               H_hr, W_hr, Z_hr = hr_shape
        if self.lr_z == 0: Z_lr, H_lr, W_lr = lr_shape
        else:               H_lr, W_lr, Z_lr = lr_shape

        self.H_hr, self.W_hr = int(H_hr), int(W_hr)
        self.scale_xy = (H_hr / H_lr + W_hr / W_lr) * 0.5

        # режим Z
        self.force_4x = bool(force_4x)
        self.hr_z_offset = int(hr_z_offset)

        # Диапазон LR, покрываемый данной HR-частью
        if self.force_4x:
            lr_z_start = self.hr_z_offset // 4
            lr_z_count = Z_hr // 4
            lr_z_end = min(lr_z_start + lr_z_count, Z_lr)
            base_idxs = list(range(lr_z_start, lr_z_end))
        else:
            base_idxs = list(range(min(Z_lr, Z_hr)))

        # равномерно сократим до limit_pairs
        if limit_pairs is not None and len(base_idxs) > limit_pairs:
            grid = np.linspace(0, len(base_idxs)-1, num=limit_pairs, dtype=int)
            self.idxs = [base_idxs[i] for i in grid]
        else:
            self.idxs = base_idxs

        print(f"[INFO] HR key={self.hr_key} shape={hr_shape} axes={self.hr_axes} z_axis={self.hr_z}")
        print(f"[INFO] LR key={self.lr_key} shape={lr_shape} axes={self.lr_axes} z_axis={self.lr_z}")
        print(f"[INFO] scale_xy≈{self.scale_xy:.3f} | Z_mode={'4x' if self.force_4x else '1x'} "
              f"| hr_z_offset={self.hr_z_offset} | LR z range used: "
              f"[{self.idxs[0] if self.idxs else '-'} .. {self.idxs[-1] if self.idxs else '-'}] | pairs={len(self.idxs)}")

    # ленивое открытие файлов в процессе-воркере
    def _ensure_open(self):
        if self.hr_f is None:
            self.hr_f = h5py.File(self.hr_mat_path, "r")
            self.HR = self.hr_f[self.hr_key]
        if self.lr_f is None:
            self.lr_f = h5py.File(self.lr_mat_path, "r")
            self.LR = self.lr_f[self.lr_key]

    # делаем dataset пиклируемым (обнуляем непиклируемые поля)
    def __getstate__(self):
        st = self.__dict__.copy()
        st["hr_f"] = None; st["lr_f"] = None; st["HR"] = None; st["LR"] = None
        return st
    def __setstate__(self, st):
        self.__dict__.update(st)

    def __len__(self) -> int:
        return len(self.idxs)

    def _hr_from_zlr(self, z_lr: int) -> np.ndarray:
        self._ensure_open()
        if self.force_4x:
            # где начинается этот LR-слой в HR с учётом смещения части
            z0_global = self.hr_z_offset + 4 * (z_lr - (self.hr_z_offset // 4))
            zs = [z0_global + k for k in range(4)]
            sls = [read_slice(self.HR, z, self.hr_z) for z in zs]
            hr2d = np.mean(np.stack(sls, 0), 0)
        else:
            hr2d = read_slice(self.HR, z_lr, self.hr_z)
        return hr2d.astype(np.float32)

    def __getitem__(self, i: int):
        self._ensure_open()
        z_lr = self.idxs[i]
        lr2d = read_slice(self.LR, z_lr, self.lr_z).astype(np.float32)
        hr2d = self._hr_from_zlr(z_lr)

        lr2d = normalize01(lr2d)
        hr2d = normalize01(hr2d)

        lr_up = upsample_to((self.H_hr, self.W_hr), lr2d)

        lr_t = torch.from_numpy(lr_up)[None]
        hr_t = torch.from_numpy(hr2d)[None]
        return lr_t, hr_t, f"z{z_lr:04d}"

    def close(self):
        try:
            if self.hr_f is not None: self.hr_f.close()
        except: pass
        try:
            if self.lr_f is not None: self.lr_f.close()
        except: pass


# --------------- Основная логика ---------------

@torch.no_grad()
def eval_bicubic(dl, device):
    psnrs, ssims = [], []
    for lr, hr, _ in dl:
        lr = lr.to(device); hr = hr.to(device)
        psnrs.append(psnr_torch(lr, hr, max_val=1.0))  # уже float
        ssim_val = ssim_torch(lr, hr, data_range=1.0, size_average=True)
        ssims.append(float(ssim_val.item()))
    return float(np.mean(psnrs)), float(np.mean(ssims))

@torch.no_grad()
def eval_model(dl: DataLoader, model: torch.nn.Module, device: torch.device,
               save_dir: Optional[Path] = None, save_n: int = 16):
    psnrs, ssims = [], []
    saved = 0
    to_pil = T.ToPILImage()

    for lr, hr, names in dl:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
            pred = model(lr)
        # подстрахуем spatial
        if pred.shape[-2:] != hr.shape[-2:]:
            pred = torch.nn.functional.interpolate(pred, size=hr.shape[-2:], mode="bilinear", align_corners=False)

        psnrs.append(psnr_torch(pred, hr, max_val=1.0))
        ssim_val = ssim_torch(pred, hr, data_range=1.0, size_average=True)
        ssims.append(float(ssim_val.item()))

        if save_dir is not None and saved < save_n:
            for b in range(min(pred.size(0), save_n - saved)):
                stem = names[b]
                to_pil(lr[b].clamp(0,1)).save(save_dir / f"{stem}_lr.png")
                to_pil(pred[b].clamp(0,1)).save(save_dir / f"{stem}_pred.png")
                to_pil(hr[b].clamp(0,1)).save(save_dir / f"{stem}_hr.png")
                saved += 1
                if saved >= save_n: break

    return float(np.mean(psnrs)), float(np.mean(ssims))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_mat", type=str, required=True, help="путь к ILS1.mat (HR)")
    ap.add_argument("--lr_mat", type=str, required=True, help="путь к ILS_LR.mat (LR)")
    ap.add_argument("--ckpt",   type=str, required=True, help="путь к твоему .pt чекпойнту модели")
    ap.add_argument("--limit_pairs", type=int, default=100, help="сколько пар слоёв валидировать (равномерно по Z)")
    ap.add_argument("--batch_size",  type=int, default=4)
    ap.add_argument("--workers",     type=int, default=0)
    ap.add_argument("--save_dir",    type=str, default="preds_x2")  # как ты просил
    ap.add_argument("--save_n",      type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "-")

    # датасет и лоадер
    # ILS1 = первая часть HR -> hr_z_offset=0
    ds = MRCCM_ILSpairs(
        args.hr_mat, args.lr_mat,
        hr_key="l1",
        limit_pairs=args.limit_pairs,
        hr_z_offset=0,     # ILS1
        force_4x=True
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=(device.type=="cuda"))

    # bicubic baseline
    b_psnr, b_ssim = eval_bicubic(dl, device)
    print(f"[baseline] Bicubic  | PSNR: {b_psnr:.2f} dB | SSIM: {b_ssim:.4f}")

    # модель
    model = UNet2D(UNetConfig(
        in_channels=1, out_channels=1, base_channels=64, depth=4,
        norm_enc=True, norm_dec=True, up_mode="bilinear", dropout=0.0
    )).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")

    def strip_prefix(sd, prefix="module."):
        return { (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items() }
    
    if isinstance(ckpt, dict):
        # частые варианты упаковки
        for key in ("model", "state_dict", "net", "ema"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    
    ckpt = strip_prefix(ckpt, "module.")
    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    print(f"[ckpt] loaded params: {len(ckpt)} | missing: {len(missing)} | unexpected: {len(unexpected)}")
    if len(ckpt) == 0 or len(missing) > 0:
        print("[WARN] Weights didn’t match architecture — prediction may be identity.")

    model.eval()

    # предсказания и сохранение примеров
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    m_psnr, m_ssim = eval_model(dl, model, device, save_dir=out_dir, save_n=args.save_n)
    print(f"[model   ] UNet2D   | PSNR: {m_psnr:.2f} dB | SSIM: {m_ssim:.4f}")

    # сравнение
    d_psnr = m_psnr - b_psnr
    d_ssim = m_ssim - b_ssim
    print(f"[delta   ] vs Bicubic | ΔPSNR: {d_psnr:+.2f} dB | ΔSSIM: {d_ssim:+.4f}")
    print(f"[saved] examples in: {out_dir.resolve()}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
