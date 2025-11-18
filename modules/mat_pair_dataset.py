# mat_pair_dataset.py
import h5py
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader

# ---------- УТИЛИТЫ .MAT ----------

PREF_KEYS = ("LR","HR","l1","l2","l3","l4","tomo","data","vol")

def _get_dataset_key(f: h5py.File) -> str:
    for k in PREF_KEYS:
        if k in f and isinstance(f[k], h5py.Dataset):
            return k
    for k,v in f.items():
        if isinstance(v, h5py.Dataset):
            return k
    # рекурсивно (редко нужно)
    def walk(g):
        for kk,vv in g.items():
            if isinstance(vv, h5py.Dataset):
                return (g.name.strip("/") + "/" + kk) if g.name != "/" else kk
            if isinstance(vv, h5py.Group):
                r = walk(vv)
                if r: return r
        return None
    key = walk(f)
    if not key:
        raise KeyError("В файле не найден ни один HDF5 Dataset")
    return key

def _guess_axes_zyx(shape: Tuple[int,int,int]) -> Tuple[str,int]:
    """
    Эвристика порядка осей. Возвращает ('ZYX', 0) если Z — первая ось,
    иначе ('YXZ', 2) если Z — последняя.
    """
    if shape[0] < shape[1] and shape[0] < shape[2]:
        return ("ZYX", 0)
    if shape[2] < shape[0] and shape[2] < shape[1]:
        return ("YXZ", 2)
    # по умолчанию считаем (Z,Y,X)
    return ("ZYX", 0)

def _read_slice(dset: h5py.Dataset, z: int, z_axis: int) -> np.ndarray:
    if z_axis == 0:   sl = dset[z, :, :]
    elif z_axis == 2: sl = dset[:, :, z]
    else: raise ValueError("Ожидался z_axis 0 или 2")
    return np.array(sl)

def _robust_window(x: np.ndarray, lo=1, hi=99) -> Tuple[float,float]:
    vmin, vmax = np.percentile(x, (lo, hi))
    if vmax <= vmin: vmax = vmin + 1.0
    return float(vmin), float(vmax)

def _normalize01(x: np.ndarray, mask: Optional[np.ndarray]=None) -> np.ndarray:
    x = x.astype(np.float32)
    vals = x[mask] if (mask is not None and mask.any()) else x
    vmin, vmax = _robust_window(vals, 1, 99)
    x = np.clip(x, vmin, vmax)
    x = (x - x.min()) / max(np.ptp(x), 1e-6)
    return x

# ---------- ПАРНЫЕ СРЕЗЫ HR/LR ----------

def _combine_hr_slices(hr_vol: h5py.Dataset, z0: int, method="mean", z_axis=0) -> np.ndarray:
    """
    Собирает 2D HR-карту из четырёх срезов: mean | center
    """
    zs = [z0, z0+1, z0+2, z0+3]
    sls = [_read_slice(hr_vol, z, z_axis) for z in zs]
    if method == "center":
        return sls[1]  # либо 2-й, либо 3-й — как предпочитаешь
    return np.mean(np.stack(sls, axis=0), axis=0)

def _upsample_lr_to_hr(lr2d: np.ndarray, scale_xy: int = 4, out_shape: Optional[Tuple[int,int]]=None) -> np.ndarray:
    """
    Апскейлит LR → HR (билинейно). По умолчанию ×4 по XY.
    """
    if out_shape is None:
        H, W = lr2d.shape
        out_shape = (H*scale_xy, W*scale_xy)
    # skimage resize ожидает (out_H, out_W)
    return resize(lr2d, out_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)

# ---------- DATASET ----------

class ILS1_LR_Pairs(Dataset):
    """
    Пары (LR_slice_upsampled, HR_slice) из ILS_LR (Z=888) и ILS1 (HR, Z=888*4).
    Для каждого z_lr берётся HR z_hr = 4*z_lr..4*z_lr+3 (mean или center).
    Опционально режет патчи.
    """
    def __init__(
        self,
        hr_path: str,          # путь к ILS1.mat
        lr_path: str,          # путь к ILS_LR.mat
        hr_key: Optional[str] = "l1",
        lr_key: Optional[str] = None,
        hr_z_axis: Optional[int] = None,
        lr_z_axis: Optional[int] = None,
        hr_from_four: str = "mean",   # 'mean' | 'center'
        make_patches: bool = True,
        patch_size: int = 256,
        repeats_per_slice: int = 4,
        seed: int = 42,
    ):
        self.hr_f = h5py.File(hr_path, "r")
        self.lr_f = h5py.File(lr_path, "r")

        # ключи
        self.hr_key = hr_key if (hr_key and hr_key in self.hr_f) else _get_dataset_key(self.hr_f)
        self.lr_key = lr_key if (lr_key and lr_key in self.lr_f) else _get_dataset_key(self.lr_f)

        self.HR = self.hr_f[self.hr_key]  # h5py.Dataset
        self.LR = self.lr_f[self.lr_key]

        # оси
        if hr_z_axis is None:
            self.hr_axes, self.hr_z_axis = _guess_axes_zyx(self.HR.shape)
        else:
            self.hr_axes, self.hr_z_axis = ("ZYX" if hr_z_axis==0 else "YXZ"), hr_z_axis

        if lr_z_axis is None:
            self.lr_axes, self.lr_z_axis = _guess_axes_zyx(self.LR.shape)
        else:
            self.lr_axes, self.lr_z_axis = ("ZYX" if lr_z_axis==0 else "YXZ"), lr_z_axis

        # формы
        if self.hr_z_axis == 0:
            Z_hr, H_hr, W_hr = self.HR.shape
        else:
            H_hr, W_hr, Z_hr = self.HR.shape

        if self.lr_z_axis == 0:
            Z_lr, H_lr, W_lr = self.LR.shape
        else:
            H_lr, W_lr, Z_lr = self.LR.shape

        # ожидания: Z_hr ≈ 4*Z_lr; H_hr ≈ 4*H_lr; W_hr ≈ 4*W_lr
        self.scale_xy = (H_hr / H_lr + W_hr / W_lr) * 0.5
        self.scale_z  = Z_hr / Z_lr
        self.hr_from_four = hr_from_four
        self.make_patches = make_patches
        self.patch = int(patch_size)
        self.repeats = int(repeats_per_slice)
        self.rng = np.random.RandomState(seed)

        # ограничим пересечение по Z на всякий случай
        max_z_lr = min(Z_lr, Z_hr // 4)
        self.valid_z_lr = list(range(max_z_lr))
        self.H_hr, self.W_hr = H_hr, W_hr

        # лог
        print(f"[INFO] HR key={self.hr_key} shape={self.HR.shape} axes={self.hr_axes} z_axis={self.hr_z_axis}")
        print(f"[INFO] LR key={self.lr_key} shape={self.LR.shape} axes={self.lr_axes} z_axis={self.lr_z_axis}")
        print(f"[INFO] scale_xy≈{self.scale_xy:.3f}  scale_z≈{self.scale_z:.3f}  (ожидаем ~4.0 и ~4.0)")
        print(f"[INFO] slices: using {len(self.valid_z_lr)} LR-slices (0..{max(self.valid_z_lr)})")

    def __len__(self):
        return len(self.valid_z_lr) * (self.repeats if self.make_patches else 1)

    def _pair_by_z(self, z_lr: int) -> Tuple[np.ndarray, np.ndarray]:
        # HR из 4-х срезов
        z0 = int(4 * z_lr)
        hr2d = _combine_hr_slices(self.HR, z0, method=self.hr_from_four, z_axis=self.hr_z_axis).astype(np.float32)
        # LR один срез → апскейл ×4 до размера HR
        lr2d = _read_slice(self.LR, z_lr, self.lr_z_axis).astype(np.float32)
        lr_up = _upsample_lr_to_hr(lr2d, scale_xy=int(round(self.scale_xy)), out_shape=(hr2d.shape[0], hr2d.shape[1]))
        return lr_up, hr2d

    def _sample_patch_same_xy(self, lr_up: np.ndarray, hr2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = hr2d.shape
        p = min(self.patch, H, W)
        if (H < p) or (W < p):
            # паддинг до p
            pad_y = max(0, p - H); pad_x = max(0, p - W)
            hr2d = np.pad(hr2d, ((0,pad_y),(0,pad_x)), mode='edge')
            lr_up = np.pad(lr_up, ((0,pad_y),(0,pad_x)), mode='edge')
            H, W = hr2d.shape
        y0 = self.rng.randint(0, H - p + 1)
        x0 = self.rng.randint(0, W - p + 1)
        return lr_up[y0:y0+p, x0:x0+p], hr2d[y0:y0+p, x0:x0+p]

    def __getitem__(self, idx: int):
        if self.make_patches:
            i = idx // self.repeats
        else:
            i = idx
        z_lr = self.valid_z_lr[i]
        lr_up, hr2d = self._pair_by_z(z_lr)

        # робастная нормализация отдельно (как визуальный/процедурный стандарт)
        lr_up_n = _normalize01(lr_up)
        hr2d_n  = _normalize01(hr2d)

        if self.make_patches:
            lr_p, hr_p = self._sample_patch_same_xy(lr_up_n, hr2d_n)
            # в тензоры (C,H,W)
            return {
                "z_lr": z_lr,
                "lr": torch.from_numpy(lr_p[None]),
                "hr": torch.from_numpy(hr_p[None]),
            }
        else:
            return {
                "z_lr": z_lr,
                "lr": torch.from_numpy(lr_up_n[None]),
                "hr": torch.from_numpy(hr2d_n[None]),
            }

    def close(self):
        try: self.hr_f.close()
        except: pass
        try: self.lr_f.close()
        except: pass

# ---------- ПРЕВЬЮ ----------

def preview_pair(hr_path: str, lr_path: str, out_dir: str, z_lr: int = 100, hr_from_four="mean"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = ILS1_LR_Pairs(hr_path, lr_path, hr_key="l1", hr_from_four=hr_from_four,
                       make_patches=False)
    lr_up, hr2d = ds._pair_by_z(z_lr)
    lr_up_n = _normalize01(lr_up)
    hr2d_n  = _normalize01(hr2d)

    vmin_h, vmax_h = _robust_window(hr2d)
    vmin_l, vmax_l = _robust_window(lr_up)

    plt.figure(figsize=(10,5), dpi=120)
    plt.subplot(1,2,1); plt.imshow(hr2d, cmap='gray', vmin=vmin_h, vmax=vmax_h); plt.title(f"HR (z_hr={4*z_lr}..{4*z_lr+3})"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(lr_up, cmap='gray', vmin=vmin_l, vmax=vmax_l); plt.title(f"LR upscaled ×{ds.scale_xy:.1f} (z_lr={z_lr})"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out/"pair_raw.png"); plt.close()

    plt.figure(figsize=(10,5), dpi=120)
    plt.subplot(1,2,1); plt.imshow(hr2d_n, cmap='gray', vmin=0, vmax=1); plt.title("HR normalized"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(lr_up_n, cmap='gray', vmin=0, vmax=1); plt.title("LR up normalized"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out/"pair_norm.png"); plt.close()

    # патч для контроля
    ds.make_patches = True
    sample = ds[0]
    hr_p = sample["hr"].numpy().squeeze()
    lr_p = sample["lr"].numpy().squeeze()
    plt.figure(figsize=(6,3), dpi=120)
    plt.subplot(1,2,1); plt.imshow(hr_p, cmap='gray', vmin=0, vmax=1); plt.title("HR patch"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(lr_p, cmap='gray', vmin=0, vmax=1); plt.title("LR patch"); plt.axis('off')
    plt.tight_layout(); plt.savefig(out/"patch_norm.png"); plt.close()

    ds.close()
    print(f"[SAVED] {out/'pair_raw.png'}")
    print(f"[SAVED] {out/'pair_norm.png'}")
    print(f"[SAVED] {out/'patch_norm.png'}")

# ---------- ПРИМЕР ИСПОЛЬЗОВАНИЯ (раскомментируй) ----------
if __name__ == "__main__":
    HR_PATH = r"C:\MEC_ILS\mat_volumes_ILS\ILS1.mat"   # HR-part1
    LR_PATH = r"C:\MEC_ILS\mat_volumes_ILS\ILS_LR.mat" # LR (888 слоёв)

    # превью пары и патча
    preview_pair(HR_PATH, LR_PATH, out_dir=r"C:\MEC_ILS\preview_pairs", z_lr=120, hr_from_four="mean")

    # датасет и лоадер
    ds = ILS1_LR_Pairs(HR_PATH, LR_PATH, hr_key="l1", hr_from_four="mean",
                       make_patches=True, patch_size=256, repeats_per_slice=4)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

    batch = next(iter(dl))
    print("batch lr:", batch["lr"].shape, batch["lr"].dtype)
    print("batch hr:", batch["hr"].shape, batch["hr"].dtype)
    ds.close()