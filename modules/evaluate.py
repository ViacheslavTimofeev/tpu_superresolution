"""
Проверка качества сохраненных моделей.
"""
import os, argparse, math, time, re
from pathlib import Path
from typing import Tuple, Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pytorch_msssim import ssim
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from unet2d import UNet2D, UNetConfig

from sr_transforms import build_pair_transform_eval
from sr_datasets import Shuffled2DPaired, MRCCMPairedByZ

V_MIN = 8557.25
V_MAX = 47033.0

V_MIN_05 = 11072.5
V_MAX_995 = 18645.0

# ============ метрики и utils ============

def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    """Ожидает тензоры в [0,1]. Возвращает средний PSNR по батчу."""
    mse = torch.mean((x - y) ** 2, dim=[1,2,3])  # [B]
    mse = torch.clamp(mse, min=1e-10)
    val = 20.0 * torch.log10(max_val / torch.sqrt(mse))  # [B]
    return float(val.mean())

def save_tensor_as_png(x: torch.Tensor, path: Path, per_image_rescale: bool = False):
    x = x.detach().cpu()

    if per_image_rescale:
        # пер-изображенческий min-max в НОРМАЛИЗОВАННОМ пространстве
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max <= x_min + 1e-8:
            x = torch.zeros_like(x)
        else:
            x = (x - x_min) / (x_max - x_min)
    else:
        x = x.clamp(0.0, 1.0)

    if x.shape[0] == 1:
        x = x[0]          # [H,W]
    else:
        x = x.permute(1, 2, 0)  # [H,W,C]

    pil = T.ToPILImage()(x)
    pil.save(str(path))


# ============ основной eval ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sign", type=str, choices=["deeprock_x2", "deeprock_x4", "mrccm"],
                help="Конфигурация набора данных")
    ap.add_argument("--data_root", type=str, default="C:/Users/Вячеслав/Documents/superresolution/DeepRockSR-2D")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--ckpt", type=str, required=True)        # путь к .pt
    ap.add_argument("--save_dir", type=str, default="preds")  # куда сохранить примеры
    ap.add_argument("--save_n", type=int, default=16)         # сколько картинок сохранить
    ap.add_argument("--z_stride", type=int, default=1)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "-")
    
    if args.sign in ("deeprock_x2", "deeprock_x4"):
        dataset_kind = "DeepRock"
        scale = "X2" if args.sign == "deeprock_x2" else "X4"
    elif args.sign == "mrccm":
        dataset_kind = "mrccm"
        scale = None
    else:
        raise ValueError(f"Unknown sign: {args.sign}")

    tf_test = build_pair_transform_eval(
        normalize=False,
        dataset=dataset_kind,
        vmin=V_MIN,
        vmax=V_MAX,
    )

    if dataset_kind == "DeepRock":
        test_ds = Shuffled2DPaired(
            args.data_root,
            split="test",
            scale=scale,
            transform_pair=tf_test,
    )
    elif dataset_kind == "mrccm":
        test_ds = MRCCMPairedByZ(
            Path(args.data_root) / "LR_test",
            Path(args.data_root) / "HR_test",
            transform_pair=tf_test,
            stride=args.z_stride,
        )
    else:
        raise ValueError(f"Unknown dataset_kind: {dataset_kind}")


    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
    )
    print(f"[data] test samples: {len(test_ds)} | steps: {len(test_loader)}")

    
    @torch.no_grad()
    def _peek_batch(loader):
        try:
            (lr, hr) = next(iter(loader))
        except StopIteration:
            return
        import torch
        lr_f = torch.isfinite(lr)
        hr_f = torch.isfinite(hr)
        print("[peek] lr min/max:",
              float(lr[lr_f].min().item()) if lr_f.any() else float("nan"),
              float(lr[lr_f].max().item()) if lr_f.any() else float("nan"),
              "| hr min/max:",
              float(hr[hr_f].min().item()) if hr_f.any() else float("nan"),
              float(hr[hr_f].max().item()) if hr_f.any() else float("nan"),
              "| shapes:", tuple(lr.shape), tuple(hr.shape))
    _peek_batch(test_loader)

    # --- Bicubic baseline (LR upscaled = prediction) ---
    @torch.no_grad()
    def eval_bicubic_baseline(test_loader, device, mean=(0.45161797,), std=(0.20893379,)):
        psnrs, ssims = [], []
        for (lr, hr) in test_loader:
            # lr/hr уже нормализованы и LR апскейлен до HR в трансформе
            lr = lr.to(device); hr = hr.to(device)
            # метрики
            #psnrs.append(psnr(lr_dn, hr_dn, max_val=1.0))
            #ssims.append(ssim(lr_dn, hr_dn, data_range=1.0, size_average=True))
            psnrs.append(psnr(lr, hr, max_val=1.0))
            ssims.append(ssim(lr, hr, data_range=1.0, size_average=True))
        return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)
    
    bic_psnr, bic_ssim = eval_bicubic_baseline(test_loader, device)
    print(f"[baseline] Bicubic PSNR: {bic_psnr:.2f} dB | SSIM: {bic_ssim:.4f}")
    
    cfg = UNetConfig(
        in_channels=1, out_channels=1,
        base_channels=64, depth=4,
        norm_enc=False, norm_dec=False,
        up_mode="pixelshuffle", dropout=0.0
    )
    model = UNet2D(cfg).to(device)

    # --- загрузка чекпойнта (.pt) ---
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
        print("[ckpt] loaded state_dict from 'model' key")
    else:
        model.load_state_dict(ckpt, strict=True)
        print("[ckpt] loaded raw state_dict")

    model.eval()

    t0 = time.time()
    psnr_vals, ssim_vals = [], []
    out_dir = Path(args.save_dir); out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.no_grad():
        for (lr, hr) in test_loader:
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            name = [f"sample_{i}" for i in range(lr.size(0))]
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda")):
                pred = model(lr)
                if not torch.isfinite(pred).all():
                    bad = (~torch.isfinite(pred)).float().mean().item()
                    pmin = torch.nanmin(pred).item()
                    pmax = torch.nanmax(pred).item()
                    raise RuntimeError(f"Pred has non-finite values: share={bad:.6f}, min={pmin:.4g}, max={pmax:.4g}")

            # выровнять spatial, если надо
            if pred.shape[-2:] != hr.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=hr.shape[-2:], mode="bilinear", align_corners=False
                )

            #psnr_vals.append(psnr(pred_dn, hr_dn, max_val=1.0))
            #ssim_vals.append(ssim(pred_dn, hr_dn, data_range=1.0, size_average=True))
            psnr_vals.append(psnr(pred, hr, max_val=1.0))
            ssim_vals.append(ssim(pred, hr, data_range=1.0, size_average=True))

            # сохранить несколько примеров
            if saved < args.save_n:
                for b in range(min(pred.size(0), args.save_n - saved)):
                    stem = f"sample_{saved}"
                    
                    # для MRCCM хотим "как TIFF" → включаем per_image_rescale
                    per_rescale = (dataset_kind == "mrccm")

                    save_tensor_as_png(lr[b],   out_dir / f"{stem}_lr.png",   per_image_rescale=per_rescale)
                    save_tensor_as_png(pred[b], out_dir / f"{stem}_pred.png", per_image_rescale=per_rescale)
                    save_tensor_as_png(hr[b],   out_dir / f"{stem}_hr.png",   per_image_rescale=per_rescale)
                    
                    saved += 1
                    
                    if saved >= args.save_n:
                        break

    dt = time.time() - t0
    mean_psnr = sum(psnr_vals) / max(1, len(psnr_vals))
    mean_ssim = sum(ssim_vals) / max(1, len(ssim_vals))
    print(f"[done] test PSNR: {mean_psnr:.2f} dB | SSIM: {mean_ssim:.4f} | "
          f"time: {dt:.1f}s for {len(test_ds)} samples")
    print(f"[saved] examples in: {out_dir.resolve()}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()