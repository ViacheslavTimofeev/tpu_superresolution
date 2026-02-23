import os
import warnings

# 1) чтобы работало в дочерних процессах DataLoader (Windows spawn)
os.environ["PYTHONWARNINGS"] = "ignore::pydantic.warnings.UnsupportedFieldAttributeWarning"

# 2) для текущего процесса
try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
except Exception:
    pass

import os
import re
import time
import math
import argparse
import random
from pathlib import Path
from datetime import timedelta

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision.transforms.v2 import functional as TF
from PIL import Image

from sr_datasets import Shuffled2DPaired
from network_swinir import SwinIR

# -------------------------
# Utils
# -------------------------

def worker_init_fn(_):
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
    
def fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def make_loader(ds, batch_size, workers, worker_init_fn, pin=True, shuffle=False, drop_last=False, persistent=False):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        num_workers=workers,
        pin_memory=pin,
    )
    if workers and workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)

def l1_loss(pred, target):
    return F.l1_loss(pred, target)

def batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target, reduction="none").view(pred.size(0), -1).mean(dim=1)
    psnr = 20.0 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr


# -------------------------
# Minimal paired transforms (NO augmentations)
# -------------------------
def _ensure_3ch(t: torch.Tensor) -> torch.Tensor:
    # t: [C,H,W], C can be 1 or 3
    if t.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(t.shape)}")
    if t.size(0) == 1:
        t = t.repeat(3, 1, 1)
    elif t.size(0) != 3:
        raise ValueError(f"Expected C=1 or C=3, got C={t.size(0)}")
    return t

def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    # TF.to_image -> tensor in [0,255] uint8 sometimes; safest:
    x = TF.to_image(img)
    x = TF.to_dtype(x, dtype=torch.float32, scale=True)
    return x

def paired_random_crop(lr_t: torch.Tensor, hr_t: torch.Tensor, lr_patch: int, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    # lr: [C,h,w], hr: [C, h*scale, w*scale]
    _, h, w = lr_t.shape
    if h < lr_patch or w < lr_patch:
        raise ValueError(f"LR image too small for patch {lr_patch}: lr_size=({h},{w})")

    top = random.randint(0, h - lr_patch)
    left = random.randint(0, w - lr_patch)

    lr_crop = lr_t[:, top:top+lr_patch, left:left+lr_patch]
    hr_top = top * scale
    hr_left = left * scale
    hr_patch = lr_patch * scale
    hr_crop = hr_t[:, hr_top:hr_top+hr_patch, hr_left:hr_left+hr_patch]
    return lr_crop, hr_crop

class PairTransformTrain:
    def __init__(self, lr_patch: int, scale: int):
        self.lr_patch = lr_patch
        self.scale = scale

    def __call__(self, lr_pil: Image.Image, hr_pil: Image.Image):
        lr = _ensure_3ch(pil_to_tensor01(lr_pil))
        hr = _ensure_3ch(pil_to_tensor01(hr_pil))
        lr, hr = paired_random_crop(lr, hr, self.lr_patch, self.scale)
        return lr, hr

class PairTransformValid:
    def __init__(self, scale: int):
        self.scale = scale

    def __call__(self, lr_pil: Image.Image, hr_pil: Image.Image):
        # full-image validation (no crop)
        lr = _ensure_3ch(pil_to_tensor01(lr_pil))
        hr = _ensure_3ch(pil_to_tensor01(hr_pil))
        return lr, hr
        
def assert_finite(x, name: str):
    finite = torch.isfinite(x)
    if not finite.all():
        bad = (~finite).sum().item()
        x_f = x[finite]
        if x_f.numel() > 0:
            mn = x_f.min().item()
            mx = x_f.max().item()
        else:
            mn, mx = float("nan"), float("nan")
        raise RuntimeError(f"{name} has non-finite values: count={bad}, finite_min={mn}, finite_max={mx}")

# -------------------------
# Train/Val loops
# -------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, sched=None, batch_sched=False, grad_clip=1.0):
    model.train()
    total = 0.0
    n = 0
    t0 = time.time()

    for lr, hr in loader:
        
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=False):#(device.type == "cuda")):
            out = model(lr)
            loss = loss_fn(out, hr)
        assert_finite(out, "out")
        assert_finite(loss, "loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        if batch_sched and sched is not None:
            sched.step()
        scaler.update()

        total += loss.item()
        n += 1

    return total / max(1, n), time.time() - t0

@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    sum_psnr = 0.0
    n_imgs = 0
    t0 = time.time()

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(lr)
            loss = loss_fn(out, hr)

        total += loss.item()
        n += 1

        psnr_vals = batch_psnr(out, hr)
        sum_psnr += psnr_vals.sum().item()
        n_imgs += lr.size(0)

    mean_loss = total / max(1, n)
    mean_psnr = sum_psnr / max(1, n_imgs)
    return mean_loss, mean_psnr, time.time() - t0


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--scale", type=str, choices=["X2", "X4"], required=True)
    ap.add_argument("--weights", type=str, required=True, help="Path to SwinIR pretrained checkpoint (.pth/.pt)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr_patch", type=int, default=64, help="LR patch size (HR patch = lr_patch*scale)")
    ap.add_argument("--lr", type=float, default=2e-5, help="Fine-tune LR (safe default)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pin", action="store_true")
    ap.add_argument("--no_persistent", action="store_true")

    # optional: freeze part of model by regex
    ap.add_argument("--freeze_regex", type=str, default=None,
                    help="Regex over parameter names to freeze (e.g. 'conv_first|layers\\.0|layers\\.1')")
    # scheduler (simple + robust)
    ap.add_argument("--scheduler", type=str, choices=["None", "Cosine"], default="Cosine")
    ap.add_argument("--min_lr", type=float, default=2e-6, help="For CosineAnnealingLR")
    ap.add_argument("--grad_clip", type=float, default=1.0)

    args = ap.parse_args()

    seed_everything(args.seed)
    if args.workers is None:
        cpu = os.cpu_count() or 4
        args.workers = min(8, max(2, cpu // 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "-")

    scale_int = 2 if args.scale.upper() == "X2" else 4

    # ---- datasets / loaders ----
    train_tf = PairTransformTrain(lr_patch=args.lr_patch, scale=scale_int)
    valid_tf = PairTransformValid(scale=scale_int)

    train_ds = Shuffled2DPaired(args.data_root, split="train", scale=args.scale, transform_pair=train_tf)
    valid_ds = Shuffled2DPaired(args.data_root, split="valid", scale=args.scale, transform_pair=valid_tf)

    train_loader = make_loader(
        train_ds, args.batch_size, args.workers,
        pin=not args.no_pin, shuffle=True, drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent=not args.no_persistent,
    )
    valid_loader = make_loader(
        valid_ds, max(1, args.batch_size // 2), args.workers,
        worker_init_fn=worker_init_fn,
        pin=not args.no_pin, shuffle=False, drop_last=False,
        persistent=not args.no_persistent,
    )

    # ---- model ----
    model = SwinIR(
        upscale=scale_int,
        in_chans=3,
        img_size=64,         # doesn't have to match real image size; used for internal patch resolution init
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ).to(device)

    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt["params"] if isinstance(ckpt, dict) and "params" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=True)
    print(f"[weights] loaded: {args.weights}")
    print(f"[weights] missing={len(missing)}, unexpected={len(unexpected)}")

    # ---- optional freezing ----
    if args.freeze_regex:
        pattern = re.compile(args.freeze_regex)
        froze = 0
        for name, p in model.named_parameters():
            if pattern.search(name):
                p.requires_grad = False
                froze += 1
        print(f"[freeze] regex='{args.freeze_regex}', froze_params={froze}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[params] trainable tensors: {len(trainable)} / total: {len(list(model.parameters()))}")

    # ---- optimizer / scheduler ----
    opt = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    sched = None
    batch_sched = False
    if args.scheduler == "Cosine":
        # epoch-wise cosine, stable for fine-tuning
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.min_lr)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- train loop ----
    best_loss = float("inf")
    best_psnr = -float("inf")
    t_all = time.time()
    
    def assert_finite(x, name):
        if not torch.isfinite(x).all():
            bad = (~torch.isfinite(x)).sum().item()
            mn = torch.nanmin(x).item()
            mx = torch.nanmax(x).item()
            raise RuntimeError(f"{name} has non-finite values: count={bad}, nanmin={mn}, nanmax={mx}")
            
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_t = train_one_epoch(
            model, train_loader, opt, scaler, device,
            loss_fn=l1_loss, sched=sched, batch_sched=batch_sched,
            grad_clip=args.grad_clip,
        )
        val_loss, val_psnr, val_t = validate(model, valid_loader, device, loss_fn=l1_loss)

        if sched is not None and (not batch_sched):
            sched.step()

        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[{args.scale}] epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.2e} | "
            f"train L1={tr_loss:.6f} ({tr_t:.1f}s) | "
            f"val L1={val_loss:.6f}, PSNR={val_psnr:.2f}dB ({val_t:.1f}s)"
        )

        # save best by val_loss (main), and track PSNR too
        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_loss,
                    "val_psnr": val_psnr,
                    "args": vars(args),
                },
                f"best_swinir_finetune_{args.scale}.pt",
            )

        # optional: also save best PSNR checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_psnr": best_psnr,
                    "val_loss": val_loss,
                    "args": vars(args),
                },
                f"bestpsnr_swinir_finetune_{args.scale}.pt",
            )

    print(f"[time] total: {fmt(time.time() - t_all)}")
    print(f"[done] best_val_loss={best_loss:.6f}, best_val_psnr={best_psnr:.2f} dB")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()