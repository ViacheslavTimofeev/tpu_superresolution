# train.py
import os, time, math, argparse, random, re
from pathlib import Path
from typing import Tuple, Optional, Callable

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim as ssim_ms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
import torch.nn.functional as F

from unet2d import UNet2D, UNetConfig  # твоя модель

import copy
from datetime import timedelta

from sr_transforms import build_pair_transform
from sr_datasets import Shuffled2DPaired, MRCCMPairedByZ, DeepRockPatchIterable#DeepRockPatchDataset

V_MIN = 8557.25
V_MAX = 47033.0

V_MIN_05 = 11072.5
V_MAX_995 = 18645.0

def fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))
    
# ====== 1) Настройки reproducibility ======
def seed_everything(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # для фиксированных размеров быстрее

# ====== loss and metrics ======
def L1_loss(pred, target):
    return F.l1_loss(pred, target)

def mse_loss(pred, target):
    return F.mse_loss(pred, target)
    
def batch_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """
    PSNR по батчу, на выходе тензор [B] в dB.
    Ожидается диапазон [0,1], на всякий случай clamp.
    """
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target, reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # [B]
    psnr = 20.0 * torch.log10(max_val / torch.sqrt(mse + 1e-8))
    return psnr

class ComboLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_ssim=0.05):
        super().__init__()
        self.w_l1 = float(w_l1)
        self.w_ssim = float(w_ssim)

    def forward(self, pred, target):
        # Оба уже в [0,1] (Normalize отключен) — считаем L1 прямо так
        l1 = F.l1_loss(pred, target)

        loss = self.w_l1 * l1

        # Считаем SSIM только если он реально нужен
        if self.w_ssim > 0:
            # SSIM иногда "плывёт", если выход чуть вылез за [0,1].
            # КЛИПИМ ТОЛЬКО КОПИИ для SSIM, L1 не трогаем:
            #pred01 = pred.detach()  # чтобы не тащить лишний grad через clamp
            #targ01 = target.detach()
            pred01 = pred.clamp(0, 1)
            targ01 = target.clamp(0, 1)
            with torch.amp.autocast("cuda", enabled=False):
                ssim_val = ssim_ms(pred01.float(), targ01.float(), data_range=1.0)
            # защитимся от редких NaN:
            if torch.isnan(ssim_val) or torch.isinf(ssim_val):
                ssim_val = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
            loss = loss + self.w_ssim * (1.0 - ssim_val)

        return loss

# ====== 5) DataLoader ======
def make_loader(ds, batch_size, workers, pin=True, shuffle=False, drop_last=False, persistent=False):
    kwargs = dict(
        dataset=ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        num_workers=workers, pin_memory=pin, persistent_workers=persistent,
    )
    if workers and workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)

# ====== 6) Профилирование загрузки ======
@torch.no_grad()
def warmup_profile(dl, n_batches=3):
    t0 = time.time()
    for i, (lr, hr) in enumerate(dl):
        if i == 0:
            print(f"[profile] first batch load: {time.time()-t0:.2f}s")
        if i + 1 >= n_batches: break
    print(f"[profile] {n_batches} batches load: {time.time()-t0:.2f}s")

# ====== 7) Тренировка/валидация ======
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, loss_fn, sched):
    model.train()
    data_t = 0.0
    step_t = 0.0
    n_steps = 0
    end = time.time()
    total_loss = 0.0

    for it, (lr, hr) in enumerate(loader):
        # время на загрузку батча
        data_time = time.time() - end

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            #lr = torch.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
            #hr = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)

            out = model(lr)
            if not torch.isfinite(out).all():
                raise RuntimeError(
                    "Model produced NaN/Inf — понизь max_lr, проверь residual_scale/init"
                )

            loss = loss_fn(out, hr)

        # обычный backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        sched.step()
        scaler.update()

        batch_time = time.time() - t0

        total_loss += loss.item()
        data_t += data_time
        step_t += batch_time
        n_steps += 1
        end = time.time()

    mean_loss = total_loss / max(1, n_steps)
    mean_data_t = data_t / max(1, n_steps)
    mean_step_t = step_t / max(1, n_steps)

    return mean_loss, mean_data_t, mean_step_t


@torch.no_grad()
def validate(model, loader, device, loss_fn):
    model.eval()
    tot = 0.0; n = 0

    sum_psnr = 0.0
    sum_ssim = 0.0
    n_imgs = 0

    for it, (lr, hr) in enumerate(loader):
        if it == 0:
            print(f"[val] batch0 shapes: lr={tuple(lr.shape)}, hr={tuple(hr.shape)}, "
                  f"numel(lr)={lr.numel()}, numel(hr)={hr.numel()}")
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            lr = torch.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
            hr = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)
            out = model(lr)
            loss = loss_fn(out, hr)

        # метрики
        psnr_vals = batch_psnr(out, hr)
        sum_psnr += psnr_vals.sum().item()

        pred01 = out.clamp(0, 1).float()
        targ01 = hr.clamp(0, 1).float()
        with torch.amp.autocast("cuda", enabled=False):
            ssim_val = ssim_ms(pred01, targ01, data_range=1.0)
        sum_ssim += ssim_val.item() * lr.size(0)

        n_imgs += lr.size(0)

        tot += loss.item(); n += 1

    mean_loss = tot / max(1, n)
    mean_psnr = sum_psnr / max(1, n_imgs)
    mean_ssim = sum_ssim / max(1, n_imgs)
    return mean_loss, mean_psnr, mean_ssim


# ====== 8) Основной запуск ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--epochs_x2", type=int, default=5)
    ap.add_argument("--epochs_x4", type=int, default=0, help="Эпох для X4 (0 = пропустить фазу X4)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pin", action="store_true")
    ap.add_argument("--no_persistent", action="store_true")
    ap.add_argument("--time_log_every", type=int, default=20, help="Как часто писать время обучения (в эпохах). 0 = не логировать")
    ap.add_argument("--data_layout", type=str, choices=["DeepRock", "mrccm"], default="mrccm",
                    help="Где лежат данные: старая структура DeepRock или новый экспорт MRCCM (LR + ILS*/HR)")
    ap.add_argument("--z_start", type=int, default=None)
    ap.add_argument("--z_end", type=int, default=None)
    ap.add_argument("--z_stride", type=int, default=1)

    args = ap.parse_args()

    seed_everything(args.seed)
    t_all_start = time.time()
    
    # Автовыбор воркеров
    if args.workers is None:
        cpu = os.cpu_count() or 4
        args.workers = min(2, cpu) if os.name == "nt" else min(8, max(2, cpu // 2))
    print(f"[cfg] workers={args.workers}, pin={not args.no_pin}, persistent={not args.no_persistent}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "-")

    # === Трансформы ===
    pair_tf_train = build_pair_transform(
        patch_size=None,
        do_flips=False,
        do_blur=False,
        dataset=args.data_layout,
        vmin=V_MIN,
        vmax=V_MAX,
        normalize=False
    )
    pair_tf_valid = build_pair_transform(
        patch_size=None,       # без кропа, целые кадры
        do_flips=False,
        do_blur=False,
        dataset=args.data_layout,
        vmin=V_MIN,
        vmax=V_MAX,
        normalize=False
    )

    # === Датасеты/лоадеры X2 ===
    if args.data_layout == "mrccm":
    # ожидается структура:
    # data_root/
    #   LR_train/, HR_train/
    #   LR_test/,  HR_test/
        base_train_ds_x2 = MRCCMPairedByZ(Path(args.data_root) / "LR_train",
                                     Path(args.data_root) / "HR_train",
                                     transform_pair=pair_tf_train,
                                     stride=args.z_stride)
        valid_ds_x2 = MRCCMPairedByZ(Path(args.data_root) / "LR_test",
                                     Path(args.data_root) / "HR_test",
                                     transform_pair=pair_tf_valid,
                                     stride=max(1, args.z_stride//2))
    elif args.data_layout == "DeepRock":
        train_ds_x2 = DeepRockPatchIterable(
        root=args.data_root,
        split="train",
        scale="X2",
        patch_size=args.patch_size,      # например, 128
        transform_pair=pair_tf_train,    # БЕЗ кропа
        pad_mode="reflect",
        shuffle_images=True,
        shuffle_patches=False,
    )
        valid_ds_x2 = Shuffled2DPaired(args.data_root, "valid", "X2", transform_pair=pair_tf_valid)
    else:
        raise ValueError(f"Unknown data_layout: {args.data_layout}")
    
    train_loader_x2 = make_loader(
        train_ds_x2, args.batch_size, args.workers,
        pin=not args.no_pin, shuffle=False, drop_last=False, persistent=not args.no_persistent
    )
    valid_loader_x2 = make_loader(
        valid_ds_x2, max(1, args.batch_size//2), args.workers,
        pin=not args.no_pin, shuffle=False, drop_last=False, persistent=not args.no_persistent
    )
    
    loss_fn = mse_loss
    #loss_fn = L1_loss
    #loss_fn = ComboLoss(w_l1=1, w_ssim=0.1)#, window_size=11, sigma=1.5
    # === Модель X2 ===
    # UNet2D должен выдавать такое же пространственное разрешение, как вход.
    # Мы уже апскейлим LR -> размера HR в трансформе, поэтому in_ch=out_ch=1.
    cfg = UNetConfig(
        in_channels=1, out_channels=1,
        base_channels=64, depth=4,
        norm_enc=False, norm_dec=False,
        up_mode="pixelshuffle"
    )
    model = UNet2D(cfg).to(device)
    
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=opt, max_lr=5e-4,
        steps_per_epoch=len(train_loader_x2), epochs=args.epochs_x2,
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print("\n[profile X2 loader]")
    warmup_profile(train_loader_x2, n_batches=3)
    
    best = math.inf
    t_x2_start = time.time()
    for epoch in range(1, args.epochs_x2 + 1):
        tr_loss, d_t, b_t = train_one_epoch(
            model, train_loader_x2, opt, scaler, device, epoch, loss_fn, sched
        )
        val_loss, val_psnr, val_ssim = validate(model, valid_loader_x2, device, loss_fn)

        print(
            f"[X2] epoch {epoch}: "
            f"train_loss {tr_loss:.7f}, val_loss {val_loss:.7f} | "
            f"val_PSNR {val_psnr:.2f} dB, val_SSIM {val_ssim:.4f} | "
            f"(data {d_t:.3f}/batch {b_t:.3f})"
        )
        
        if args.time_log_every and (epoch % args.time_log_every == 0 or epoch == 1):
            elapsed = time.time() - t_x2_start
            # простая оценка ETA по средней длительности эпохи
            avg_per_epoch = elapsed / epoch
            remain_epochs = args.epochs_x2 - epoch
            eta = avg_per_epoch * remain_epochs
            print(f"[X2][time] elapsed={fmt(elapsed)} | avg/epoch={fmt(avg_per_epoch)} | ETA≈{fmt(eta)}")
    
        if val_loss < best:
            best = val_loss
            torch.save({"model": model.state_dict()}, "best_x2.pt")
    
    print(f"[X2][time] total={fmt(time.time() - t_x2_start)}")
    
    # === Переход на X4 (дообучение той же модели) ===
    if args.epochs_x4 > 0 and args.data_layout == "DeepRock":
        train_x4 = Shuffled2DPaired(args.data_root, "train", "X4", transform_pair=pair_tf_train)
        valid_x4 = Shuffled2DPaired(args.data_root, "valid", "X4", transform_pair=pair_tf_valid)
    
        train_loader_x4 = make_loader(
            train_x4, args.batch_size, args.workers,
            pin=not args.no_pin, shuffle=True, drop_last=True, persistent=not args.no_persistent
        )
        valid_loader_x4 = make_loader(
            valid_x4, max(1, args.batch_size//2), args.workers,
            pin=not args.no_pin, shuffle=False, drop_last=False, persistent=not args.no_persistent
        )
    
        if args.profile_batches > 0:
            print("\n[profile X4 loader]")
            warmup_profile(train_loader_x4, n_batches=args.profile_batches)
    
        # (опционально) доинициализироваться весами X2
        if Path("best_x2.pt").exists():
            ckpt = torch.load("best_x2.pt", map_path="cpu") if hasattr(torch, "load") else torch.load("best_x2.pt", map_location="cpu")
            model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
            print("[X4] loaded weights from best_x2.pt")
    
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched_x4 = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=opt, max_lr=5e-4,
            steps_per_epoch=len(train_loader_x4), epochs=args.epochs_x4,
            pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
        )
        # таймер для X4 (если используешь)
        t_x4_start = time.time()
    
        best = math.inf
        for epoch in range(1, args.epochs_x4 + 1):
            tr_loss, d_t, b_t = train_one_epoch(
                model, train_loader_x4, opt, scaler, device, epoch, loss_fn, sched_x4
            )
            val_loss, val_psnr, val_ssim = validate(model, valid_loader_x4, device, loss_fn)
            print(
                f"[X4] epoch {epoch}: "
                f"train_loss {tr_loss:.7f}, val_loss {val_loss:.7f} | "
                f"val_PSNR {val_psnr:.2f} dB, val_SSIM {val_ssim:.4f} | "
                f"(data {d_t:.3f}/batch {b_t:.3f})"
            )
    
            if args.time_log_every and (epoch % args.time_log_every == 0 or epoch == 1):
                elapsed = time.time() - t_x4_start
                avg_ep = elapsed / epoch
                eta = avg_ep * (args.epochs_x4 - epoch)
                print(f"[X4][time] elapsed={fmt(elapsed)} | avg/epoch={fmt(avg_ep)} | ETA≈{fmt(eta)}")
    
            if val_loss < best:
                best = val_loss
                torch.save({"model": model.state_dict()}, "best_x4.pt")

        print(f"[X4][time] total={fmt(time.time() - t_x4_start)}")
    else:
        print("[X4] skipped (either epochs_x4=0 or data_layout=mrccm)")
     

    print(f"[ALL][time] total train time={fmt(time.time() - t_all_start)}")
    
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)  # Windows-safe
    except RuntimeError:
        pass
    main()