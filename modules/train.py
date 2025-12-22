"""
Универсальный модуль обучения моделей, поддерживает разные форматы и размеры данных.
"""
import os, time, math, argparse, random, re
from pathlib import Path
from typing import Tuple, Optional, Callable
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_msssim import ssim as ssim_ms
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
import torch.nn.functional as F

from unet2d import UNet2D, UNetConfig
from ms_resunet import MS_ResUNet

import copy
from datetime import timedelta

from sr_transforms import build_pair_transform
from sr_datasets import Shuffled2DPaired, MRCCMPairedByZ, DeepRockPatchIterable, DeepRockPrecomputedPatches, DeepRockDiskPatchPairs

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

# ====== 5) DataLoader ======
def make_loader(ds, batch_size, workers, pin=True, shuffle=False, drop_last=False, persistent=False):
    is_iterable = isinstance(ds, IterableDataset)

    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=(False if is_iterable else shuffle),  # для IterableDataset shuffle запрещён
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=pin,
    )

    # persistent_workers для IterableDataset обычно не нужен
    if workers and workers > 0 and (not is_iterable):
        kwargs["persistent_workers"] = persistent
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
def train_one_epoch(model, loader, optimizer, scaler, device, epoch, loss_fn, sched, is_batch_sched):
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
            out = model(lr)
            if not torch.isfinite(out).all():
                raise RuntimeError(
                    "Model produced NaN/Inf — понизь max_lr, проверь residual_scale/init"
                )

            loss = loss_fn(out, hr)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        if is_batch_sched and sched is not None:
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
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--sign", type=str, choices=["deeprock_x2", "deeprock_x4", "deeprock_patches_x4", "mrccm"],
                help="Конфигурация набора данных")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--scheduler", type=str, choices=["OneCycle", "Exponential", "None"], default="None")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--loss", type=str, choices=["mse", "l1"], default="mse")
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--weight_decay", type=float, default=0)
    ap.add_argument("--do_flips", type=bool, default=False)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pin", action="store_true")
    ap.add_argument("--no_persistent", action="store_true")
    ap.add_argument("--time_log_every", type=int, default=20)
    ap.add_argument("--z_start", type=int, default=None)
    ap.add_argument("--z_end", type=int, default=None)
    ap.add_argument("--z_stride", type=int, default=1)
    ap.add_argument("--precomputed_patches", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None,
                help="Путь к .pt с ключом 'model' для дообучения")
    ap.add_argument("--patch_iter", action="store_true",
                help="Патчевое обучение на DeepRock через DeepRockPatchIterable (полное покрытие патчами).")
    ap.add_argument("--finetune", action="store_true",
                help="Использовать чекпоинт как инициализацию (fine-tune), "
                     "не восстанавливая оптимизатор/шедулер.")
    ap.add_argument("--freeze_regex", type=str, default=None,
                    help="Регулярка по имени параметров для заморозки, например 'inc|down'.")
    ap.add_argument("--ft_lr", type=float, default=None,
                    help="Отдельный learning rate для fine-tuning (если не задан, используется --lr).")

    args = ap.parse_args()

    # --- seed + device ---
    seed_everything(args.seed)
    t_all_start = time.time()

    if args.workers is None:
        cpu = os.cpu_count() or 4
        args.workers = min(2, cpu) if os.name == "nt" else min(8, max(2, cpu // 2))
    print(f"[cfg] workers={args.workers}, pin={not args.no_pin}, persistent={not args.no_persistent}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "-")

    # --- выбор лосса ---
    if args.loss == "mse":
        loss_fn = mse_loss
    elif args.loss == "l1":
        loss_fn = L1_loss
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # --- декодируем sign -> (dataset_kind, scale) ---
    if args.sign in ("deeprock_x2", "deeprock_x4", "deeprock_patches_x4"):
        dataset_kind = "DeepRock"
        if args.sign == "deeprock_x2":
            scale = "X2"
        else:
            scale = "X4"  # и для обычного x4, и для патчевого x4
    elif args.sign == "mrccm":
        dataset_kind = "mrccm"
        scale = None
    else:
        raise ValueError(f"Unknown sign: {args.sign}")
        
    # для патчевого режима не делаем RandomCrop в трансформах
    # --- трансформы зависят от dataset_kind и патчевого режима ---

    # для дисковых патчей и iterable-патчей RandomCrop в трансформах не нужен
    if args.sign == "deeprock_patches_x4" or (dataset_kind == "DeepRock" and args.patch_iter):
        patch_size_train = None
    else:
        patch_size_train = 96  # можешь заменить на args.patch_size, если хочешь управлять из CLI

    pair_tf_train = build_pair_transform(
        patch_size=patch_size_train,
        do_flips=args.do_flips,
        dataset=dataset_kind,
        vmin=V_MIN,
        vmax=V_MAX,
        normalize=False,
    )

    pair_tf_valid = build_pair_transform(
        do_flips=False,
        dataset=dataset_kind,
        vmin=V_MIN,
        vmax=V_MAX,
        normalize=False,
    )


    if dataset_kind == "DeepRock":
        if args.sign == "deeprock_patches_x4":
            # Патчевое обучение с готовых патчей на диске
            train_ds = DeepRockDiskPatchPairs(
                root=args.data_root,
                split="train",
                scale=scale,
                transform_pair=pair_tf_train,
            )
            valid_ds = DeepRockDiskPatchPairs(
                root=args.data_root,
                split="valid",
                scale=scale,
                transform_pair=pair_tf_valid,
            )
        else:
            # Обычный DeepRock: либо precomputed .pt, либо Iterable-патчи, либо full-size Shuffled
            if args.precomputed_patches is not None:
                train_ds = DeepRockPrecomputedPatches(args.precomputed_patches)

            elif args.patch_iter:
                # DeepRockPatchIterable 
                train_ds = DeepRockPatchIterable(
                    root=args.data_root,
                    split="train",
                    scale=scale,
                    patch_size=args.patch_size,    # 100 для патчей 100x100
                    transform_pair=pair_tf_train,  # без RandomCrop
                    pad_mode="reflect",
                    shuffle_images=True,
                    shuffle_patches=False,          # можешь включить True, если хочешь
                )

            else:
                # классический случай: одна картинка = один sample
                train_ds = Shuffled2DPaired(
                    args.data_root,
                    split="train",
                    scale=scale,
                    transform_pair=pair_tf_train,
                )

            # Валидация — по крупным изображениям, как и раньше
            valid_ds = Shuffled2DPaired(
                args.data_root,
                split="valid",
                scale=scale,
                transform_pair=pair_tf_valid,
            )


    elif dataset_kind == "mrccm":
        # MRCCM: 2D-срезы по z
        train_ds = MRCCMPairedByZ(
            Path(args.data_root) / "LR_train",
            Path(args.data_root) / "HR_train",
            transform_pair=pair_tf_train,
            stride=args.z_stride,
            z_start=args.z_start,
            z_end=args.z_end,
        )
        valid_ds = MRCCMPairedByZ(
            Path(args.data_root) / "LR_test",
            Path(args.data_root) / "HR_test",
            transform_pair=pair_tf_valid,
            stride=max(1, args.z_stride // 2),
            z_start=args.z_start,
            z_end=args.z_end,
        )
    else:
        raise ValueError(f"Unknown dataset_kind: {dataset_kind}")

    # --- DataLoader для всех случаев одинаковый ---
    train_loader = make_loader(
        train_ds, args.batch_size, args.workers,
        pin=not args.no_pin,
        shuffle=True,
        drop_last=False,
        persistent=not args.no_persistent,
    )

    valid_loader = make_loader(
        valid_ds, max(1, args.batch_size // 2), args.workers,
        pin=not args.no_pin, shuffle=False, drop_last=False,
        persistent=not args.no_persistent,
    )

    print(f"\n[profile {args.sign} loader]")
    warmup_profile(train_loader, n_batches=3)

    # --- Модель (общая для всех sign) ---
    cfg = UNetConfig(
        in_channels=1, out_channels=1,
        base_channels=args.base_channels,  # можно вынести в args позже
        depth=4,
        norm_enc=False, norm_dec=False,
        up_mode="pixelshuffle",
    )
    #model = UNet2D(cfg).to(device)
    model = MS_ResUNet().to(device)
    
    # === Загрузка чекпоинта для дообучения ===
    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        # модель может быть сохранена как {"model": ..., "opt": ..., ...}
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"[ckpt] loaded model weights from {args.resume}")
        
    # === заморозка слоёв при fine-tune (если нужно) ===
    if args.finetune and args.freeze_regex is not None:
        pattern = re.compile(args.freeze_regex)
        for name, p in model.named_parameters():
            if pattern.search(name):
                p.requires_grad = False
        print(f"[finetune] froze params matching regex: {args.freeze_regex}")
    
    # параметры, которые вообще будут обучаться
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # lr: отдельный для fine-tune, если задан
    lr = args.ft_lr if (args.finetune and args.ft_lr is not None) else args.lr
    opt = optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay)
    #opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

     # --- scheduler ---
    sched = None
    is_batch_sched = False

    if args.scheduler == "OneCycle":
        sched = OneCycleLR(
            optimizer=opt,
            max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )
        is_batch_sched = True

    elif args.scheduler == "Exponential":
        #target_factor = 0.0625
        #gamma = target_factor ** (1.0 / args.epochs)
        #sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        drop_every = 50          # каждые 50 эпох
        decay_factor = 0.5       # хотим уменьшать в 2 раза
        
        gamma = decay_factor ** (1.0 / drop_every)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

        print(f"[sched] ExponentialLR: gamma={gamma:.6f}")

    elif args.scheduler == "None":
        sched = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")
        
    if ckpt is not None and (not args.finetune):
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
            print("[ckpt] restored optimizer state")
        if "sched" in ckpt and sched is not None:
            sched.load_state_dict(ckpt["sched"])
            print("[ckpt] restored scheduler state")

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # --- единый цикл обучения для всех sign ---
    best = math.inf
    t_start = time.time()

        # списки для логирования лоссов по эпохам
    history_train_loss = []
    history_val_loss = []

    for epoch in range(1, args.epochs + 1):
        t_ep_start = time.time()
    
        t_tr_start = time.time()
        tr_loss, d_t, b_t = train_one_epoch(
            model, train_loader, opt,
            scaler, device, epoch,
            loss_fn, sched=sched, is_batch_sched=is_batch_sched,
        )
        t_tr = time.time() - t_tr_start
    
        t_val_start = time.time()
        val_loss, val_psnr, val_ssim = validate(model, valid_loader, device, loss_fn)
        t_val = time.time() - t_val_start
        
        history_train_loss.append(tr_loss)
        history_val_loss.append(val_loss)

        t_ep = time.time() - t_ep_start
    
        print(
            f"[{args.sign}] epoch {epoch}: "
            f"train_loss {tr_loss:.7f}, val_loss {val_loss:.7f} | "
            f"val_PSNR {val_psnr:.2f} dB, val_SSIM {val_ssim:.4f} | "
            f"(data {d_t:.3f}/batch {b_t:.3f}) | "
            f"time: train {t_tr:.1f}s, val {t_val:.1f}s, total {t_ep:.1f}s"
        )

        # шаг по lr-схеме раз в эпоху (для Exponential)
        if (not is_batch_sched) and (sched is not None):
            sched.step()

        if args.time_log_every and (epoch % args.time_log_every == 0 or epoch == 1):
            elapsed = time.time() - t_start
            avg_ep = elapsed / epoch
            eta = avg_ep * (args.epochs - epoch)
            print(f"[{args.sign}][time] elapsed={fmt(elapsed)} | avg/epoch={fmt(avg_ep)} | ETA≈{fmt(eta)}")

        if val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched is not None else None,
                "epoch": epoch,
                "args": vars(args),
            }, f"best_{args.sign}.pt")

        # --- сохранение графика лоссов ---
    epochs = list(range(1, len(history_train_loss) + 1))
    fig_path = f"loss_curve_{args.sign}.png"

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history_train_loss, label="train loss")
    plt.plot(epochs, history_val_loss, label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Train vs Val loss ({args.sign})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    print(f"[plot] saved loss curves to {fig_path}")

    print(f"[{args.sign}][time] total={fmt(time.time() - t_start)}")
    print(f"[ALL][time] total train time={fmt(time.time() - t_all_start)}")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)  # Windows-safe
    except RuntimeError:
        pass
    main()