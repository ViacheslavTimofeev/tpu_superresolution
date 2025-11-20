"""
Создает .pt-файл с готовыми парами HR-LR. В теории может дать ускорение при загрузке в pytorch Dataset.
Создается без аугментаций (только с обязательными трансформами)
"""
from pathlib import Path
import torch
from PIL import Image
from sr_datasets import _get_dirs_deeprock, pad_to_multiple
from sr_transforms import build_pair_transform

def prepare_deeprock_patches(
    data_root: str,
    split: str,
    scale: str,
    patch_size: int,
    out_path: str,
):
    # те же директории, что и в DeepRockPatchIterable
    hr_dir, lr_dir = _get_dirs_deeprock(data_root, split, scale)
    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    hr_files = sorted([p for p in hr_dir.iterdir() if p.suffix.lower() in exts])
    hr_map = {p.stem: p for p in hr_files}

    lr_files = sorted([p for p in lr_dir.iterdir() if p.suffix.lower() in exts])

    pairs_paths = []
    from sr_datasets import _strip_lr_suffix
    for p in lr_files:
        hr_stem = _strip_lr_suffix(p.stem, scale)
        hr_p = hr_map.get(hr_stem)
        if hr_p is not None:
            pairs_paths.append((p, hr_p))

    print(f"Found {len(pairs_paths)} LR/HR pairs")

    # тот же пайплайн, что в train (но БЕЗ флипов/блюра)
    tf = build_pair_transform(
        do_flips=False,
        do_blur=False,
        dataset="DeepRock",
        normalize=False,
    )

    all_lr = []
    all_hr = []

    ps = patch_size

    for i, (lr_path, hr_path) in enumerate(pairs_paths):
        with Image.open(lr_path) as im_lr, Image.open(hr_path) as im_hr:
            im_lr = im_lr.copy()
            im_hr = im_hr.copy()

        lr_t, hr_t = tf(im_lr, im_hr)           # [1,H,W], float32 in [0,1]

        # паддинг до кратности patch_size
        hr_pad, _ = pad_to_multiple(hr_t, ps, mode="reflect")
        lr_pad, _ = pad_to_multiple(lr_t, ps, mode="reflect")

        _, H_pad, W_pad = hr_pad.shape
        n_h = H_pad // ps
        n_w = W_pad // ps

        for ih in range(n_h):
            for iw in range(n_w):
                top = ih * ps
                left = iw * ps
                lr_patch = lr_pad[:, top:top+ps, left:left+ps]
                hr_patch = hr_pad[:, top:top+ps, left:left+ps]

                # в uint8 для экономии места
                all_lr.append((lr_patch * 255.0).round().to(torch.uint8))
                all_hr.append((hr_patch * 255.0).round().to(torch.uint8))

        if (i+1) % 50 == 0:
            print(f"{i+1}/{len(pairs_paths)} images processed")

    lr_tensor = torch.stack(all_lr)  # [N,1,ps,ps]
    hr_tensor = torch.stack(all_hr)

    print("Final shapes:", lr_tensor.shape, hr_tensor.shape)

    torch.save({"lr": lr_tensor, "hr": hr_tensor}, out_path)
    print("Saved to", out_path)

if __name__ == "__main__":
    prepare_deeprock_patches(
        data_root="C:/Users/Вячеслав/Documents/superresolution/DeepRockSR-2D",
        split="train",
        scale="X4",
        patch_size=128,
        out_path="deeprock_x4_train_patches_ps128_u8.pt",
    )