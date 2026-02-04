from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable, Iterable, List, Tuple, Dict
import json, re
from PIL import Image
import math
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import random
import math

def _get_dirs_deeprock(root: str, split: str, scale: str) -> Tuple[Path, Path]:
    root = Path(root)

    hr_dir = root / "shuffled2D" / f"shuffled2D_{split}_HR"
    lr_dir = root / "shuffled2D" / f"shuffled2D_{split}_LR_default_{scale}"

    if not (hr_dir.exists() and lr_dir.exists()):
        raise FileNotFoundError(f"Не найдены HR/LR директории для split={split}, scale={scale}")
    return hr_dir, lr_dir

def _strip_lr_suffix(stem: str, scale: str) -> str:
    suf = scale.lower()
    if not suf.startswith('x'):
        suf = 'x' + suf
    # убираем хвост "_x2"/"-x2"/"x2"
    return re.sub(fr'([_-]?){re.escape(suf)}$', '', stem, flags=re.IGNORECASE)

class Shuffled2DPaired(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        scale: str = "X2",
        exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
        transform_pair: Optional[Callable] = None,
    ):
        self.hr_dir, self.lr_dir = _get_dirs_deeprock(root, split, scale)
        self.exts = exts
        self.transform_pair = transform_pair

        hr_files = sorted([p for p in self.hr_dir.iterdir() if p.suffix.lower() in exts])
        if not hr_files:
            raise RuntimeError(f"Нет HR-файлов в {self.hr_dir}")
        hr_map = {p.stem: p for p in hr_files}

        lr_files = sorted([p for p in self.lr_dir.iterdir() if p.suffix.lower() in exts])
        pairs = []
        for p in lr_files:
            hr_stem = _strip_lr_suffix(p.stem, scale)
            hr = hr_map.get(hr_stem)
            if hr is not None:
                pairs.append((p, hr))
        if not pairs:
            raise RuntimeError("Не найдено пар LR↔HR по совпадающим именам файлов.")
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _open_raw(p: Path) -> Image.Image:
        with Image.open(p) as img:
            return img.copy()

    def __getitem__(self, idx: int):
        lr_path, hr_path = self.pairs[idx]
        lr = self._open_raw(lr_path)  # PIL
        hr = self._open_raw(hr_path)  # PIL
        if self.transform_pair is not None:
            lr, hr = self.transform_pair(lr, hr)  # тензоры
        return lr, hr