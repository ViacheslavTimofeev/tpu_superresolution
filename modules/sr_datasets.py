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

RX_Z = re.compile(r"_z(\d{4})\.(?:tif|tiff|png)$", re.IGNORECASE)
RX_G = re.compile(r"_g(\d{4})\.(?:tif|tiff|png)$", re.IGNORECASE)
EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _open_pil(p: Path) -> Image.Image:
    with Image.open(p) as im:
        return im.copy()

def _get_dirs_deeprock(root: str, split: str, scale: str) -> Tuple[Path, Path]:
    root = Path(root)
    """
    hr_dir = Path("C:/Users/Вячеслав/Documents/superresolution/DeepRockSR-2D_patches") / f"HR_{split}"
    lr_dir = Path("C:/Users/Вячеслав/Documents/superresolution/DeepRockSR-2D_patches") / f"LR_{split}"
    """
    hr_dir = Path("C:/Users/Вячеслав/Documents/superresolution/beton_dataset") / f"beton_{split}_HR"
    lr_dir = Path("C:/Users/Вячеслав/Documents/superresolution/beton_dataset") / f"beton_{split}_LR_default_{scale}"
    """
    hr_dir = root / "shuffled2D" / f"shuffled2D_{split}_HR"
    lr_dir = root / "shuffled2D" / f"shuffled2D_{split}_LR_default_{scale}"
    """
    """
    hr_dir = root / "carbonate2D" / f"carbonate2D_{split}_HR_micro"
    lr_dir = root / "carbonate2D" / f"carbonate2D_{split}_LR_default_{scale}_micro"
    """
    if not (hr_dir.exists() and lr_dir.exists()):
        raise FileNotFoundError(f"Не найдены HR/LR директории для split={split}, scale={scale}")
    return hr_dir, lr_dir

def _strip_lr_suffix(stem: str, scale: str) -> str:
    suf = scale.lower()
    if not suf.startswith('x'):
        suf = 'x' + suf
    # убираем хвост "_x2"/"-x2"/"x2"
    return re.sub(fr'([_-]?){re.escape(suf)}$', '', stem, flags=re.IGNORECASE)

def _get_dirs_deeprock_patches(root: str, split: str, scale: str | None = None) -> Tuple[Path, Path]:
    """
    Для патчей, сохранённых диском:

        root/
          HR_train/
          HR_valid/
          LR_train/
          LR_valid/

    split ∈ {"train", "valid"}.
    scale сейчас не используется, оставлен для интерфейсной совместимости.
    """
    root = Path(root)

    if split == "train":
        hr_dir = root / "HR_train"
        lr_dir = root / "LR_train"
    elif split == "valid":
        hr_dir = root / "HR_valid"
        lr_dir = root / "LR_valid"
    else:
        raise ValueError(f"Unknown split={split!r} for deeprock_patches")

    if not hr_dir.exists():
        raise FileNotFoundError(f"Нет папки с HR-патчами: {hr_dir}")
    if not lr_dir.exists():
        raise FileNotFoundError(f"Нет папки с LR-патчами: {lr_dir}")

    return hr_dir, lr_dir


def _strip_patch_suffix(stem: str) -> str:
    """
    Убираем суффиксы _HR / _LR в конце имени патча.
    Например:
      'block01_p000010_HR' -> 'block01_p000010'
      'block01_p000010_LR' -> 'block01_p000010'
    """
    return re.sub(r"_(HR|LR)$", "", stem, flags=re.IGNORECASE)

def compute_patch_grid(h: int, w: int, patch_size: int):
    """
    Возвращает список координат (top, left) для патчей patch_size × patch_size,
    так что с padding мы покрываем всю картинку.
    """
    import math

    n_h = math.ceil(h / patch_size)  # сколько патчей по вертикали
    n_w = math.ceil(w / patch_size)  # сколько по горизонтали

    grid = []
    for i in range(n_h):
        for j in range(n_w):
            top = i * patch_size
            left = j * patch_size
            grid.append((top, left))
    return grid


def pad_to_multiple(img: torch.Tensor, patch_size: int, mode: str = "reflect"):
    # img: [C, H, W]
    _, h, w = img.shape
    import math
    n_h = math.ceil(h / patch_size)
    n_w = math.ceil(w / patch_size)
    H_pad = n_h * patch_size
    W_pad = n_w * patch_size
    pad_h = H_pad - h
    pad_w = W_pad - w

    # разнести паддинг симметрично
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # F.pad ожидает порядок (left, right, top, bottom)
    img_padded = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return img_padded, (pad_left, pad_right, pad_top, pad_bottom)


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

class DeepRockDiskPatchPairs(Dataset):
    """
    Датасет для дисковых патчей DeepRock:

        data_root/
          HR_train/
          HR_valid/
          LR_train/
          LR_valid/

    split определяет какую пару директорий использовать.
    Пары LR↔HR сопоставляются по базовому имени без суффиксов _HR/_LR.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        scale: str = "X4",
        transform_pair: Optional[Callable] = None,
        exts: Tuple[str, ...] = EXTS,
    ):
        self.hr_dir, self.lr_dir = _get_dirs_deeprock_patches(root, split, scale)
        self.exts = exts
        self.transform_pair = transform_pair

        hr_files = sorted([p for p in self.hr_dir.iterdir() if p.suffix.lower() in exts])
        if not hr_files:
            raise RuntimeError(f"Нет HR-патчей в {self.hr_dir}")
        hr_map = {_strip_patch_suffix(p.stem): p for p in hr_files}

        lr_files = sorted([p for p in self.lr_dir.iterdir() if p.suffix.lower() in exts])
        if not lr_files:
            raise RuntimeError(f"Нет LR-патчей в {self.lr_dir}")
        lr_map = {_strip_patch_suffix(p.stem): p for p in lr_files}

        common_keys = sorted(set(hr_map.keys()) & set(lr_map.keys()))
        if not common_keys:
            raise RuntimeError("Не найдено пар LR↔HR по базовым именам (_HR/_LR).")

        self.pairs = [(lr_map[k], hr_map[k]) for k in common_keys]

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _open_raw(p: Path) -> Image.Image:
        with Image.open(p) as img:
            return img.copy()

    def __getitem__(self, idx: int):
        lr_path, hr_path = self.pairs[idx]
        lr = self._open_raw(lr_path)
        hr = self._open_raw(hr_path)

        if self.transform_pair is not None:
            lr_t, hr_t = self.transform_pair(lr, hr)
            return lr_t, hr_t

        # fallback: ч/б [0,1]
        hr = TF.to_dtype(TF.to_image(hr.convert("L")), torch.float32, scale=True)
        lr = TF.to_dtype(TF.to_image(lr.convert("L")), torch.float32, scale=True)
        return lr, hr


class MRCCMPairedByZ(Dataset):
    """
    Пары из двух папок: LR_dir и HR_dir (например, LR_train/HR_train или LR_test/HR_test).
    Сопоставление по глобальному индексу: сначала _gNNNN, иначе _zNNNN.
    """
    def __init__(
        self,
        lr_dir: str | Path,
        hr_dir: str | Path,
        transform_pair: Optional[Callable] = None,
        stride: int = 1,
    ):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.transform_pair = transform_pair
        self.stride = max(1, int(stride))

        if not self.lr_dir.is_dir():
            raise FileNotFoundError(f"Нет папки LR: {self.lr_dir}")
        if not self.hr_dir.is_dir():
            raise FileNotFoundError(f"Нет папки HR: {self.hr_dir}")

        # индексы файлов → путь
        lr_idx: Dict[int, Path] = {}
        for p in self.lr_dir.iterdir():
            if p.is_file() and p.suffix.lower() in EXTS:
                m = RX_Z.search(p.name)
                if m: lr_idx[int(m.group(1))] = p

        hr_idx: Dict[int, Path] = {}
        for p in self.hr_dir.iterdir():
            if p.is_file() and p.suffix.lower() in EXTS:
                mg = RX_G.search(p.name)
                if mg:
                    hr_idx[int(mg.group(1))] = p
                else:
                    mz = RX_Z.search(p.name)
                    if mz: hr_idx[int(mz.group(1))] = p

        keys = sorted(set(lr_idx.keys()) & set(hr_idx.keys()))
        if not keys:
            raise RuntimeError("Не найдено пересечение ключей LR и HR по _g/_z индексам.")

        pairs = [(k, lr_idx[k], hr_idx[k]) for k in keys]
        pairs.sort(key=lambda t: t[0])
        self.pairs = [(lr, hr) for _, lr, hr in pairs][::self.stride]

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        lr_path, hr_path = self.pairs[idx]
        lr = _open_pil(lr_path)
        hr = _open_pil(hr_path)
        if self.transform_pair:
            lr, hr = self.transform_pair(lr, hr)
        return lr, hr


class DeepRockPatchIterable(IterableDataset):
    """
    Iterable-датасет для DeepRock (shuffled2D_*), который:
      - для каждой эпохи один раз проходит по всем LR/HR файлам,
      - для каждого файла: читает -> transform_pair -> паддит -> режет на патчи,
      - ничего не кэширует (кроме размеров для __len__),
      - полностью покрывает изображения патчами, как у Buono.

    Важно:
      - transform_pair должен быть без RandomCrop (patch_size=None в build_pair_transform).
      - Все изображения DeepRock считаем одинакового размера (500x500), чтобы корректно оценить __len__.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        scale: str = "X2",
        patch_size: int = 100,
        transform_pair: Optional[Callable] = None,
        pad_mode: str = "reflect",
        shuffle_images: bool = True,
        shuffle_patches: bool = False,
    ):
        super().__init__()
        self.patch_size = int(patch_size)
        self.pad_mode = pad_mode
        self.transform_pair = transform_pair
        self.shuffle_images = shuffle_images
        self.shuffle_patches = shuffle_patches

        # 1) собираем пары путей, как в Shuffled2DPaired
        hr_dir, lr_dir = _get_dirs_deeprock(root, split, scale)
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

        hr_files = sorted([p for p in hr_dir.iterdir() if p.suffix.lower() in exts])
        if not hr_files:
            raise RuntimeError(f"Нет HR-файлов в {hr_dir}")
        hr_map = {p.stem: p for p in hr_files}

        lr_files = sorted([p for p in lr_dir.iterdir() if p.suffix.lower() in exts])
        pairs_paths: List[Tuple[Path, Path]] = []
        for p in lr_files:
            hr_stem = _strip_lr_suffix(p.stem, scale)
            hr_p = hr_map.get(hr_stem)
            if hr_p is not None:
                pairs_paths.append((p, hr_p))
        if not pairs_paths:
            raise RuntimeError("Не найдено пар LR↔HR по совпадающим именам файлов.")

        self.pairs_paths = pairs_paths

        # 2) Оцениваем, сколько патчей на одно изображение (чтобы работал __len__)
        #    Открываем ОДНУ пару, прогоняем через transform -> pad -> считаем n_h, n_w.
        lr_sample, hr_sample = self._load_and_transform_pair(self.pairs_paths[0])
        hr_sample, _ = pad_to_multiple(hr_sample, self.patch_size, mode=self.pad_mode)
        _, H_pad, W_pad = hr_sample.shape
        n_h = H_pad // self.patch_size
        n_w = W_pad // self.patch_size
        self._patches_per_image = n_h * n_w
        self._length = len(self.pairs_paths) * self._patches_per_image

    def _load_and_transform_pair(self, pair_paths: Tuple[Path, Path]):
        lr_path, hr_path = pair_paths
        lr_pil = _open_pil(lr_path)
        hr_pil = _open_pil(hr_path)

        if self.transform_pair is not None:
            lr_t, hr_t = self.transform_pair(lr_pil, hr_pil)
        else:
            # минимальный дефолт: Grayscale -> resize LR до HR -> ToTensor [0,1]
            hr_pil_g = hr_pil.convert("L")
            lr_pil_g = lr_pil.convert("L").resize(hr_pil_g.size, Image.BICUBIC)

            hr_t = TF.to_image(hr_pil_g)              # [1,H,W], uint8
            lr_t = TF.to_image(lr_pil_g)              # [1,H,W], uint8
            hr_t = TF.to_dtype(hr_t, torch.float32, scale=True)  # [0,1]
            lr_t = TF.to_dtype(lr_t, torch.float32, scale=True)
        return lr_t, hr_t

    def __len__(self) -> int:
        # число патчей за одну "полную" эпоху
        return self._length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # один процесс
            indices = list(range(len(self.pairs_paths)))
        else:
            # делим список файлов между воркерами
            per_worker = int(math.ceil(len(self.pairs_paths) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.pairs_paths))
            indices = list(range(start, end))

        rng = random.Random()  # локальный RNG на воркера

        if self.shuffle_images:
            rng.shuffle(indices)

        ps = self.patch_size

        for img_idx in indices:
            lr_t, hr_t = self._load_and_transform_pair(self.pairs_paths[img_idx])

            # паддинг до кратности patch_size
            hr_t, _ = pad_to_multiple(hr_t, ps, mode=self.pad_mode)
            lr_t, _ = pad_to_multiple(lr_t, ps, mode=self.pad_mode)

            _, H_pad, W_pad = hr_t.shape
            n_h = H_pad // ps
            n_w = W_pad // ps

            coords = [(i * ps, j * ps) for i in range(n_h) for j in range(n_w)]
            if self.shuffle_patches:
                rng.shuffle(coords)

            for top, left in coords:
                lr_patch = lr_t[..., top:top + ps, left:left + ps]
                hr_patch = hr_t[..., top:top + ps, left:left + ps]
                yield lr_patch, hr_patch


class DeepRockPrecomputedPatches(Dataset):
    def __init__(self, path: str):
        data = torch.load(path, map_location="cpu")
        self.lr = data["lr"]  # uint8 [N,1,ps,ps]
        self.hr = data["hr"]

    def __len__(self):
        return self.lr.shape[0]

    def __getitem__(self, idx):
        lr = self.lr[idx].float() / 255.0
        hr = self.hr[idx].float() / 255.0
        return lr, hr

class MSResUNetPNGDataset(Dataset):
    """
    Для каждого файла PNG из директории:
      - читаем его как HR (внутри preprocess_msresunet_png);
      - строим LR через down+up bicubic с заданным upscale_factor;
      - переводим обе картинки в YCbCr и берём только Y-канал;
      - делим на 255 → тензоры (1, H, W) с диапазоном примерно [0.06, 0.92].

    Параметры
    ---------
    root : str | Path
        Путь к директории, где лежат HR PNG-изображения.
        Используются только файлы с расширением ".png".
    upscale_factor : int
        Во сколько раз HR больше "идеального" LR по каждой оси
        (обычно 2 или 4).
    image_size : Optional[int]
        Если задано, HR изображение предварительно приводится к
        квадрату image_size x image_size bicubic-интерполяцией
        (без кропа). Если None — используется исходный размер PNG.
    """

    def __init__(
        self,
        root: str | Path,
        upscale_factor: int
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.upscale_factor = int(upscale_factor)

        if not self.root.is_dir():
            raise FileNotFoundError(f"Директория с данными не найдена: {self.root}")

        # Берём только PNG, т.к. ты говоришь, что исходные данные именно PNG
        self.files: List[Path] = sorted(
            [p for p in self.root.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        )

        if not self.files:
            raise RuntimeError(f"В директории {self.root} не найдено ни одного PNG-файла.")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]

        # preprocess_msresunet_png:
        #   PNG -> RGB ->
        #   LR = down+up bicubic ->
        #   обе в YCbCr -> Y-канал -> /255 -> (1,H,W)
        lr_y, hr_y = preprocess_msresunet_png(
            path=str(path),
            upscale_factor=self.upscale_factor
        )
        return lr_y, hr_y