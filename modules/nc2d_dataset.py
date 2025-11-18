# nc2d_dataset.py
"""
2D SR-пайплайн для реальных пар HR/LR в .nc:
- парсинг и матчинг блоков по одинаковым именам (HR ∩ LR)
- ленивое чтение xarray без загрузки всего объёма в RAM
- выравнивание LR->HR по масштабу/центру среза (круг керна)
- квадратный общий кроп вокруг керна, исключая серые поля/чёрные углы
- выбор случайных патчей внутри маски породы
- нормализация по маске (mean/std только по породе)
- готовый PyTorch Dataset + DataLoader

Зависимости: numpy, xarray, scikit-image, torch
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import re
import numpy as np
import xarray as xr

from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_opening, remove_small_holes, remove_small_objects, disk
from skimage.measure import label, regionprops
from skimage.transform import AffineTransform, warp

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Поиск и матчинг .nc блоков
# ---------------------------

def find_matched_blocks(hr_dir: Path, lr_dir: Path, pattern: str = "block*.nc") -> List[Tuple[Path, Path]]:
    """
    Возвращает список пар (hr_path, lr_path) для файлов, чьи ИМЕНА совпадают в обеих папках.
    Например, block00000032.nc в HR и LR.
    """
    hr_dir = Path(hr_dir)
    lr_dir = Path(lr_dir)
    hr_files = {p.name: p for p in hr_dir.glob(pattern)}
    lr_files = {p.name: p for p in lr_dir.glob(pattern)}

    common = sorted(set(hr_files).intersection(lr_files))
    pairs = [(hr_files[name], lr_files[name]) for name in common]
    if not pairs:
        raise FileNotFoundError(
            f"Не найдено совпадающих блоков по шаблону '{pattern}'. "
            f"HR={hr_dir}, LR={lr_dir}"
        )
    return pairs


# ---------------------------
# Чтение .nc и извлечение срезов
# ---------------------------
def _get_z_range(path: Path, engine: str | None, meta: NCMeta) -> tuple[int, int]:
    """
    Возвращает (z_start, z_end) включительно в ГЛОБАЛЬНЫХ индексах ядра.
    Если атрибутов нет, считаем (0, zdim-1).
    """
    ds = xr.open_dataset(path, engine=engine or "netcdf4")
    try:
        zr = ds.attrs.get("zdim_range", None)
        if zr is None:
            # fallback — весь локальный диапазон
            z_start, z_end = 0, int(ds[meta.var_name].shape[meta.dims_order_zyx[0]]) - 1
        else:
            z_start, z_end = int(zr[0]), int(zr[1])
        if z_end < z_start:
            z_start, z_end = z_end, z_start
        return z_start, z_end
    finally:
        ds.close()

Z_NAMES = {"z", "Z", "tomo_zdim", "Zdim", "zdim"}
Y_NAMES = {"y", "Y", "tomo_ydim", "Ydim", "ydim"}
X_NAMES = {"x", "X", "tomo_xdim", "Xdim", "xdim"}

from typing import Sequence

VOLUME_VAR_CANDIDATE_HINTS = (
    "tomo", "volume", "vol", "img", "data", "recon", "ct", "scan"
)

def _is_string_dtype(da) -> bool:
    # xarray.DataArray -> numpy dtype
    dt = np.dtype(da.dtype)
    # строковые/байтовые типы нам не подходят
    return dt.kind in ("S", "U", "O")  # bytes, unicode, object

def _guess_volume_var(ds: xr.Dataset,
                      prefer_var: Optional[str] = None
                      ) -> Tuple[str, Sequence[str]]:
    """
    Выбирает "основной" 3D-объём из Dataset:
    - если передан prefer_var и он существует и 3D — берём его
    - иначе фильтруем переменные по критериям:
        * ранк=3 (3 измерения)
        * не строковый dtype
        * имена измерений содержат что-то из наших Z/Y/X
    - среди кандидатов предпочитаем:
        * чьи имена содержат подсказки VOLUME_VAR_CANDIDATE_HINTS
        * с максимальным числом элементов (product of shape)
    Возвращает (var_name, dims)
    Бросает RuntimeError, если не найдено.
    """
    if prefer_var and prefer_var in ds.data_vars:
        da = ds[prefer_var]
        if da.ndim == 3 and not _is_string_dtype(da):
            return prefer_var, list(da.dims)

    candidates = []
    for name, da in ds.data_vars.items():
        if da.ndim != 3:
            continue
        if _is_string_dtype(da):
            continue
        dims = list(da.dims)
        # хотя бы одно измерение должно напоминать Z/Y/X
        if not any(d in Z_NAMES for d in dims) and not any(d in Y_NAMES for d in dims) and not any(d in X_NAMES for d in dims):
            # всё равно оставим как слабого кандидата — вдруг имена необычные
            pass
        shape = tuple(int(s) for s in da.shape)
        size = int(np.prod(shape))
        name_l = name.lower()
        score = 0
        # бонусы за «говорящее» имя
        if any(h in name_l for h in VOLUME_VAR_CANDIDATE_HINTS):
            score += 10
        # бонус за наличие знакомых имён осей
        score += 3 * sum(dim in Z_NAMES for dim in dims)
        score += 2 * sum(dim in Y_NAMES for dim in dims)
        score += 2 * sum(dim in X_NAMES for dim in dims)
        # главный критерий — побеждает самый большой объём
        candidates.append((score, size, name, dims))

    if not candidates:
        raise RuntimeError(f"Не найден подходящий 3D-объём (ранк=3) среди data_vars: {list(ds.data_vars)}")

    # сортируем по (score, size), берём лучший
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, best_name, best_dims = candidates[0]
    return best_name, best_dims

def _axis_index(dims: Iterable[str], names: set) -> Optional[int]:
    for i, d in enumerate(dims):
        if d in names:
            return i
    return None

@dataclass
class NCMeta:
    """метаданные для одного .nc файла (чтобы не вычислять каждый раз)."""
    var_name: str
    dims_order_zyx: Tuple[int, int, int]  # индексы (z,y,x) в исходных dims
    zdim: int

def _inspect_nc(path: Path, engine: str | None, prefer_var: Optional[str] = None) -> NCMeta:
    # Открываем напрямую нужным движком (например, "netcdf4")
    ds = xr.open_dataset(path, engine=engine or "netcdf4")
    try:
        var_name, dims = _guess_volume_var(ds, prefer_var=prefer_var)
        ai = [
            _axis_index(dims, Z_NAMES),
            _axis_index(dims, Y_NAMES),
            _axis_index(dims, X_NAMES),
        ]
        # fallback: если имена осей нестандартные — считаем порядок как (Z,Y,X)
        if any(a is None for a in ai):
            ai = [0, 1, 2]
        zdim = int(ds[var_name].shape[ai[0]])
        return NCMeta(var_name=var_name, dims_order_zyx=tuple(ai), zdim=zdim)
    finally:
        ds.close()


def _read_slice(path: Path, meta: NCMeta, z: int, engine: str | None) -> np.ndarray:
    ds = xr.open_dataset(path, engine=engine or "netcdf4")
    try:
        arr = ds[meta.var_name]
        dims = list(arr.dims)
        arr_zyx = arr.transpose(dims[meta.dims_order_zyx[0]],
                                dims[meta.dims_order_zyx[1]],
                                dims[meta.dims_order_zyx[2]])
        return np.asarray(arr_zyx[z])  # (Y, X)
    finally:
        ds.close()


# ---------------------------
# Геометрия керна и маски
# ---------------------------
from skimage.filters import threshold_otsu

def reconstruction_mask(img2d: np.ndarray, p_dark: float = 0.1) -> np.ndarray:
    """
    Маска реконструкции: отсекает чёрные углы/заведомо неинформативный фон.
    Порог — очень низкий перцентиль, чтобы не съесть керн.
    """
    x = img2d.astype(np.float32)
    t = np.percentile(x, p_dark)
    return x > t

def rock_mask(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32)
    rec = reconstruction_mask(x)
    vals = x[rec] if rec.any() else x
    lo, hi = np.percentile(vals, (1, 99))
    xw = np.clip(x, lo, hi)
    xw = (xw - xw.min()) / max(np.ptp(xw), 1e-6)

    t = threshold_otsu(xw[rec]) if rec.any() else threshold_otsu(xw)
    m = (xw > t) & rec
    m = binary_opening(m, disk(3))
    m = remove_small_holes(m, 10_000)
    m = remove_small_objects(m, 10_000)
    lab = label(m)
    if lab.max() == 0:
        return rec
    lab_id = max(regionprops(lab), key=lambda r: r.area).label
    return lab == lab_id


def circle_params_from_mask(m: np.ndarray) -> Tuple[float, float, float]:
    """
    Центр (cy,cx) — центр масс маски.
    Радиус r: два способа, берём максимальный из них (устойчивее):
      - rms-дистанция: sqrt(mean((y-cy)^2 + (x-cx)^2))
      - по площади: r_area = sqrt(area/pi)
    """
    ys, xs = np.where(m)
    cy, cx = ys.mean(), xs.mean()
    r_rms = np.sqrt(((ys - cy)**2 + (xs - cx)**2).mean())
    r_area = np.sqrt(m.sum() / np.pi)
    r = float(max(r_rms, r_area))
    return float(cy), float(cx), r


def tight_square_crop(mask: np.ndarray) -> Tuple[slice, slice]:
    """
    Квадратный кроп, минимально охватывающий маску.
    """
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h = y1 - y0 + 1
    w = x1 - x0 + 1
    side = max(h, w)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0 = max(cy - side // 2, 0)
    x0 = max(cx - side // 2, 0)
    return slice(y0, y0 + side), slice(x0, x0 + side)


# ---------------------------
# Выравнивание LR -> HR по срезу
# ---------------------------

from skimage.transform import AffineTransform, warp
import numpy as np

def align_lr_to_hr_slice(hr2d, lr2d, margin=4, use_circumscribed_square=True, return_debug=False):
    m_hr = rock_mask(hr2d)
    m_lr = rock_mask(lr2d)

    cy_hr, cx_hr, r_hr = circle_params_from_mask(m_hr)
    cy_lr, cx_lr, r_lr = circle_params_from_mask(m_lr)

    # во сколько раз LR надо УВЕЛИЧИТЬ, чтобы радиус совпал с HR
    k = float(r_hr / max(r_lr, 1e-6))

    # ОДНА аффинная трансформация: T(x) = k*x + t,
    # где t подобран так, чтобы центр совпал после масштаба
    tx = float(cx_hr - k * cx_lr)
    ty = float(cy_hr - k * cy_lr)
    T = AffineTransform(scale=(k, k), translation=(tx, ty))

    # ВАЖНО: для изображения — mode='edge'; для маски — constant(0), order=0
    lr_to_hr   = warp(lr2d, T.inverse, order=1, preserve_range=True, mode='edge')
    m_lr_to_hr = warp(m_lr.astype(float), T.inverse, order=0, preserve_range=True,
                      mode='constant', cval=0.0) > 0.5

    # общий квадрат по HR-кругу
    H, W = hr2d.shape
    if use_circumscribed_square:
        S = int(np.floor(2 * r_hr - margin))
    else:
        S = int(np.floor(np.sqrt(2.0) * r_hr - margin))
    S  = int(np.clip(S, 32, min(H, W)))
    y0 = int(np.clip(round(cy_hr - S/2), 0, H - S))
    x0 = int(np.clip(round(cx_hr - S/2), 0, W - S))

    hr_crop = hr2d[y0:y0+S, x0:x0+S].astype(np.float32)
    lr_crop = lr_to_hr[y0:y0+S, x0:x0+S].astype(np.float32)

    # маска пары — строго пересечение
    m_pair = (m_hr & m_lr_to_hr)[y0:y0+S, x0:x0+S]
    if not m_pair.any():
        # безопасный фолбэк
        m_pair = m_hr[y0:y0+S, x0:x0+S]

    if not return_debug:
        return hr_crop, lr_crop, m_pair
    dbg = {
        "y0": y0, "x0": x0, "S": S,
        "hr_full": hr2d, "lr_aligned_full": lr_to_hr,
        "mask_hr_full": m_hr, "mask_lr_aligned_full": m_lr_to_hr,
        "k": k, "tx": tx, "ty": ty,
        "vmin": float(np.percentile(hr2d[m_hr], 1)) if m_hr.any() else float(np.percentile(hr2d, 1)),
        "vmax": float(np.percentile(hr2d[m_hr], 99)) if m_hr.any() else float(np.percentile(hr2d, 99)),
    }
    return hr_crop, lr_crop, m_pair, dbg



def circumscribed_square_from_circle(cy, cx, r, H, W, margin=0):
    S = int(np.floor(2*r - margin))
    y0 = int(np.clip(round(cy - S/2), 0, max(0, H - S)))
    x0 = int(np.clip(round(cx - S/2), 0, max(0, W - S)))
    return slice(y0, y0+S), slice(x0, x0+S), S

def square_crop_hr_lr(hr2d, lr2d):
    m_hr = rock_mask(hr2d)
    m_lr = rock_mask(lr2d)
    cy_hr, cx_hr, r_hr = circle_params_from_mask(m_hr)
    cy_lr, cx_lr, r_lr = circle_params_from_mask(m_lr)
    # масштаб LR->HR
    k = r_hr / max(r_lr, 1e-6)
    t_scale = AffineTransform(scale=(1/k, 1/k))
    t_shift = AffineTransform(translation=(cx_hr - cx_lr, cy_hr - cy_lr))
    lr_to_hr = warp(warp(lr2d, t_scale.inverse, order=1, preserve_range=True),
                    t_shift.inverse, order=1, preserve_range=True)
    m_lr_to_hr = warp(warp(m_lr.astype(float), t_scale.inverse, order=0, preserve_range=True),
                      t_shift.inverse, order=0, preserve_range=True) > 0.5
    # общий описанный квадрат по HR-кругу
    H, W = hr2d.shape
    ysl, xsl, S = circumscribed_square_from_circle(cy_hr, cx_hr, r_hr, H, W, margin=4)
    hr_crop = hr2d[ysl, xsl]
    lr_crop = lr_to_hr[ysl, xsl]
    m_pair = (m_hr & m_lr_to_hr)[ysl, xsl]
    return hr_crop, lr_crop, m_pair

# ---------------------------
# Выбор патча и нормализация
# ---------------------------

def sample_patch_pair(hr2d, lr2d, mask2d, patch=256, min_frac=0.7, 
                      min_nonzero_frac=0.05, min_std=1e-3, max_tries=80):
    """
    Дополнительные отсечки:
    - не менее min_nonzero_frac пикселей > t_dark (rec-mask) в LR и HR внутри патча
    - std(HR по маске) >= min_std (убираем «почти плоские» патчи)
    """
    H, W = hr2d.shape
    # предрасчёт «не тёмных» пикселей
    rec_hr = reconstruction_mask(hr2d)
    rec_lr = reconstruction_mask(lr2d)

    for _ in range(max_tries):
        y0 = np.random.randint(0, max(1, H - patch + 1))
        x0 = np.random.randint(0, max(1, W - patch + 1))
        m  = mask2d[y0:y0+patch, x0:x0+patch]
        if m.mean() < min_frac:
            continue

        hrp = hr2d[y0:y0+patch, x0:x0+patch]
        lrp = lr2d[y0:y0+patch, x0:x0+patch]
        rh  = rec_hr[y0:y0+patch, x0:x0+patch]
        rl  = rec_lr[y0:y0+patch, x0:x0+patch]

        # доля нетёмных пикселей (внутри патча, не обязательно в маске)
        if rh.mean() < min_nonzero_frac or rl.mean() < min_nonzero_frac:
            continue

        # достаточный контраст по породе
        mm = m > 0
        if mm.any() and hrp[mm].std() < min_std:
            continue

        return hrp, lrp, m

    # жёсткий fallback: центр по маске
    ys, xs = np.where(mask2d)
    if len(ys):
        cy, cx = int(np.median(ys)), int(np.median(xs))
        y0 = np.clip(cy - patch//2, 0, max(0, H - patch))
        x0 = np.clip(cx - patch//2, 0, max(0, W - patch))
    else:
        y0 = max(0, (H - patch)//2); x0 = max(0, (W - patch)//2)
    return (hr2d[y0:y0+patch, x0:x0+patch],
            lr2d[y0:y0+patch, x0:x0+patch],
            mask2d[y0:y0+patch, x0:x0+patch])


def masked_standardize(hrp: np.ndarray, lrp: np.ndarray, mp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Нормализация (x - mean) / std по маске породы (одни и те же mean/std для HR и LR).
    """
    mm = mp > 0
    if mm.any():
        mu = float(hrp[mm].mean())
        sd = float(hrp[mm].std() + 1e-6)
    else:
        mu = float(hrp.mean())
        sd = float(hrp.std() + 1e-6)
    return ( (lrp - mu) / sd, (hrp - mu) / sd )


# ---------------------------
# PyTorch Dataset
# ---------------------------

class NCSlicePairDataset(Dataset):
    """
    Ленивая выдача 2D-патчей (LR, HR, mask) для 2D-SR.
    - Хранит список совпадающих по именам пар (HR_path, LR_path)
    - Для каждого блока — все z-срезы
    - На каждом __getitem__:
        * читает HR/LR срез
        * выравнивает LR к HR (масштаб + сдвиг по кругу керна)
        * строит общую маску и квадратный кроп
        * выбирает случайный патч внутри породы
        * нормализует по маске породы (обе картинки одинаково)
    """

    def __init__(
        self,
        hr_dir: Path,
        lr_dir: Path,
        pattern: str = "block*.nc",
        engine: str = "h5netcdf",
        patch: int = 256,
        min_frac: float = 0.7,
        repeats_per_slice: int = 4,
        align_lr: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir)
        self.engine = engine
        self.patch = int(patch)
        self.min_frac = float(min_frac)
        self.repeats = int(repeats_per_slice)
        self.align_lr = bool(align_lr)
        self.rng = np.random.RandomState(seed)

        # пары блоков по одинаковому имени
        self.pairs: List[Tuple[Path, Path]] = find_matched_blocks(self.hr_dir, self.lr_dir, pattern)

        # метаданные (var_name, порядок осей, zdim) для каждого файла
        self.meta_hr: List[NCMeta] = [_inspect_nc(h, engine=self.engine) for h, _ in self.pairs]
        self.meta_lr: List[NCMeta] = [_inspect_nc(l, engine=self.engine) for _, l in self.pairs]

        # индексы (i_block, z)
        self.index: List[Tuple[int, int, int]] = []  # (i_block, z_hr, z_lr)

        for i, (hr_path, lr_path) in enumerate(self.pairs):
            mh = self.meta_hr[i]
            ml = self.meta_lr[i]
            hr_z0, hr_z1 = _get_z_range(hr_path, self.engine, mh)
            lr_z0, lr_z1 = _get_z_range(lr_path, self.engine, ml)
            g0, g1 = max(hr_z0, lr_z0), min(hr_z1, lr_z1)
            if g0 > g1:
                continue  # блоки не перекрываются по Z
        
            for g in range(g0, g1 + 1):
                z_hr = g - hr_z0
                z_lr = g - lr_z0
                self.index.append((i, z_hr, z_lr))
        
        self.rng.shuffle(self.index)

        # перемешаем
        self.rng.shuffle(self.index)

    def __len__(self) -> int:
        return len(self.index) * self.repeats

    def __getitem__(self, idx: int):
        i = idx // self.repeats
        blk, z_hr, z_lr = self.index[i]
        hr_path, lr_path = self.pairs[blk]
        mh, ml = self.meta_hr[blk], self.meta_lr[blk]
        
        hr2d = _read_slice(hr_path, mh, z_hr, self.engine).astype(np.float32)
        lr2d = _read_slice(lr_path, ml, z_lr, self.engine).astype(np.float32)

        if self.align_lr:
            hr2d, lr2d, m = align_lr_to_hr_slice(hr2d, lr2d)
        else:
            # без выравнивания: общая маска — пересечение масок
            mhm = rock_mask(hr2d)
            mlm = rock_mask(lr2d)
            m = mhm & mlm
            if not m.any():
                m = mhm

            ysl, xsl = tight_square_crop(m)
            hr2d = hr2d[ysl, xsl]
            lr2d = lr2d[ysl, xsl]
            m = m[ysl, xsl]

        # случайный патч внутри породы
        hrp, lrp, mp = sample_patch_pair(hr2d, lr2d, m, patch=self.patch, min_frac=self.min_frac)

        # нормализация по маске
        lrp, hrp = masked_standardize(hrp, lrp, mp)

        # в тензоры
        lr_t = torch.from_numpy(lrp[None])         # (1,H,W)
        hr_t = torch.from_numpy(hrp[None])         # (1,H,W)
        m_t  = torch.from_numpy(mp.astype(np.float32)[None])

        return lr_t, hr_t, m_t


# ---------------------------
# Удобный билдер DataLoader
# ---------------------------

def build_dataloader(
    hr_dir: Path,
    lr_dir: Path,
    pattern: str = "block*.nc",
    engine: str = "netcdf4",
    patch: int = 256,
    min_frac: float = 0.7,
    repeats_per_slice: int = 4,
    batch_size: int = 8,
    num_workers: int = 0,              # начни с 0 для надёжности на Windows/NetCDF
    shuffle: bool = True,
    pin_memory: bool = True,
    align_lr: bool = True,
    seed: int = 42,
) -> Tuple[NCSlicePairDataset, DataLoader]:
    """
    Строит Dataset и DataLoader. На Windows с NetCDF начните с num_workers=0.
    """
    ds = NCSlicePairDataset(
        hr_dir=hr_dir, lr_dir=lr_dir, pattern=pattern,
        engine=engine, patch=patch, min_frac=min_frac,
        repeats_per_slice=repeats_per_slice, align_lr=align_lr, seed=seed
    )
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        pin_memory=pin_memory, drop_last=False,
    )
    return ds, dl

# --- PREVIEW: посмотреть случайную обработанную пару (кроп + патч) ---

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _save_img_with_rect(img, path, rect, title=None, vmin=None, vmax=None, color='red', lw=1.5):
    """
    img: 2D массив
    rect: (x0, y0, w, h) в координатах изображения
    """
    x0, y0, w, h = rect
    plt.figure(figsize=(4,4), dpi=140)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.add_patch(mpatches.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=lw))
    if title: plt.title(title)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()

def _save_img(img, path, title=None, vmin=None, vmax=None):
    plt.figure(figsize=(4,4), dpi=140)
    plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    if title: plt.title(title)
    plt.axis('off'); plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()

def preview_random_pair(
    hr_dir: Path,
    lr_dir: Path,
    pattern: str = "block*.nc",
    engine: str = "netcdf4",
    patch: int = 256,
    min_frac: float = 0.7,
    seed: int = 0,
    out_dir: Path | None = None,
):
    """
    Выбирает случайную пару HR/LR по имени, случайный z-срез.
    1) Выравнивает LR к HR (масштаб и сдвиг по кругу керна)
    2) Делает общий квадратный кроп
    3) Семплирует один патч внутри породы
    4) Сохраняет PNG: HR_crop, LR_aligned_crop, mask_crop, и соответствующие patch'и
    """
    rng = np.random.RandomState(seed)
    pairs = find_matched_blocks(hr_dir, lr_dir, pattern)
    if out_dir is None:
        out_dir = Path.cwd() / "preview_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # случайная пара
    hr_path, lr_path = random.choice(pairs)
    mh = _inspect_nc(hr_path, engine)
    ml = _inspect_nc(lr_path, engine)
    
    # Глобальные диапазоны Z
    hr_z0, hr_z1 = _get_z_range(hr_path, engine, mh)
    lr_z0, lr_z1 = _get_z_range(lr_path, engine, ml)
    
    # Пересечение по глобальному Z
    g0 = max(hr_z0, lr_z0)
    g1 = min(hr_z1, lr_z1)
    if g0 > g1:
        raise RuntimeError(f"Нет пересечения по Z у {hr_path.name} и {lr_path.name}: "
                           f"HR[{hr_z0},{hr_z1}] vs LR[{lr_z0},{lr_z1}]")
    
    # выбираем ГЛОБАЛЬНЫЙ z и переводим в локальные индексы
    g = rng.randint(g0, g1 + 1)  # глобальный Z
    z_hr = g - hr_z0
    z_lr = g - lr_z0
    
    # читаем один и тот же физический срез
    hr2d = _read_slice(hr_path, mh, z_hr, engine).astype(np.float32)
    lr2d = _read_slice(lr_path, ml, z_lr, engine).astype(np.float32)
    
    out = align_lr_to_hr_slice(hr2d, lr2d, return_debug=True)
    hr_crop, lr_crop, m_crop, dbg = out
    
    y0, x0, S = dbg["y0"], dbg["x0"], dbg["S"]
    rect = (x0, y0, S, S)
    hr_full = dbg["hr_full"]
    lr_full_aligned = dbg["lr_aligned_full"]

    from skimage.exposure import match_histograms
    from skimage.metrics import structural_similarity as ssim
    
    hrm = hr_full * dbg["mask_hr_full"]
    lrm = lr_full_aligned * dbg["mask_lr_aligned_full"]
    lrm_hm = match_histograms(lrm, hrm, channel_axis=None)
    
    # Огрубляем (сглаживаем HR до LR за счёт гаусса, чтобы подавить детализацию)
    hrm_blur = gaussian(hrm, sigma=1.5, preserve_range=True)
    
    r = np.corrcoef(hrm_blur[dbg["mask_hr_full"]].ravel(),
                    lrm_hm[dbg["mask_hr_full"]].ravel())[0,1]
    s = ssim(hrm_blur, lrm_hm, data_range=lrm_hm.max()-lrm_hm.min(),
             gaussian_weights=True, sigma=1.5)

    def robust_window(x, mask=None, lo=1, hi=99):
        x = x.astype(np.float32)
        if mask is not None and mask.any():
            vals = x[mask]
        else:
            vals = x
        return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    
    # окна для full
    vmin_hr, vmax_hr = robust_window(hr_full, dbg["mask_hr_full"])
    vmin_lr, vmax_lr = robust_window(lr_full_aligned, dbg["mask_lr_aligned_full"])
    
    _save_img(hr_full, out_dir / "00_HR_full.png", f"HR full (z={z_hr}, {hr_full.shape})", vmin_hr, vmax_hr)
    _save_img(lr_full_aligned, out_dir / "00_LR_full_aligned.png", f"LR full aligned ({lr_full_aligned.shape})", vmin_lr, vmax_lr)
    _save_img_with_rect(hr_full, out_dir / "00a_HR_full_with_crop.png", rect, ..., vmin_hr, vmax_hr)
    _save_img_with_rect(lr_full_aligned, out_dir / "00b_LR_full_aligned_with_crop.png", rect, ..., vmin_lr, vmax_lr)
    
    # окна для crop
    vmin_hrc, vmax_hrc = robust_window(hr_crop, m_crop)
    vmin_lrc, vmax_lrc = robust_window(lr_crop, m_crop)
    _save_img(hr_crop, out_dir / "01_HR_crop.png", ..., vmin_hrc, vmax_hrc)
    _save_img(lr_crop, out_dir / "02_LR_aligned_crop.png", ..., vmin_lrc, vmax_lrc)
    _save_img(m_crop.astype(float), out_dir / "03_mask_crop.png", "Mask (rock)", 0, 1)
    
    # --- 2. Один патч (как и раньше) ---
    hrp, lrp, mp = sample_patch_pair(hr_crop, lr_crop, m_crop, patch=patch, min_frac=min_frac)
    lrp_n, hrp_n = masked_standardize(hrp, lrp, mp)
    vmin_p = float(np.percentile(hrp[mp], 1)) if mp.any() else float(np.percentile(hrp, 1))
    vmax_p = float(np.percentile(hrp[mp], 99)) if mp.any() else float(np.percentile(hrp, 99))
    _save_img(hrp, out_dir / "11_HR_patch_raw.png", f"HR patch raw ({hrp.shape})", vmin_p, vmax_p)
    _save_img(lrp, out_dir / "12_LR_patch_raw.png", f"LR patch raw ({lrp.shape})", vmin_p, vmax_p)
    _save_img(mp.astype(float), out_dir / "13_mask_patch.png", "Mask patch", 0, 1)
    
    def to01(x):
        x = (x - x.min()) / max(np.ptp(x), 1e-6); return x
    _save_img(to01(hrp_n), out_dir / "21_HR_patch_norm.png", "HR patch normalized", 0, 1)
    _save_img(to01(lrp_n), out_dir / "22_LR_patch_norm.png", "LR patch normalized", 0, 1)
    
    # --- 3. Записать служебную инфу ---
    info = {
        "hr_file": str(hr_path.name),
        "lr_file": str(lr_path.name),
        "full_shape_hr": tuple(map(int, hr_full.shape)),
        "full_shape_lr_aligned": tuple(map(int, lr_full_aligned.shape)),
        "crop_box": {"y0": int(y0), "x0": int(x0), "S": int(S)},
        "crop_shape": tuple(map(int, hr_crop.shape)),
        "patch_shape": tuple(map(int, hrp.shape)),
    }

    def robust_stats(x, mask=None):
        x = x.astype(np.float32)
        vals = x[mask] if (mask is not None and mask.any()) else x
        return {
            "min": float(vals.min()),
            "p1": float(np.percentile(vals, 1)),
            "median": float(np.median(vals)),
            "p99": float(np.percentile(vals, 99)),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "hr_var": mh.var_name,
            "lr_var": ml.var_name,
        }

    info.update({
        "hr_stats_full": robust_stats(hr_full, dbg["mask_hr_full"]),
        "lr_stats_full": robust_stats(lr_full_aligned, dbg["mask_lr_aligned_full"]),
        "hr_stats_crop": robust_stats(hr_crop, m_crop),
        "lr_stats_crop": robust_stats(lr_crop, m_crop),
    })

    info.update({
    "z_global": int(g),
    "z_local_hr": int(z_hr),
    "z_local_lr": int(z_lr),
    "hr_z_range": (int(hr_z0), int(hr_z1)),
    "lr_z_range": (int(lr_z0), int(lr_z1)),
    })

    info.update({"corr_full_masked": float(r), "ssim_full_masked": float(s)})
    
    with open(out_dir / "preview_info.txt", "w", encoding="utf-8") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print("saved to:", out_dir)
    return out_dir, info

    
# ---------------------------
# Пример использования (скрипт)
# ---------------------------

if __name__ == "__main__":
    # Пример: подставьте свои пути
    HR_DIR = Path(r"C:\MEC_ILS\ILS_HR")
    LR_DIR = Path(r"C:\MEC_ILS\ILS_LR")
    preview_random_pair(HR_DIR, LR_DIR, patch=256, seed=2)
    ds, dl = build_dataloader(
        hr_dir=HR_DIR,
        lr_dir=LR_DIR,
        pattern="block*.nc",
        engine="netcdf4",    # можно попробовать "netcdf4" если h5netcdf капризничает
        patch=256,
        min_frac=0.7,
        repeats_per_slice=2,
        batch_size=4,
        num_workers=0,
        align_lr=True,
    )

    # sanity-check: одна пачка
    lr, hr, m = next(iter(dl))
    print("LR batch:", lr.shape, lr.dtype)
    print("HR batch:", hr.shape, hr.dtype)
    print("Mask   :", m.shape, m.dtype)