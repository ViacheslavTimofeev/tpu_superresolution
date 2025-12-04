from __future__ import annotations

from typing import Optional, Tuple, Callable, Sequence
import torch
from torchvision.transforms import v2 as T
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from typing import Any
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import resize
import random

# ---------- Базовые парные трансформы (LR, HR) ----------

class PairCompose:
    def __init__(self, transforms: Sequence[Callable]):
        self.transforms = list(transforms)
    def __call__(self, lr, hr):
        for t in self.transforms:
            lr, hr = t(lr, hr)
        return lr, hr


def convert_rgb_to_ycbcr(image: Any) -> Any:
    """
    Локальный аналог imgproc.convert_rgb_to_ycbcr из MS-ResUNet.

    Поддерживаем:
      - np.ndarray: (H, W, 3), значения ~0..255
      - torch.Tensor: (3, H, W) или (1, 3, H, W), значения ~0..255

    Возвращаем:
      - np.ndarray: (H, W, 3), float32
      - torch.Tensor: (3, H, W), float32
    """
    # ----- Вариант для numpy -----
    if isinstance(image, np.ndarray):
        # image[..., 0] = R, 1 = G, 2 = B
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        y  = 16.0  + (64.738 * r + 129.057 * g + 25.064 * b) / 256.0
        cb = 128.0 + (-37.945 * r - 74.494 * g + 112.439 * b) / 256.0
        cr = 128.0 + (112.439 * r - 94.154 * g - 18.285 * b) / 256.0

        ycbcr = np.stack([y, cb, cr], axis=-1).astype(np.float32)  # (H,W,3)
        return ycbcr

    # ----- Вариант для torch.Tensor -----
    if isinstance(image, torch.Tensor):
        # (1,3,H,W) → (3,H,W)
        if image.ndim == 4:
            image = image.squeeze(0)

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(
                f"convert_rgb_to_ycbcr ожидает (3,H,W) или (1,3,H,W), получил {tuple(image.shape)}"
            )

        r = image[0, :, :]
        g = image[1, :, :]
        b = image[2, :, :]

        y  = 16.0  + (64.738 * r + 129.057 * g + 25.064 * b) / 256.0
        cb = 128.0 + (-37.945 * r - 74.494 * g + 112.439 * b) / 256.0
        cr = 128.0 + (112.439 * r - 94.154 * g - 18.285 * b) / 256.0

        ycbcr = torch.stack([y, cb, cr], dim=0).float()  # (3,H,W)
        return ycbcr

    raise TypeError(f"Unknown type for convert_rgb_to_ycbcr: {type(image)}")


def image_to_y_tensor(ycbcr: Any) -> torch.Tensor:
    """
    Превращает YCbCr-изображение в тензор Y-канала (1, H, W) в [0,1],
    в духе imgproc.image2tensor(..., range_norm=False, half=False).

    Поддерживаем:
      - np.ndarray: (H, W, 3) или (H, W) с диапазоном ~0..255
      - torch.Tensor:
          * (3, H, W) или (1, 3, H, W) — считаем, что это YCbCr, берём канал 0
          * (H, W) или (1, H, W)       — считаем, что это уже Y

    Возвращаем:
      - torch.Tensor shape (1, H, W), float32, примерно [0.06, 0.92] для стандартного Y.
    """
    # ----- numpy -----
    if isinstance(ycbcr, np.ndarray):
        if ycbcr.ndim == 3:
            # (H,W,3) -> берём Y
            y = ycbcr[..., 0]
        elif ycbcr.ndim == 2:
            # (H,W) -> считаем, что это уже Y
            y = ycbcr
        else:
            raise ValueError(f"image_to_y_tensor: неожиданный shape np.ndarray: {ycbcr.shape}")

        y = y.astype(np.float32) / 255.0
        return torch.from_numpy(y).unsqueeze(0).float()  # (1,H,W)

    # ----- torch.Tensor -----
    if isinstance(ycbcr, torch.Tensor):
        t = ycbcr

        # (1,3,H,W) -> (3,H,W)
        if t.ndim == 4 and t.shape[1] == 3:
            t = t.squeeze(0)

        if t.ndim == 3 and t.shape[0] == 3:
            # (3,H,W): YCbCr, берём канал 0
            y = t[0, :, :]
        elif t.ndim == 3 and t.shape[0] == 1:
            # (1,H,W): уже Y
            y = t[0, :, :]
        elif t.ndim == 2:
            # (H,W): уже Y
            y = t
        else:
            raise ValueError(f"image_to_y_tensor: неожиданный shape Tensor: {tuple(t.shape)}")

        y = y.float() / 255.0
        return y.unsqueeze(0)  # (1,H,W)

    raise TypeError(f"image_to_y_tensor: неподдерживаемый тип {type(ycbcr)}")

    
class PairRGBToYCbCrY:
    """
    Парный трансформ (LR, HR) -> (Y_lr, Y_hr) через RGB -> YCbCr -> Y.

    Идея: повторить логику MS-ResUNet:
      - работаем с RGB-картинками (или псевдо-RGB из серого),
      - считаем YCbCr по BT.601-коэффициентам,
      - берём только Y, делим на 255 -> (1,H,W) в [0,1].

    Рекомендуется ставить после PairToTensor01, когда вход уже torch.Tensor.
    """

    def _to_y(self, x: Any) -> torch.Tensor:
        # 1) PIL.Image -> np.float32 (0..255)
        if isinstance(x, Image.Image):
            arr = np.array(x).astype(np.float32)
            # Если grayscale -> расширяем до (H,W,3) простым дублированием
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            ycbcr = convert_rgb_to_ycbcr(arr)          # np (H,W,3)
            return image_to_y_tensor(ycbcr)            # Tensor (1,H,W) в [0,1]

        # 2) numpy -> Y (через нашу функцию)
        if isinstance(x, np.ndarray):
            arr = x.astype(np.float32)
            if arr.ndim == 2:
                # уже (H,W), считаем, что это Y
                return image_to_y_tensor(arr)
            if arr.ndim == 3 and arr.shape[2] == 3:
                ycbcr = convert_rgb_to_ycbcr(arr)
                return image_to_y_tensor(ycbcr)
            raise ValueError(f"PairRGBToYCbCrY: np.ndarray с shape={arr.shape} не поддержан")

        # 3) torch.Tensor (обычный кейс после PairToTensor01)
        if isinstance(x, torch.Tensor):
            t = x

            # (H,W) -> (1,H,W)
            if t.ndim == 2:
                t = t.unsqueeze(0)

            # (C,H,W)
            if t.ndim != 3:
                raise ValueError(f"PairRGBToYCbCrY: ожидался Tensor (C,H,W), получил {tuple(t.shape)}")

            # Если один канал (серое), дублируем до псевдо-RGB
            if t.shape[0] == 1:
                t = t.repeat(3, 1, 1)  # (3,H,W)

            if t.shape[0] != 3:
                raise ValueError(f"PairRGBToYCbCrY: ожидался 3-канальный тензор, получил C={t.shape[0]}")

            # если t сейчас в [0,1]; приводим к 0..255 как в их коде
            t_255 = t * 255.0

            # convert_rgb_to_ycbcr вернёт torch.Tensor (3,H,W)
            ycbcr = convert_rgb_to_ycbcr(t_255)
            y = image_to_y_tensor(ycbcr)   # (1,H,W) в [0,1]

            return y

        raise TypeError(f"PairRGBToYCbCrY: неподдерживаемый тип {type(x)}")

    def __call__(self, lr, hr):
        lr_y = self._to_y(lr)
        hr_y = self._to_y(hr)
        return lr_y, hr_y


class PairGrayscale:
    def __init__(self, num_output_channels: int = 1):
        self.t = T.Grayscale(num_output_channels)

    def gray(self, x):
        if isinstance(x, torch.Tensor):
            # (C,H,W) или (H,W)
            if x.ndim == 2:        # (H,W)
                return x.unsqueeze(0)  # (1,H,W)
            if x.ndim == 3 and x.shape[0] == 1:
                return x               # уже (1,H,W)
            return self.t(x)

        if isinstance(x, Image.Image):
            if x.mode in ("L", "F", "I", "I;16"):
                return x
            if x.mode in ("RGB", "RGBA"):
                return self.t(x)
            return self.t(x)

        arr = np.asarray(x)
        if arr.ndim == 2:
            return arr  # уже "серое"
        return self.t(x)

    def __call__(self, lr, hr):
        return self.gray(lr), self.gray(hr)


class PairUpscaleLRtoHR:
    """Апскейлит LR до точного размера HR (bicubic)."""
    def __call__(self, lr, hr):
        if lr.size != hr.size:  # PIL: size=(W,H)
            H, W = hr.size[1], hr.size[0]
            lr = TF.resize(lr, size=[H, W],
                           interpolation=InterpolationMode.BICUBIC,
                           antialias=True)
        return lr, hr

class PairRandomCrop:
    """
    Согласованный RandomCrop для пары (LR, HR).

    Предполагает, что LR и HR уже приведены к одному размеру
    (например, после PairUpscaleLRtoHR), и вырезает один и тот же
    прямоугольник из обоих изображений.
    """
    def __init__(self, size: int | Tuple[int, int]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def __call__(self, lr, hr):
        th, tw = self.size  # target height/width

        # Определяем текущий размер
        if isinstance(hr, Image.Image):
            w, h = hr.size   # PIL: (W, H)
        else:
            # torch.Tensor: (..., H, W)
            h, w = hr.shape[-2], hr.shape[-1]

        # Если патч совпадает с размером изображения — ничего не делаем
        if h == th and w == tw:
            return lr, hr

        # Если патч больше изображения — просто центр-кроп по минимальному размеру
        if h < th or w < tw:
            th = min(th, h)
            tw = min(tw, w)
            top = max(0, (h - th) // 2)
            left = max(0, (w - tw) // 2)
        else:
            # Случайный сдвиг так, чтобы патч целиком лежал внутри
            top = int(torch.randint(0, h - th + 1, (1,)).item())
            left = int(torch.randint(0, w - tw + 1, (1,)).item())

        if isinstance(hr, Image.Image):
            hr = TF.crop(hr, top, left, th, tw)
            lr = TF.crop(lr, top, left, th, tw)
        else:
            # Для тензоров режем по последним двум осям (..., H, W)
            hr = hr[..., top:top + th, left:left + tw]
            lr = lr[..., top:top + th, left:left + tw]

        return lr, hr

class PairFlips:
    """Согласованные флипы."""
    def __init__(self, p_flip=0.5, p_vflip=0.5):
        self.pf, self.pv = float(p_flip), float(p_vflip)
    def __call__(self, lr, hr):
        if torch.rand(()) < self.pf:
            lr, hr = TF.hflip(lr), TF.hflip(hr)
        if torch.rand(()) < self.pv:
            lr, hr = TF.vflip(lr), TF.vflip(hr)
        return lr, hr

class PairToTensor01:
    """ToImage -> float32 [0,1] для обеих."""
    def __init__(self):
        self.to_img = T.ToImage()
        self.to_f32 = T.ToDtype(torch.float32, scale=True)
    def __call__(self, lr, hr):
        lr = self.to_f32(self.to_img(lr))
        hr = self.to_f32(self.to_img(hr))
        return lr, hr
        
class PairMinMaxScale:
    def __init__(self, vmin: float, vmax: float):
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.eps  = 1e-6

    def __call__(self, lr, hr):
        lr = (lr - self.vmin) / (self.vmax - self.vmin + self.eps)
        hr = (hr - self.vmin) / (self.vmax - self.vmin + self.eps)
        lr = lr.clamp(0.0, 1.0)
        hr = hr.clamp(0.0, 1.0)
        return lr, hr

class PairGaussianBlur:
    """
    Согласованный GaussianBlur на тензорах [0,1].
    """
    def __init__(self, kernel_size: int, sigma=(0.1, 2.0), p: float = 0.5):
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size должен быть нечётным")
        self.kernel = (kernel_size, kernel_size)
        self.sigma = sigma
        self.p = float(p)

    def __call__(self, lr, hr):
        if torch.rand(()) >= self.p:
            return lr, hr
        if isinstance(self.sigma, (tuple, list)):
            s_low, s_high = self.sigma
            sigma = float(torch.empty(1).uniform_(s_low, s_high))
        else:
            sigma = float(self.sigma)
        lr = TF.gaussian_blur(lr, kernel_size=self.kernel, sigma=sigma)
        hr = TF.gaussian_blur(hr, kernel_size=self.kernel, sigma=sigma)
        return lr, hr

class PairNormalize:
    """Одинаковый Normalize к обоим тензорам (каналы те же)."""
    def __init__(self, mean, std):
        self.norm = T.Normalize(mean=mean, std=std)
    def __call__(self, lr, hr):
        return self.norm(lr), self.norm(hr)

# ---------- Сборщики пайплайнов ----------

def build_pair_transform(
    do_flips: bool = True,
    do_blur: bool = True,
    blur_kernel: int = 3,
    blur_sigma: Tuple[float, float] = (0.1, 1.5),
    mean: Tuple[float, ...] = (0.45161797,),
    std: Tuple[float, ...]  = (0.20893379,),
    normalize: bool = False,
    dataset: str = "DeepRock",
    patch_size: Optional[int] = None,
    vmin: float | None = None,
    vmax: float | None = None
) -> PairCompose:

    stages = []
    """
    # все, что под docstrings, то для нашего эксперимента
    stages.append(PairGrayscale())  
    stages.append(PairUpscaleLRtoHR())

    if patch_size is not None:
        stages.append(PairRandomCrop(patch_size))
    if do_flips:
        stages.append(PairFlips())
    stages.append(PairToTensor01())
    if dataset == "mrccm":
        stages.append(PairMinMaxScale(vmin, vmax))
    if do_blur:
        stages.append(PairGaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma, p=0.5))
    if normalize:
        stages.append(PairNormalize(mean=mean, std=std))
    """
    # все, что ниже, то для рекреации статьи
    stages.append(PairUpscaleLRtoHR())
    if patch_size is not None:
        stages.append(PairRandomCrop(patch_size))
    if do_flips:
        stages.append(PairFlips())
    stages.append(PairToTensor01())
    stages.append(PairRGBToYCbCrY())

    return PairCompose(stages)
    
def build_pair_transform_eval(
    mean: Tuple[float, ...] = (0.45161797,),
    std: Tuple[float, ...]  = (0.20893379,),
    normalize: bool = False,
    dataset: str = "DeepRock",
    vmin: float | None = None,
    vmax: float | None = None
) -> PairCompose:

    stages = []
    """
    stages.append(PairGrayscale())
    stages.append(PairUpscaleLRtoHR())
    stages.append(PairToTensor01())
    if dataset == "mrccm":
        stages.append(PairMinMaxScale(vmin, vmax))

    if normalize:
        stages.append(PairNormalize(mean, std))
    """
    stages.append(PairUpscaleLRtoHR())
    stages.append(PairToTensor01())
    stages.append(PairRGBToYCbCrY())
    return PairCompose(stages)