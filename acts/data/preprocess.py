from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CTPreprocessResult:
    image_uint8: np.ndarray
    sam_images_rgb: np.ndarray
    original_size: tuple[int, int]
    sam_input_size: tuple[int, int]


def window_ct(volume: np.ndarray, window_min: float = -160.0, window_max: float = 240.0) -> np.ndarray:
    # Abdominal soft-tissue window: center=40, width=400.
    clipped = np.clip(volume.astype(np.float32), window_min, window_max)
    return (clipped - window_min) / (window_max - window_min)


def to_uint8(volume01: np.ndarray) -> np.ndarray:
    return np.clip(volume01 * 255.0, 0, 255).astype(np.uint8)


def resize_slice(slice2d: np.ndarray, size: int | tuple[int, int] = 256, nearest: bool = False) -> np.ndarray:
    if isinstance(size, int):
        out_size = (size, size)
    else:
        out_size = (int(size[1]), int(size[0]))
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(slice2d)
    return np.asarray(image.resize(out_size, mode))


def resize_volume_xy(volume: np.ndarray, size: int | tuple[int, int] = 256, nearest: bool = False) -> np.ndarray:
    slices = [resize_slice(volume[:, :, z], size=size, nearest=nearest) for z in range(volume.shape[2])]
    return np.stack(slices, axis=2)


def gray_to_rgb(volume_uint8: np.ndarray) -> np.ndarray:
    return np.repeat(volume_uint8[..., None], 3, axis=3)


def preprocess_ct_for_sam(
    volume: np.ndarray,
    window_min: float = -160.0,
    window_max: float = 240.0,
    sam_size: int = 256,
) -> CTPreprocessResult:
    image_uint8 = to_uint8(window_ct(volume, window_min=window_min, window_max=window_max))
    resized = resize_volume_xy(image_uint8, size=sam_size, nearest=False)
    sam_images_rgb = gray_to_rgb(resized)
    return CTPreprocessResult(
        image_uint8=image_uint8,
        sam_images_rgb=sam_images_rgb,
        original_size=(int(volume.shape[0]), int(volume.shape[1])),
        sam_input_size=(int(resized.shape[0]), int(resized.shape[1])),
    )


def ct_to_rgb_slices(volume: np.ndarray, size: int = 256) -> np.ndarray:
    return preprocess_ct_for_sam(volume, sam_size=size).sam_images_rgb


def resize_mask_to_original(mask: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    restored = resize_slice(mask.astype(np.uint8), size=original_size, nearest=True)
    return (restored > 0).astype(np.uint8)


def restore_mask_volume_xy(mask_volume: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    slices = [resize_mask_to_original(mask_volume[:, :, z], original_size) for z in range(mask_volume.shape[2])]
    return np.stack(slices, axis=2).astype(np.uint8)

