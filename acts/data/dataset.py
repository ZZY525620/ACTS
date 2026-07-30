from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .nifti import NiftiImage, load_nii


@dataclass(frozen=True)
class CTCase:
    case_id: str
    image: NiftiImage
    label: NiftiImage
    liver_mask: np.ndarray


def load_flare_case(data_dir: str | Path, case_id: str, liver_label: int = 1) -> CTCase:
    data_dir = Path(data_dir)
    image_path, label_path = resolve_flare_paths(data_dir, case_id)
    image = load_nii(image_path)
    label = load_nii(label_path)
    if image.array.shape != label.array.shape:
        raise ValueError(f"Image/label shape mismatch: {image.array.shape} vs {label.array.shape}")
    liver_mask = (label.array == liver_label).astype(np.uint8)
    return CTCase(case_id=case_id, image=image, label=label, liver_mask=liver_mask)


def resolve_flare_paths(data_dir: str | Path, case_id: str) -> tuple[Path, Path]:
    """Resolve a FLARE case from either flat or image/labels folder layouts."""
    data_dir = Path(data_dir)
    image_name = f"FLARE22_Tr_{case_id}_0000.nii.gz"
    label_name = f"FLARE22_Tr_{case_id}.nii.gz"
    candidates = [
        (data_dir / image_name, data_dir / label_name),
        (data_dir / "image" / image_name, data_dir / "labels" / label_name),
        (data_dir / "images" / image_name, data_dir / "labels" / label_name),
        (data_dir / "image" / image_name, data_dir / "labels" / "labels" / label_name),
        (data_dir / "images" / image_name, data_dir / "labels" / "labels" / label_name),
    ]
    for image_path, label_path in candidates:
        if image_path.exists() and label_path.exists():
            return image_path, label_path
    searched = "\n".join(f"image={image_path} label={label_path}" for image_path, label_path in candidates)
    raise FileNotFoundError(f"Could not find FLARE case {case_id} under {data_dir}. Searched:\n{searched}")


def choose_reference_slice(mask_volume: np.ndarray) -> int:
    areas = mask_volume.sum(axis=(0, 1))
    if areas.max() == 0:
        raise ValueError("No foreground liver label found.")
    return int(np.argmax(areas))


def slice_range(mask_volume: np.ndarray) -> tuple[int, int]:
    areas = mask_volume.sum(axis=(0, 1))
    indices = np.where(areas > 0)[0]
    if indices.size == 0:
        return -1, -1
    return int(indices[0]), int(indices[-1])

