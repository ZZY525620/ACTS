"""Small NIfTI-1 reader/writer for .nii.gz files.

This avoids adding nibabel as a hard dependency for the first MVP. It supports
the simple NIfTI files used by the FLARE22 examples in this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import struct
from typing import Any

import numpy as np

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - fallback path for minimal runtimes.
    nib = None


_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    512: np.uint16,
    768: np.uint32,
}


@dataclass(frozen=True)
class NiftiImage:
    array: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, ...]
    header: bytes | None = None
    nib_image: Any | None = None


def _endian(header: bytes) -> str:
    if struct.unpack("<i", header[:4])[0] == 348:
        return "<"
    if struct.unpack(">i", header[:4])[0] == 348:
        return ">"
    raise ValueError("Not a valid NIfTI-1 header.")


def load_nii(path: str | Path) -> NiftiImage:
    path = Path(path)
    if nib is not None:
        nii = nib.load(str(path))
        array = np.asanyarray(nii.dataobj)
        spacing = tuple(float(x) for x in nii.header.get_zooms()[: array.ndim])
        return NiftiImage(
            array=np.asarray(array),
            affine=np.asarray(nii.affine, dtype=np.float32),
            spacing=spacing,
            header=None,
            nib_image=nii,
        )

    with gzip.open(path, "rb") as f:
        header = f.read(348)
        endian = _endian(header)
        dim = struct.unpack(endian + "8h", header[40:56])
        ndim = int(dim[0])
        shape = tuple(int(x) for x in dim[1 : ndim + 1])
        datatype = struct.unpack(endian + "h", header[70:72])[0]
        pixdim = struct.unpack(endian + "8f", header[76:108])
        vox_offset = int(struct.unpack(endian + "f", header[108:112])[0])
        scl_slope = struct.unpack(endian + "f", header[112:116])[0]
        scl_inter = struct.unpack(endian + "f", header[116:120])[0]

        if datatype not in _DTYPES:
            raise ValueError(f"Unsupported NIfTI datatype: {datatype}")

        f.seek(vox_offset)
        raw = f.read()

    dtype = np.dtype(_DTYPES[datatype]).newbyteorder(endian)
    array = np.frombuffer(raw, dtype=dtype).copy()
    expected = int(np.prod(shape))
    if array.size < expected:
        raise ValueError(f"{path} ended early: got {array.size}, expected {expected}")
    array = array[:expected].reshape(shape, order="F")

    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        slope = 1.0 if scl_slope == 0.0 else scl_slope
        array = array.astype(np.float32) * slope + scl_inter

    affine = np.eye(4, dtype=np.float32)
    affine[0, 0] = pixdim[1]
    affine[1, 1] = pixdim[2]
    affine[2, 2] = pixdim[3]
    return NiftiImage(array=array, affine=affine, spacing=tuple(pixdim[1 : ndim + 1]), header=header)


def save_nii_like(path: str | Path, array: np.ndarray, reference: NiftiImage, dtype=np.uint8) -> None:
    """Save a 3D array as a simple NIfTI-1 .nii.gz using a reference header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=dtype)
    if nib is not None:
        header = reference.nib_image.header.copy() if reference.nib_image is not None else None
        image = nib.Nifti1Image(arr, reference.affine, header=header)
        image.set_data_dtype(np.dtype(dtype))
        nib.save(image, str(path))
        return

    if reference.header is None:
        raise ValueError("Fallback NIfTI writer requires a byte header reference.")
    header = bytearray(reference.header)
    endian = _endian(reference.header)

    datatype = {
        np.dtype(np.uint8): 2,
        np.dtype(np.int16): 4,
        np.dtype(np.int32): 8,
        np.dtype(np.float32): 16,
        np.dtype(np.float64): 64,
        np.dtype(np.uint16): 512,
        np.dtype(np.uint32): 768,
    }[np.dtype(dtype)]
    bitpix = np.dtype(dtype).itemsize * 8

    struct.pack_into(endian + "8h", header, 40, 3, arr.shape[0], arr.shape[1], arr.shape[2], 1, 1, 1, 1)
    struct.pack_into(endian + "h", header, 70, datatype)
    struct.pack_into(endian + "h", header, 72, bitpix)
    struct.pack_into(endian + "f", header, 108, 352.0)
    struct.pack_into(endian + "f", header, 112, 1.0)
    struct.pack_into(endian + "f", header, 116, 0.0)

    with gzip.open(path, "wb") as f:
        f.write(header)
        f.write(b"\0\0\0\0")
        f.write(np.asarray(arr, dtype=np.dtype(dtype).newbyteorder(endian)).ravel(order="F").tobytes())

