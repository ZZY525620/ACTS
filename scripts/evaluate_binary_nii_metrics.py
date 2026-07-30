from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


DATASET_ORGAN_LABELS = {
    "FLARE": {
        "liver": 1,
        "right_kidney": 2,
        "spleen": 3,
        "pancreas": 4,
        "left_kidney": 13,
    },
    "AMOS": {
        "spleen": 1,
        "right_kidney": 2,
        "left_kidney": 3,
        "liver": 6,
        "pancreas": 10,
    },
}


def as_bool_mask(array: np.ndarray) -> np.ndarray:
    return np.asarray(array) > 0


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = as_bool_mask(pred)
    gt = as_bool_mask(gt)
    denom = float(pred.sum() + gt.sum())
    if denom == 0.0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = as_bool_mask(pred)
    gt = as_bool_mask(gt)
    union = float(np.logical_or(pred, gt).sum())
    if union == 0.0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def surface(mask: np.ndarray) -> np.ndarray:
    mask = as_bool_mask(mask)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, eroded)


def nsd(pred: np.ndarray, gt: np.ndarray, spacing: tuple[float, float, float], tolerance_mm: float) -> float:
    pred = as_bool_mask(pred)
    gt = as_bool_mask(gt)
    if not pred.any() and not gt.any():
        return 1.0
    if not pred.any() or not gt.any():
        return 0.0

    pred_surface = surface(pred)
    gt_surface = surface(gt)

    dist_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=spacing)
    dist_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_ok = dist_to_gt[pred_surface] <= tolerance_mm
    gt_ok = dist_to_pred[gt_surface] <= tolerance_mm
    denom = float(pred_surface.sum() + gt_surface.sum())
    if denom == 0.0:
        return 1.0
    return float((pred_ok.sum() + gt_ok.sum()) / denom)


def load_pred_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return as_bool_mask(image.get_fdata()), image


def default_label(dataset: str, organ: str) -> int | None:
    dataset_key = dataset.strip().upper()
    organ_key = organ.strip()
    return DATASET_ORGAN_LABELS.get(dataset_key, {}).get(organ_key)


def load_gt_mask(path: Path, dataset: str, organ: str, label: int | None) -> tuple[np.ndarray, nib.Nifti1Image, int | None]:
    image = nib.load(str(path))
    array = image.get_fdata()
    if label is None:
        label = default_label(dataset, organ)
    if label is None:
        return as_bool_mask(array), image, None
    return np.asarray(array == label), image, label


def parse_label(value: str) -> int | None:
    value = value.strip()
    if value == "":
        return None
    return int(value)


def evaluate_row(row: dict[str, str], default_tolerance: float) -> dict[str, str | float | int | None]:
    method = row["method"].strip()
    dataset = row.get("dataset", "").strip()
    case_id = row["case_id"].strip()
    organ = row["organ"].strip()
    pred_path = Path(row["pred_path"].strip())
    gt_path = Path(row["gt_path"].strip())
    label = parse_label(row.get("label", ""))
    tolerance = float(row.get("tolerance_mm", "").strip() or default_tolerance)

    pred, pred_img = load_pred_mask(pred_path)
    gt, gt_img, label = load_gt_mask(gt_path, dataset, organ, label)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch for {method} {case_id} {organ}: pred={pred.shape}, gt={gt.shape}")

    spacing = tuple(float(v) for v in gt_img.header.get_zooms()[:3])
    return {
        "method": method,
        "dataset": dataset,
        "case_id": case_id,
        "organ": organ,
        "label": label if label is not None else "",
        "pred_path": str(pred_path),
        "gt_path": str(gt_path),
        "spacing_xyz": "x".join(f"{v:.6g}" for v in spacing),
        "tolerance_mm": tolerance,
        "pred_voxels": int(pred.sum()),
        "gt_voxels": int(gt.sum()),
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "nsd": nsd(pred, gt, spacing=spacing, tolerance_mm=tolerance),
    }


def write_summary(rows: list[dict[str, str | float | int | None]], path: Path) -> None:
    groups: dict[tuple[str, str], list[dict[str, str | float | int | None]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), str(row["organ"])), []).append(row)

    fieldnames = ["method", "organ", "n", "dice_mean", "iou_mean", "nsd_mean"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (method, organ), group in sorted(groups.items()):
            writer.writerow(
                {
                    "method": method,
                    "organ": organ,
                    "n": len(group),
                    "dice_mean": np.mean([float(r["dice"]) for r in group]),
                    "iou_mean": np.mean([float(r["iou"]) for r in group]),
                    "nsd_mean": np.mean([float(r["nsd"]) for r in group]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary NIfTI predictions with Dice, IoU, and NSD.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with method,dataset,case_id,organ,pred_path,gt_path,label.")
    parser.add_argument("--out", required=True, type=Path, help="Output per-case CSV path.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Optional mean-by-method-organ CSV path.")
    parser.add_argument("--tolerance-mm", type=float, default=1.0, help="NSD tolerance in millimeters.")
    args = parser.parse_args()

    with args.manifest.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = [evaluate_row(row, default_tolerance=args.tolerance_mm) for row in rows]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys()) if results else []
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        write_summary(results, args.summary_out)

    print(f"Saved per-case metrics: {args.out}")
    if args.summary_out is not None:
        print(f"Saved summary metrics: {args.summary_out}")


if __name__ == "__main__":
    main()

