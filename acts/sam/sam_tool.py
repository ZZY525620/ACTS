from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from acts.prompts.prompt_from_mask import mask_to_box


@dataclass(frozen=True)
class Prediction:
    masks: list[np.ndarray]
    scores: list[float]
    logits: list[np.ndarray]


class SegmentAnythingSAMTool:
    """Wrapper around Meta's segment-anything SamPredictor.

    Install requirement:
        pip install git+https://github.com/facebookresearch/segment-anything.git

    Checkpoint examples:
        sam_vit_b_01ec64.pth, sam_vit_l_0b3195.pth, sam_vit_h_4b8939.pth
    """

    def __init__(self, model_path: str, model_type: str = "vit_b", device: str = "cpu"):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "segment_anything is not installed. Install it before using --sam-backend sam."
            ) from exc

        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.device = device
        self.slice_index = 0

    def set_slice_index(self, slice_index: int) -> None:
        self.slice_index = int(slice_index)

    def set_image(self, image: np.ndarray) -> None:
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        self.predictor.set_image(image)

    def predict(
        self,
        box: list[int] | None = None,
        points: list[list[int]] | None = None,
        point_labels: list[int] | None = None,
        mask_input: np.ndarray | None = None,
    ) -> Prediction:
        point_coords = np.asarray(points, dtype=np.float32) if points is not None else None
        point_label_arr = np.asarray(point_labels, dtype=np.int32) if point_labels is not None else None
        box_arr = np.asarray(box, dtype=np.float32) if box is not None else None
        mask_arr = _sam_mask_input(mask_input)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_label_arr,
            box=box_arr,
            mask_input=mask_arr,
            multimask_output=True,
        )
        return Prediction(
            masks=[m.astype(np.uint8) for m in masks],
            scores=[float(s) for s in scores],
            logits=[l.astype(np.float32) for l in logits],
        )

    @staticmethod
    def reference_box(mask: np.ndarray) -> list[int]:
        box = mask_to_box(mask, expand_ratio=0.05)
        if box is None:
            raise ValueError("Reference mask is empty.")
        return box


def _sam_mask_input(mask_input: np.ndarray | None) -> np.ndarray | None:
    if mask_input is None:
        return None
    mask = (np.asarray(mask_input) > 0).astype(np.float32)
    if mask.shape != (256, 256):
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        mask = (np.asarray(image) > 0).astype(np.float32)
    # SamPredictor expects low-resolution logits with shape [1, 256, 256].
    logits = np.where(mask > 0, 20.0, -20.0).astype(np.float32)
    return logits[None, :, :]

