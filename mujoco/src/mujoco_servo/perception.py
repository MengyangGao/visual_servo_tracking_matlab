from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

try:
    import torch
    from PIL import Image
    from transformers import (
        AutoModelForMaskGeneration,
        GroundingDinoForObjectDetection,
        GroundingDinoProcessor,
        Sam2Processor,
    )
except Exception:  # pragma: no cover - optional dependency path
    torch = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    AutoModelForMaskGeneration = None  # type: ignore[assignment]
    GroundingDinoForObjectDetection = None  # type: ignore[assignment]
    GroundingDinoProcessor = None  # type: ignore[assignment]
    Sam2Processor = None  # type: ignore[assignment]

from .config import BoardConfig
from .image_features import bbox_from_mask, mask_centroid
from .types import CameraIntrinsics, Detection, FeatureObservation, Pose


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.models\.grounding_dino\.processing_grounding_dino",
)


def _select_device(device: str) -> str:
    if device != "auto":
        return device
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def _torch_dtype_for_device(device: str):
    if torch is None:
        return None
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float32
    return torch.float32


def _move_float_tensors(batch: Any, dtype: Any) -> Any:
    if torch is None or dtype is None:
        return batch
    items = getattr(batch, "items", None)
    if items is None:
        return batch
    for key, value in list(batch.items()):
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(dtype=dtype)
    return batch


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _normalize_prompt(prompt: str) -> str:
    parts = _split_prompt_labels(prompt)
    if not parts:
        return ""
    return ". ".join(parts) + "."


def _split_prompt_labels(prompt: str) -> list[str]:
    return [part.strip().lower().rstrip(".") for part in re.split(r"[;,/|\n]+", prompt) if part.strip()]


def _resize_max_side(image_bgr: np.ndarray, max_side: int | None) -> tuple[np.ndarray, float]:
    if max_side is None or max_side <= 0:
        return image_bgr, 1.0
    height, width = image_bgr.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return image_bgr, 1.0
    scale = float(max_side) / float(long_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _scale_box(box: np.ndarray, scale: float) -> np.ndarray:
    box = np.asarray(box, dtype=np.float64).reshape(4)
    if scale == 1.0:
        return box
    return box / float(scale)


def _scale_mask(mask: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.shape[:2] == output_shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (int(output_shape[1]), int(output_shape[0])), interpolation=cv2.INTER_NEAREST) > 0


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(4)
    b = np.asarray(b, dtype=np.float64).reshape(4)
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
    return float(inter / (area_a + area_b - inter))


def _select_detection(detections: list[Detection], previous: Detection | None = None) -> Detection:
    if not detections:
        raise ValueError("No detections available.")
    if previous is None:
        return max(detections, key=lambda det: det.score)
    return max(detections, key=lambda det: 0.7 * det.score + 0.3 * _box_iou(det.box, previous.box))


@dataclass(slots=True)
class GroundingDinoDetector:
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    device: str = "auto"
    processor: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    model_dtype: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if GroundingDinoProcessor is None or GroundingDinoForObjectDetection is None:
            raise ImportError("GroundingDINO dependencies are not available. Install the open-vocab extra.")
        if torch is None:
            raise ImportError("PyTorch is not available.")
        self.device = _select_device(self.device)
        dtype = _torch_dtype_for_device(self.device)
        self.processor = GroundingDinoProcessor.from_pretrained(self.model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(self.model_id, torch_dtype=dtype).to(self.device)
        self.model_dtype = next(self.model.parameters()).dtype
        self.model.eval()

    def detect(self, image_bgr: np.ndarray, prompt: str, max_side: int | None = None) -> list[Detection]:
        if Image is None:
            raise ImportError("Pillow is not available.")
        labels = _split_prompt_labels(prompt)
        prompt = _normalize_prompt(prompt)
        if not prompt:
            return []
        resized_bgr, scale = _resize_max_side(image_bgr, max_side)
        image_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        inputs = _move_float_tensors(inputs, self.model_dtype)
        input_ids = inputs.input_ids.detach().cpu() if torch.is_tensor(inputs.input_ids) else inputs.input_ids
        with torch.no_grad():
            outputs = self.model(**inputs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image_rgb.shape[:2]],
                text_labels=[labels],
            )[0]
        detections: list[Detection] = []
        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels = results.get("labels", [])
        for box, score, label in zip(boxes, scores, labels):
            detections.append(
                Detection(
                    box=_scale_box(_to_numpy(box).astype(np.float64, copy=False), scale),
                    score=float(_to_numpy(score).reshape(())),
                    label=str(label),
                )
            )
        return detections

    def best(self, image_bgr: np.ndarray, prompt: str, max_side: int | None = None, previous: Detection | None = None) -> Detection | None:
        detections = self.detect(image_bgr, prompt, max_side=max_side)
        if not detections:
            return None
        return _select_detection(detections, previous=previous)


@dataclass(slots=True)
class Sam2MaskGenerator:
    model_id: str = "facebook/sam2.1-hiera-tiny"
    device: str = "auto"
    processor: Any = field(init=False, repr=False)
    model: Any = field(init=False, repr=False)
    model_dtype: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if Sam2Processor is None or AutoModelForMaskGeneration is None:
            raise ImportError("SAM2 dependencies are not available. Install the open-vocab extra.")
        if torch is None:
            raise ImportError("PyTorch is not available.")
        self.device = _select_device(self.device)
        dtype = _torch_dtype_for_device(self.device)
        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = AutoModelForMaskGeneration.from_pretrained(self.model_id, dtype=dtype).to(self.device)
        self.model_dtype = next(self.model.parameters()).dtype
        self.model.eval()

    def segment(self, image_bgr: np.ndarray, box_xyxy: np.ndarray, max_side: int | None = None) -> np.ndarray | None:
        if Image is None:
            raise ImportError("Pillow is not available.")
        resized_bgr, scale = _resize_max_side(image_bgr, max_side)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        box = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
        if scale != 1.0:
            box = box * scale
        input_boxes = [[[float(box[0]), float(box[1]), float(box[2]), float(box[3])]]]
        inputs = self.processor(images=resized_rgb, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        inputs = _move_float_tensors(inputs, self.model_dtype)
        original_sizes = inputs["original_sizes"].detach().cpu() if torch.is_tensor(inputs["original_sizes"]) else inputs["original_sizes"]
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)[0]
        if masks is None or len(masks) == 0:
            return None
        mask = np.squeeze(_to_numpy(masks[0])).astype(np.float32, copy=False) > 0.5
        if scale != 1.0:
            mask = _scale_mask(mask, image_bgr.shape[:2])
        return mask.astype(np.uint8)


@dataclass(slots=True)
class OpenVocabularyTracker:
    detector: GroundingDinoDetector
    segmenter: Sam2MaskGenerator | None = None
    max_side: int | None = None
    last_detection: Detection | None = None

    def observe(self, image_bgr: np.ndarray, prompt: str) -> Detection | None:
        detection = self.detector.best(image_bgr, prompt, max_side=self.max_side, previous=self.last_detection)
        if detection is None:
            return None
        if self.segmenter is not None:
            mask = self.segmenter.segment(image_bgr, detection.box, max_side=self.max_side)
            if mask is not None:
                detection.mask = mask
                mask_box = bbox_from_mask(mask)
                if mask_box is not None:
                    detection.box = mask_box
        self.last_detection = detection
        return detection


@dataclass(slots=True)
class CharucoTracker:
    board_cfg: BoardConfig

    def detect(self, image_bgr: np.ndarray) -> FeatureObservation | None:
        corners, ids = charuco_detect(image_bgr, self.board_cfg)
        if len(ids) == 0:
            return None
        return FeatureObservation(points_px=corners, ids=ids, score=1.0, label="charuco")

    def estimate_pose(self, image_bgr: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[Pose | None, FeatureObservation | None]:
        pose, corners, ids = charuco_pose_from_image(image_bgr, self.board_cfg, intrinsics)
        if len(ids) == 0:
            return pose, None
        return pose, FeatureObservation(points_px=corners, ids=ids, score=1.0, label="charuco")


def prompt_detection_or_charuco(
    image_bgr: np.ndarray,
    prompt: str,
    board_tracker: CharucoTracker,
    detector: GroundingDinoDetector | None = None,
    intrinsics: CameraIntrinsics | None = None,
) -> tuple[Detection | None, FeatureObservation | None, Pose | None]:
    if detector is not None:
        det = detector.best(image_bgr, prompt)
    else:
        det = None
    if intrinsics is None:
        return det, board_tracker.detect(image_bgr), None
    pose, board_obs = board_tracker.estimate_pose(image_bgr, intrinsics)
    return det, board_obs, pose
