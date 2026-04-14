from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mujoco_servo import perception
from mujoco_servo.perception import GroundingDinoDetector, Sam2MaskGenerator


class _Batch(dict):
    def to(self, device):
        for key, value in list(self.items()):
            if torch.is_tensor(value):
                self[key] = value.to(device)
        return self

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _GroundingProcessor:
    last_model_id: str | None = None

    @classmethod
    def from_pretrained(cls, model_id: str):
        cls.last_model_id = model_id
        return cls()

    def __call__(self, images, text, return_tensors):  # noqa: ANN001
        del images, text, return_tensors
        return _Batch(
            pixel_values=torch.ones((1, 3, 8, 8), dtype=torch.float32),
            input_ids=torch.ones((1, 4), dtype=torch.int64),
        )

    def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes, text_labels):  # noqa: ANN001
        del outputs, input_ids, threshold, text_threshold, target_sizes, text_labels
        return [
            {
                "boxes": [np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)],
                "scores": [0.99],
                "labels": ["cup"],
            }
        ]


class _GroundingModel(torch.nn.Module):
    last_torch_dtype = None
    seen_input_dtype = None

    def __init__(self, torch_dtype):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch_dtype or torch.float32))

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype=None):
        del model_id
        cls.last_torch_dtype = torch_dtype
        return cls(torch_dtype)

    def forward(self, **inputs):  # noqa: ANN003
        self.__class__.seen_input_dtype = inputs["pixel_values"].dtype
        assert inputs["pixel_values"].dtype == next(self.parameters()).dtype
        return SimpleNamespace()


class _SamProcessor:
    last_model_id: str | None = None

    @classmethod
    def from_pretrained(cls, model_id: str):
        cls.last_model_id = model_id
        return cls()

    def __call__(self, images, input_boxes, return_tensors):  # noqa: ANN001
        del images, input_boxes, return_tensors
        return _Batch(
            pixel_values=torch.ones((1, 3, 8, 8), dtype=torch.float32),
            original_sizes=torch.tensor([[64, 64]], dtype=torch.int64),
        )

    def post_process_masks(self, pred_masks, original_sizes):  # noqa: ANN001
        del pred_masks, original_sizes
        return [np.ones((1, 64, 64), dtype=np.float32)]


class _SamModel(torch.nn.Module):
    last_dtype = None
    seen_input_dtype = None

    def __init__(self, torch_dtype):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1, dtype=torch_dtype or torch.float32))

    @classmethod
    def from_pretrained(cls, model_id: str, dtype=None):
        del model_id
        cls.last_dtype = dtype
        return cls(dtype)

    def forward(self, **inputs):  # noqa: ANN003
        self.__class__.seen_input_dtype = inputs["pixel_values"].dtype
        assert inputs["pixel_values"].dtype == next(self.parameters()).dtype
        return SimpleNamespace(pred_masks=torch.ones((1, 1, 8, 8), dtype=next(self.parameters()).dtype))


def test_grounding_dino_cpu_dtype_alignment(monkeypatch) -> None:
    monkeypatch.setattr(perception, "GroundingDinoProcessor", _GroundingProcessor)
    monkeypatch.setattr(perception, "GroundingDinoForObjectDetection", _GroundingModel)

    detector = GroundingDinoDetector(model_id="fake-grounding", device="cpu")
    detections = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), "cup")

    assert _GroundingModel.last_torch_dtype == torch.float32
    assert _GroundingModel.seen_input_dtype == torch.float32
    assert detections and detections[0].label == "cup"


def test_sam2_cpu_dtype_alignment(monkeypatch) -> None:
    monkeypatch.setattr(perception, "Sam2Processor", _SamProcessor)
    monkeypatch.setattr(perception, "AutoModelForMaskGeneration", _SamModel)

    segmenter = Sam2MaskGenerator(model_id="fake-sam", device="cpu")
    mask = segmenter.segment(np.zeros((64, 64, 3), dtype=np.uint8), np.array([10.0, 12.0, 24.0, 28.0], dtype=np.float64))

    assert _SamModel.last_dtype == torch.float32
    assert _SamModel.seen_input_dtype == torch.float32
    assert mask is not None
    assert mask.shape == (64, 64)
