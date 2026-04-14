from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def _backend_from_name(name: str) -> int | None:
    mapping = {
        "auto": None,
        "any": cv2.CAP_ANY,
        "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
        "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    }
    return mapping.get(name.lower(), None)


@dataclass(slots=True)
class VideoCaptureStream:
    source: int | str
    backend: str = "auto"
    width: int | None = None
    height: int | None = None
    capture: cv2.VideoCapture = field(init=False, repr=False)

    def __post_init__(self) -> None:
        backend_code = _backend_from_name(self.backend)
        if isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source, backend_code if backend_code is not None else cv2.CAP_ANY)
        else:
            self.capture = cv2.VideoCapture(str(self.source), backend_code if backend_code is not None else cv2.CAP_ANY)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open video source '{self.source}' with backend '{self.backend}'.")
        if self.width is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self.capture.read()
        if not ok:
            return False, None
        return True, frame

    def close(self) -> None:
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()


def maybe_resize(frame_bgr: np.ndarray, width: int | None = None, height: int | None = None) -> np.ndarray:
    if width is None and height is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if width is None:
        scale = height / h
        width = int(round(w * scale))
    elif height is None:
        scale = width / w
        height = int(round(h * scale))
    return cv2.resize(frame_bgr, (int(width), int(height)), interpolation=cv2.INTER_AREA)
