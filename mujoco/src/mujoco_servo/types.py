from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


def _array(value, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass(slots=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        self.position = _array(self.position, dtype=np.float64).reshape(3)
        self.rotation = _array(self.rotation, dtype=np.float64).reshape(3, 3)

    @classmethod
    def identity(cls) -> "Pose":
        return cls(np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))

    @classmethod
    def from_quat(cls, position: np.ndarray, quat_wxyz: np.ndarray) -> "Pose":
        from scipy.spatial.transform import Rotation

        quat_wxyz = _array(quat_wxyz, dtype=np.float64).reshape(4)
        rot = Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return cls(position, rot.as_matrix())

    def copy(self) -> "Pose":
        return Pose(self.position.copy(), self.rotation.copy())

    def as_matrix(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.position
        return mat

    def inverse(self) -> "Pose":
        rot_t = self.rotation.T
        return Pose(-(rot_t @ self.position), rot_t)

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        return self.rotation @ _array(point, dtype=np.float64).reshape(3) + self.position


@dataclass(slots=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float | None = None
    cy: float | None = None
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.fx = float(self.fx)
        self.fy = float(self.fy)
        self.cx = float(self.width / 2.0 if self.cx is None else self.cx)
        self.cy = float(self.height / 2.0 if self.cy is None else self.cy)
        self.distortion = _array(self.distortion, dtype=np.float64).reshape(-1)


@dataclass(slots=True)
class Detection:
    box: np.ndarray
    score: float
    label: str
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.box = _array(self.box, dtype=np.float64).reshape(4)
        self.score = float(self.score)
        self.label = str(self.label)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) * 0.5, (self.box[1] + self.box[3]) * 0.5], dtype=np.float64)


@dataclass(slots=True)
class FeatureObservation:
    points_px: np.ndarray
    ids: Optional[np.ndarray] = None
    score: float = 1.0
    label: str = ""
    depth_m: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.points_px = _array(self.points_px, dtype=np.float64).reshape(-1, 2)
        if self.ids is not None:
            self.ids = _array(self.ids, dtype=np.int32).reshape(-1)
        if self.depth_m is not None:
            self.depth_m = _array(self.depth_m, dtype=np.float64).reshape(-1)
        self.score = float(self.score)
        self.label = str(self.label)


@dataclass(slots=True)
class TrackingSample:
    time_s: float
    target_pose: Pose
    camera_pose: Pose
    ee_pose: Pose
    joint_positions: np.ndarray
    joint_targets: np.ndarray
    phase: str = ""
    pixel_error_norm: float = 0.0
    position_error_norm: float = 0.0
    feature_error_norm: float = 0.0
    detection: Optional[Detection] = None

    def __post_init__(self) -> None:
        self.time_s = float(self.time_s)
        self.joint_positions = _array(self.joint_positions, dtype=np.float64).reshape(-1)
        self.joint_targets = _array(self.joint_targets, dtype=np.float64).reshape(-1)
        self.phase = str(self.phase)
        self.pixel_error_norm = float(self.pixel_error_norm)
        self.position_error_norm = float(self.position_error_norm)
        self.feature_error_norm = float(self.feature_error_norm)


@dataclass(slots=True)
class CalibrationResult:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    reprojection_error: float
    num_frames: int
    output_dir: Path

    def __post_init__(self) -> None:
        self.camera_matrix = _array(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        self.distortion = _array(self.distortion, dtype=np.float64).reshape(-1)
        self.reprojection_error = float(self.reprojection_error)
        self.num_frames = int(self.num_frames)
        self.output_dir = Path(self.output_dir)


@dataclass(slots=True)
class RunSummary:
    name: str
    output_dir: Path
    samples: list[TrackingSample] = field(default_factory=list)
    calibration: Optional[CalibrationResult] = None

    def add_sample(self, sample: TrackingSample) -> None:
        self.samples.append(sample)
