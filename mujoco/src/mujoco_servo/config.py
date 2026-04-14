from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

from .geometry import look_at
from .types import CameraIntrinsics, Pose


def discover_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        panda_xml = parent / "reference" / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
        if panda_xml.exists():
            return parent
    raise FileNotFoundError("Could not locate repository root with reference/mujoco_menagerie.")


def _default_camera_backend() -> str:
    if sys.platform == "darwin":
        return "avfoundation"
    if sys.platform.startswith("linux"):
        return "v4l2"
    if sys.platform.startswith("win"):
        return "dshow"
    return "auto"


@dataclass(slots=True)
class BoardConfig:
    pattern_dims: tuple[int, int] = (7, 7)
    square_length_m: float = 0.03
    marker_length_m: float = 0.0225
    dictionary_name: str = "DICT_4X4_1000"
    image_size_px: tuple[int, int] = (2048, 2048)
    margin_px: int = 0
    border_bits: int = 1
    thickness_m: float = 0.006

    @property
    def width_m(self) -> float:
        return self.pattern_dims[0] * self.square_length_m

    @property
    def height_m(self) -> float:
        return self.pattern_dims[1] * self.square_length_m


@dataclass(slots=True)
class SimulationConfig:
    dt: float = 0.04
    steps: int = 180
    world_camera_name: str = "world_camera"
    world_camera_height: int = 480
    world_camera_width: int = 640
    target_height_m: float = 0.42
    target_offset_m: float = 0.48
    target_motion_radius_m: tuple[float, float] = (0.30, 0.22)
    target_motion_speed: float = 0.42
    position_gain: float = 4.2
    orientation_gain: float = 2.6
    ibvs_gain: float = 1.0
    ibvs_damping: float = 1e-3
    ik_damping: float = 1e-3
    joint_target_smoothing_alpha: float = 0.22
    joint_target_max_step_rad: float = 0.10
    align_position_threshold_m: float = 0.06
    align_feature_threshold: float = 0.04
    align_hold_frames: int = 6
    align_timeout_frames: int = 10
    hand_to_camera_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.11)
    hand_to_camera_rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    target_opening_angle_deg: float = 0.0
    gripper_ctrl: float = 255.0
    record_video: bool = False
    render_world: bool = False
    render_camera: bool = True
    display: bool = True


@dataclass(slots=True)
class LiveConfig:
    source: str = "camera"
    device_index: int = 0
    video_path: str | None = None
    camera_backend: str = field(default_factory=_default_camera_backend)
    camera_width: int = 1280
    camera_height: int = 720
    prompt: str = "charuco board"
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "facebook/sam2.1-hiera-tiny"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    device: str = "auto"
    inference_max_side: int = 960
    use_sam2: bool = True
    tracking_depth_m: float = 0.75
    standoff_m: float = 0.88
    target_box_fraction: float = 0.24
    render_world: bool = False
    max_frames: int = 0
    record_video: bool = False
    display: bool = True
    detection_smoothing_alpha: float = 0.16
    joint_target_smoothing_alpha: float = 0.20
    joint_target_max_step_rad: float = 0.10
    acquire_hold_frames: int = 3
    align_position_threshold_m: float = 0.05
    align_feature_threshold: float = 0.04
    align_hold_frames: int = 4
    align_timeout_frames: int = 6


@dataclass(slots=True)
class PathsConfig:
    repo_root: Path
    project_dir: Path
    reference_dir: Path
    panda_xml: Path
    resolved_panda_xml: Path
    outputs_dir: Path
    results_dir: Path
    cache_dir: Path
    generated_scene_xml: Path
    board_texture_png: Path
    logs_dir: Path
    plots_dir: Path
    calibration_dir: Path

    def ensure(self) -> None:
        for path in [
            self.outputs_dir,
            self.results_dir,
            self.cache_dir,
            self.logs_dir,
            self.plots_dir,
            self.calibration_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    board: BoardConfig = field(default_factory=BoardConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    camera: CameraIntrinsics | None = None
    target_center_m: np.ndarray = field(default_factory=lambda: np.array([0.34, 0.0, 0.0], dtype=np.float64))
    fixed_camera_pose: Pose = field(default_factory=Pose.identity)
    eye_camera_offset: Pose = field(default_factory=Pose.identity)

    def __post_init__(self) -> None:
        if self.camera is None:
            self.camera = CameraIntrinsics(width=1280, height=720, fx=900.0, fy=900.0)
        self.target_center_m = np.asarray(self.target_center_m, dtype=np.float64).reshape(3)
        self.target_center_m[2] = float(self.sim.target_height_m)
        self.fixed_camera_pose = self.fixed_camera_pose.copy()
        self.eye_camera_offset = self.eye_camera_offset.copy()


def build_default_config() -> AppConfig:
    repo_root = discover_repo_root()
    project_dir = repo_root / "mujoco"
    reference_dir = repo_root / "reference" / "mujoco_menagerie" / "franka_emika_panda"
    paths = PathsConfig(
        repo_root=repo_root,
        project_dir=project_dir,
        reference_dir=reference_dir,
        panda_xml=reference_dir / "panda.xml",
        resolved_panda_xml=project_dir / "outputs" / "cache" / "panda_resolved.xml",
        outputs_dir=project_dir / "outputs",
        results_dir=project_dir / "results",
        cache_dir=project_dir / "outputs" / "cache",
        generated_scene_xml=project_dir / "outputs" / "cache" / "panda_servo_scene.xml",
        board_texture_png=project_dir / "outputs" / "cache" / "charuco_board.png",
        logs_dir=project_dir / "outputs" / "logs",
        plots_dir=project_dir / "outputs" / "plots",
        calibration_dir=project_dir / "outputs" / "calibration",
    )
    paths.ensure()

    fixed_camera = look_at(
        eye=np.array([0.88, -0.92, 0.82], dtype=np.float64),
        target=np.array([0.34, 0.0, 0.42], dtype=np.float64),
        up=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    eye_offset = Pose.identity()
    target_center = np.array([0.34, 0.0, SimulationConfig().target_height_m], dtype=np.float64)

    return AppConfig(
        paths=paths,
        fixed_camera_pose=fixed_camera,
        eye_camera_offset=eye_offset,
        target_center_m=target_center,
    )
