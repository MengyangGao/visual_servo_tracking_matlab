from __future__ import annotations

import numpy as np

from mujoco_servo.config import build_default_config
from mujoco_servo import runtime
from mujoco_servo.geometry import look_at
from mujoco_servo.runtime import run_simulation
from mujoco_servo.types import CameraIntrinsics


class FakeInteractiveViewer:
    def __init__(self) -> None:
        self.statuses: list[list[str]] = []
        self.motion_traces: list[tuple[int, int]] = []
        self.image_panels: list[tuple[tuple[int, ...], ...]] = []
        self.synced = 0
        self.closed = False

    def set_status(self, lines) -> None:  # noqa: ANN001
        self.statuses.append(list(lines))

    def set_motion_traces(
        self,
        target_positions,
        ee_positions,
        *,
        trail_length=24,
        current_target=None,
        current_ee=None,
        target_pose=None,
        ee_pose=None,
        workspace_center=None,
        workspace_size=None,
    ) -> None:  # noqa: ANN001
        del trail_length, current_target, current_ee, target_pose, ee_pose, workspace_center, workspace_size
        self.motion_traces.append((len(target_positions), len(ee_positions)))

    def set_image_panels(self, images, *, margin=16, gap=12) -> None:  # noqa: ANN001
        del margin, gap
        self.image_panels.append(tuple(tuple(image.shape) for image in images))

    def set_image_panel(self, image_bgr, *, margin=16) -> None:  # noqa: ANN001
        del margin
        self.set_image_panels([image_bgr])

    def sync(self) -> None:
        self.synced += 1

    def is_running(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


def test_simulation_smoke_run() -> None:
    cfg = build_default_config()
    cfg.sim.steps = 3
    cfg.sim.display = False
    cfg.sim.render_world = False
    cfg.sim.record_video = False
    cfg.camera = CameraIntrinsics(width=320, height=240, fx=250.0, fy=250.0)
    result = run_simulation(cfg=cfg, task="t2-fixed", display=False)
    assert len(result.samples) == 3
    assert result.output_dir.exists()
    assert result.samples[0].phase in {"aligning", "tracking"}
    assert all(np.isclose(sample.target_pose.position[2], cfg.sim.target_height_m) for sample in result.samples)
    assert np.allclose(result.samples[0].target_pose.rotation[:, 2], np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_simulation_prefers_standard_viewer(monkeypatch) -> None:
    cfg = build_default_config()
    cfg.sim.steps = 1
    cfg.sim.display = True
    cfg.sim.render_world = True
    cfg.sim.record_video = False
    cfg.camera = CameraIntrinsics(width=320, height=240, fx=250.0, fy=250.0)

    viewer = FakeInteractiveViewer()
    shown: list[tuple[str, tuple[int, ...]]] = []

    monkeypatch.setattr(runtime, "launch_interactive_viewer", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(runtime, "MuJoCoWorldRenderer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("world renderer should not be created when viewer is available")))
    monkeypatch.setattr(runtime.cv2, "imshow", lambda name, image: shown.append((name, tuple(image.shape))))
    monkeypatch.setattr(runtime.cv2, "waitKey", lambda delay=1: 255)
    monkeypatch.setattr(runtime.cv2, "destroyAllWindows", lambda: None)

    result = run_simulation(cfg=cfg, task="t2-fixed", display=True)

    assert len(result.samples) == 1
    assert viewer.closed
    assert viewer.synced == 1
    assert viewer.statuses
    assert viewer.statuses[0][0] == "mode: simulation/t2-fixed"
    assert any(line.startswith("phase:") for line in viewer.statuses[0])
    assert viewer.motion_traces
    assert viewer.image_panels and len(viewer.image_panels[0]) == 2
    assert any(name == "camera" for name, _ in shown)
    assert any(name == "map" for name, _ in shown)
    assert not any(name == "world" for name, _ in shown)


def test_ray_plane_intersection_projects_to_fixed_height_plane() -> None:
    intrinsics = CameraIntrinsics(width=640, height=480, fx=400.0, fy=400.0)
    camera_pose = look_at(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.42]), np.array([0.0, 1.0, 0.0]))
    point = runtime._ray_plane_intersection_world(np.array([intrinsics.cx, intrinsics.cy]), camera_pose, intrinsics, 0.42)
    assert np.allclose(point, np.array([0.0, 0.0, 0.42]), atol=1e-6)


def test_lowpass_vector_limits_step_size() -> None:
    previous = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    current = np.array([10.0, -10.0, 1.0], dtype=np.float64)
    blended = runtime._lowpass_vector(previous, current, alpha=0.5, max_step=0.2)
    assert np.allclose(blended, np.array([0.2, -0.2, 0.2], dtype=np.float64))
