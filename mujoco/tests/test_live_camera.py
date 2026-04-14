from __future__ import annotations

import cv2
import numpy as np

from mujoco_servo.config import build_default_config
from mujoco_servo import runtime
from mujoco_servo.runtime import run_live_camera
from mujoco_servo.types import Detection


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.closed = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def close(self) -> None:
        self.closed = True


class FakeTracker:
    def __init__(self) -> None:
        self.index = 0

    def observe(self, image_bgr: np.ndarray, prompt: str) -> Detection:
        del prompt
        height, width = image_bgr.shape[:2]
        box_w = max(40, width // 5)
        box_h = max(30, height // 6)
        x0 = 30 + 12 * self.index
        y0 = 40
        x1 = min(width - 1, x0 + box_w)
        y1 = min(height - 1, y0 + box_h)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        self.index += 1
        return Detection(box=np.array([x0, y0, x1, y1], dtype=np.float64), score=0.95, label="cup", mask=mask)


class FakeWorldRenderer:
    def __init__(self, model, height, width, camera_name) -> None:  # noqa: ANN001
        del model, height, width, camera_name
        self.closed = False

    def render(self, data):  # noqa: ANN001
        del data
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeInteractiveViewer:
    def __init__(self) -> None:
        self.statuses: list[list[str]] = []
        self.image_panels: list[tuple[tuple[int, ...], ...]] = []
        self.motion_traces: list[tuple[int, int]] = []
        self.synced = 0
        self.closed = False

    def set_status(self, lines) -> None:  # noqa: ANN001
        self.statuses.append(list(lines))

    def set_image_panels(self, images, *, margin=16, gap=12) -> None:  # noqa: ANN001
        del margin, gap
        self.image_panels.append(tuple(tuple(image.shape) for image in images))

    def set_image_panel(self, image_bgr, *, margin=16) -> None:  # noqa: ANN001
        del margin
        self.set_image_panels([image_bgr])

    def set_motion_traces(self, target_positions, ee_positions, *, trail_length=12, current_target=None, current_ee=None, target_pose=None, ee_pose=None, workspace_center=None, workspace_size=None) -> None:  # noqa: ANN001
        del trail_length, current_target, current_ee, target_pose, ee_pose, workspace_center, workspace_size
        self.motion_traces.append((len(target_positions), len(ee_positions)))

    def sync(self) -> None:
        self.synced += 1

    def is_running(self) -> bool:
        return not self.closed

    def close(self) -> None:
        self.closed = True


def test_live_camera_gui_and_control_smoke(monkeypatch) -> None:
    cfg = build_default_config()
    cfg.live.display = True
    cfg.live.render_world = False
    cfg.live.record_video = False
    cfg.live.max_frames = 2
    cfg.live.acquire_hold_frames = 1
    cfg.camera.width = 320
    cfg.camera.height = 240
    cfg.camera.fx = 250.0
    cfg.camera.fy = 250.0

    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(2)]
    capture = FakeCapture(frames)
    tracker = FakeTracker()
    viewer = FakeInteractiveViewer()
    shown: list[tuple[str, tuple[int, ...]]] = []
    waits: list[int] = []

    monkeypatch.setattr(runtime, "launch_interactive_viewer", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append((name, tuple(image.shape))))
    monkeypatch.setattr(cv2, "waitKey", lambda delay=1: waits.append(delay) or 255)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: shown.append(("destroy", (0,))))

    result = run_live_camera(cfg=cfg, prompt="cup", capture=capture, tracker=tracker, display=True, max_frames=2)

    assert capture.closed
    assert len(result.samples) == 2
    assert any(name == "live_camera" for name, _ in shown)
    assert len(waits) == 2
    assert result.samples[0].detection is not None
    assert result.samples[0].detection.label == "cup"
    assert result.samples[0].phase == "acquiring"
    assert result.samples[1].phase in {"aligning", "tracking"}
    assert not np.allclose(result.samples[0].joint_targets, result.samples[1].joint_targets)
    assert viewer.closed
    assert viewer.synced == 2
    assert len(viewer.image_panels) == 2
    assert len(viewer.image_panels[0]) == 2
    assert viewer.image_panels[0][0][2] == 3
    assert len(viewer.motion_traces) == 2
    assert any(name == "world_map" for name, _ in shown)


def test_live_camera_prefers_standard_viewer(monkeypatch) -> None:
    cfg = build_default_config()
    cfg.live.display = True
    cfg.live.render_world = True
    cfg.live.record_video = False
    cfg.live.max_frames = 1
    cfg.camera.width = 320
    cfg.camera.height = 240
    cfg.camera.fx = 250.0
    cfg.camera.fy = 250.0

    frames = [np.zeros((240, 320, 3), dtype=np.uint8)]
    capture = FakeCapture(frames)
    tracker = FakeTracker()
    shown: list[tuple[str, tuple[int, ...]]] = []
    viewer = FakeInteractiveViewer()

    monkeypatch.setattr(runtime, "launch_interactive_viewer", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(runtime, "MuJoCoWorldRenderer", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("world renderer should not be created when viewer is available")))
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append((name, tuple(image.shape))))
    monkeypatch.setattr(cv2, "waitKey", lambda delay=1: 255)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    result = run_live_camera(cfg=cfg, prompt="cup", capture=capture, tracker=tracker, display=True, max_frames=1)

    assert capture.closed
    assert len(result.samples) == 1
    assert any(name == "live_camera" for name, _ in shown)
    assert not any(name == "world" for name, _ in shown)
    assert viewer.closed
    assert viewer.synced == 1
    assert viewer.statuses
    assert viewer.statuses[0][0] == "mode: live-camera"
    assert any(line.startswith("phase:") for line in viewer.statuses[0])
    assert len(viewer.image_panels) == 1
    assert len(viewer.image_panels[0]) == 2
    assert viewer.motion_traces
