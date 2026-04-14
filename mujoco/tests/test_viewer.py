from __future__ import annotations

import contextlib

import mujoco
import numpy as np

from mujoco_servo.viewer import InteractiveViewer, initial_viewer_pose
from mujoco_servo.types import Pose


class FakeHandle:
    def __init__(self) -> None:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        self.viewport = mujoco.MjrRect(0, 0, 1280, 720)
        self.images: list[object] = []
        self.texts: list[object] = []
        self.user_scn = mujoco.MjvScene(model, 50)

    def set_images(self, images) -> None:  # noqa: ANN001
        self.images.append(images)

    def clear_images(self) -> None:
        self.images.append("cleared")

    def set_texts(self, texts) -> None:  # noqa: ANN001
        self.texts.append(texts)

    def clear_texts(self) -> None:
        self.texts.append("cleared")

    def sync(self) -> None:
        pass

    def is_running(self) -> bool:
        return True

    def lock(self):  # noqa: ANN001
        return contextlib.nullcontext()

    def close(self) -> None:
        pass


def test_initial_viewer_pose_prefers_task_specific_distance() -> None:
    fixed = initial_viewer_pose("t2-fixed")
    eye_in_hand = initial_viewer_pose("t2-eye")
    ibvs = initial_viewer_pose("t3-ibvs")
    live = initial_viewer_pose("live", live_mode=True)

    assert fixed[1] > ibvs[1] > eye_in_hand[1]
    assert live[1] > eye_in_hand[1]
    assert fixed[0].shape == (3,)


def test_set_image_panel_places_overlay_in_viewer_corner() -> None:
    handle = FakeHandle()
    viewer = InteractiveViewer(handle=handle)
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    viewer.set_image_panel(image)

    assert handle.images
    rect, resized = handle.images[-1]
    assert isinstance(rect, mujoco.MjrRect)
    assert rect.left > 0
    assert rect.bottom > 0
    assert rect.width <= 320
    assert rect.height <= 180
    assert resized.shape[2] == 3


def test_set_image_panels_supports_multiple_overlays() -> None:
    handle = FakeHandle()
    viewer = InteractiveViewer(handle=handle)
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    viewer.set_image_panels([image, image])

    assert handle.images
    panels = handle.images[-1]
    assert isinstance(panels, list)
    assert len(panels) == 2
    assert all(isinstance(rect, mujoco.MjrRect) for rect, _ in panels)


def test_set_motion_traces_populates_user_scene() -> None:
    handle = FakeHandle()
    viewer = InteractiveViewer(handle=handle)
    target = [np.array([0.0, 0.0, 0.4]), np.array([0.1, 0.0, 0.4]), np.array([0.2, 0.0, 0.4])]
    ee = [np.array([0.0, 0.2, 0.5]), np.array([0.1, 0.2, 0.5])]
    target_pose = Pose(np.array([0.2, 0.0, 0.4]), np.eye(3))
    ee_pose = Pose(np.array([0.1, 0.2, 0.5]), np.eye(3))

    viewer.set_motion_traces(
        target,
        ee,
        current_target=target[-1],
        current_ee=ee[-1],
        target_pose=target_pose,
        ee_pose=ee_pose,
        workspace_center=np.array([0.2, 0.0, 0.4]),
        workspace_size=np.array([0.9, 0.7, 0.004]),
    )

    assert handle.user_scn.ngeom >= 12
    assert handle.user_scn.geoms[0].category == mujoco.mjtCatBit.mjCAT_DECOR
    assert np.any(handle.user_scn.geomorder[: handle.user_scn.ngeom] >= 0)
