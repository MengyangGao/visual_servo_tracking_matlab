from __future__ import annotations

import numpy as np

from mujoco_servo.control import ImageBasedServo, PoseBasedServo
from mujoco_servo.types import CameraIntrinsics, Pose


def test_pose_servo_zero_error() -> None:
    servo = PoseBasedServo()
    pose = Pose(np.zeros(3), np.eye(3))
    twist = servo.camera_twist(pose, pose)
    assert np.allclose(twist, 0.0)


def test_image_servo_zero_error() -> None:
    servo = ImageBasedServo()
    intrinsics = CameraIntrinsics(width=640, height=480, fx=400.0, fy=400.0)
    current = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]])
    twist = servo.camera_twist(current, current, intrinsics, depth=0.5)
    assert np.allclose(twist, 0.0)

