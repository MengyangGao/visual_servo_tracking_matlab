from __future__ import annotations

import numpy as np

from mujoco_servo.config import BoardConfig
from mujoco_servo.geometry import look_at
from mujoco_servo.image_features import charuco_pose_from_image, make_charuco_board, project_board_corners, render_charuco_board_image
from mujoco_servo.types import CameraIntrinsics, Pose


def test_charuco_detection_and_pose_estimation(tmp_path) -> None:
    board_cfg = BoardConfig()
    image_path = tmp_path / "board.png"
    image = render_charuco_board_image(board_cfg, image_path)
    intrinsics = CameraIntrinsics(width=image.shape[1], height=image.shape[0], fx=900.0, fy=900.0)
    pose, corners, ids = charuco_pose_from_image(image, board_cfg, intrinsics)
    assert image_path.exists()
    assert len(ids) > 0
    assert corners.shape[1] == 2
    assert pose is not None


def test_project_board_corners() -> None:
    board_cfg = BoardConfig()
    intrinsics = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0)
    camera_pose = look_at(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    board_pose = Pose(np.array([0.0, 0.0, 0.0]), np.eye(3))
    points = project_board_corners(board_cfg, intrinsics, camera_pose, board_pose)
    assert points.shape == (4, 2)

