from __future__ import annotations

import numpy as np

from mujoco_servo.config import BoardConfig
from mujoco_servo.geometry import board_corner_points, dls_solve, look_at, mujoco_camera_pose_from_internal_pose, project_points
from mujoco_servo.types import CameraIntrinsics


def test_board_corner_points_shape() -> None:
    corners = board_corner_points(0.21, 0.15)
    assert corners.shape == (4, 3)
    assert np.isclose(np.ptp(corners[:, 0]), 0.21)
    assert np.isclose(np.ptp(corners[:, 1]), 0.15)


def test_look_at_projects_origin_to_image_center() -> None:
    intrinsics = CameraIntrinsics(width=640, height=480, fx=400.0, fy=400.0)
    camera = look_at(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    points = project_points(np.array([[0.0, 0.0, 0.0]]), intrinsics, camera)
    assert np.allclose(points[0], [intrinsics.cx, intrinsics.cy], atol=1e-6)


def test_mujoco_camera_pose_flips_y_and_z_axes() -> None:
    internal = look_at(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    mujoco_camera = mujoco_camera_pose_from_internal_pose(internal)
    expected = internal.rotation @ np.diag([1.0, -1.0, -1.0])
    assert np.allclose(mujoco_camera.rotation, expected)


def test_dls_solve_zero_rhs() -> None:
    jac = np.eye(3)
    target = np.zeros(3)
    sol = dls_solve(jac, target, damping=1e-3)
    assert np.allclose(sol, 0.0)
