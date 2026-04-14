from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .types import CameraIntrinsics, Pose


def _arr(value, dtype=np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vec = _arr(vec, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return vec / norm


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = _arr(vec, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(_arr(rot, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = _arr(quat_wxyz, dtype=np.float64).reshape(4)
    return Rotation.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]).as_matrix()


def pose_from_matrix(mat: np.ndarray) -> Pose:
    mat = _arr(mat, dtype=np.float64).reshape(4, 4)
    return Pose(mat[:3, 3], mat[:3, :3])


def compose(parent: Pose, child: Pose) -> Pose:
    rot = parent.rotation @ child.rotation
    pos = parent.rotation @ child.position + parent.position
    return Pose(pos, rot)


def inverse_pose(pose: Pose) -> Pose:
    return pose.inverse()


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Pose:
    """Return a camera pose whose +z axis looks at the target.

    The camera convention here is:
    - +x points to the right in the image
    - +y points down in the image
    - +z points forward from the camera
    """

    eye = _arr(eye, dtype=np.float64).reshape(3)
    target = _arr(target, dtype=np.float64).reshape(3)
    up = normalize(up)

    forward = normalize(target - eye)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if np.linalg.norm(np.cross(forward, alt_up)) < 1e-9:
            alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(forward, alt_up)
    right = normalize(right)
    down = normalize(np.cross(forward, right))
    rot = np.column_stack([right, down, forward])
    return Pose(eye, rot)


def mujoco_camera_pose_from_internal_pose(pose: Pose) -> Pose:
    """Convert the project's internal camera pose to MuJoCo's camera convention.

    The internal convention used by `look_at` and the image projection helpers
    treats +z as forward and +y as down. MuJoCo cameras use +x right, +y up,
    and look along -z, so the y and z axes need to be flipped before exporting
    the pose into MJCF/XML.
    """

    rot = pose.rotation @ np.diag([1.0, -1.0, -1.0])
    return Pose(pose.position.copy(), rot)


def world_to_camera(points_world: np.ndarray, camera_pose: Pose) -> np.ndarray:
    points_world = _arr(points_world, dtype=np.float64).reshape(-1, 3)
    return (camera_pose.rotation.T @ (points_world.T - camera_pose.position[:, None])).T


def project_points(points_world: np.ndarray, intrinsics: CameraIntrinsics, camera_pose: Pose) -> np.ndarray:
    points_cam = world_to_camera(points_world, camera_pose)
    z = np.clip(points_cam[:, 2], 1e-9, None)
    u = intrinsics.fx * (points_cam[:, 0] / z) + intrinsics.cx
    v = intrinsics.fy * (points_cam[:, 1] / z) + intrinsics.cy
    return np.column_stack([u, v])


def normalize_image_points(points_px: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    points_px = _arr(points_px, dtype=np.float64).reshape(-1, 2)
    x = (points_px[:, 0] - intrinsics.cx) / intrinsics.fx
    y = (points_px[:, 1] - intrinsics.cy) / intrinsics.fy
    return np.column_stack([x, y])


def interaction_matrix(points_px: np.ndarray, depths: np.ndarray | float, intrinsics: CameraIntrinsics) -> np.ndarray:
    points_px = _arr(points_px, dtype=np.float64).reshape(-1, 2)
    points_norm = normalize_image_points(points_px, intrinsics)
    if np.isscalar(depths):
        depths_arr = np.full(points_norm.shape[0], float(depths), dtype=np.float64)
    else:
        depths_arr = _arr(depths, dtype=np.float64).reshape(-1)
    if depths_arr.shape[0] != points_norm.shape[0]:
        raise ValueError("Depth array length must match number of points.")

    rows = []
    for (x, y), z in zip(points_norm, depths_arr):
        rows.append(np.array([-1.0 / z, 0.0, x / z, x * y, -(1.0 + x * x), y], dtype=np.float64))
        rows.append(np.array([0.0, -1.0 / z, y / z, 1.0 + y * y, -x * y, -x], dtype=np.float64))
    return np.vstack(rows)


def dls_solve(jacobian: np.ndarray, target: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    jacobian = _arr(jacobian, dtype=np.float64)
    target = _arr(target, dtype=np.float64).reshape(-1)
    jtj = jacobian.T @ jacobian
    eye = np.eye(jtj.shape[0], dtype=np.float64)
    return np.linalg.solve(jtj + (damping * damping) * eye, jacobian.T @ target)


def pose_error(current: Pose, target: Pose) -> tuple[np.ndarray, np.ndarray]:
    position_error = target.position - current.position
    rotation_error = Rotation.from_matrix(target.rotation @ current.rotation.T).as_rotvec()
    return position_error, rotation_error


def clamp_to_limits(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    value = _arr(value, dtype=np.float64).reshape(-1)
    lower = _arr(lower, dtype=np.float64).reshape(-1)
    upper = _arr(upper, dtype=np.float64).reshape(-1)
    return np.minimum(np.maximum(value, lower), upper)


def homogeneous(point: np.ndarray) -> np.ndarray:
    point = _arr(point, dtype=np.float64).reshape(3)
    return np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)


def board_corner_points(width_m: float, height_m: float) -> np.ndarray:
    half_w = 0.5 * float(width_m)
    half_h = 0.5 * float(height_m)
    return np.array(
        [
            [-half_w, half_h, 0.0],
            [half_w, half_h, 0.0],
            [half_w, -half_h, 0.0],
            [-half_w, -half_h, 0.0],
        ],
        dtype=np.float64,
    )


def board_pose_from_center(center: np.ndarray, normal: np.ndarray, up_hint: np.ndarray = np.array([0.0, 0.0, 1.0])) -> Pose:
    center = _arr(center, dtype=np.float64).reshape(3)
    normal = normalize(normal)
    up_hint = normalize(up_hint)
    right = np.cross(up_hint, normal)
    if np.linalg.norm(right) < 1e-9:
        right = np.cross(np.array([0.0, 1.0, 0.0]), normal)
    right = normalize(right)
    down = normalize(np.cross(normal, right))
    rot = np.column_stack([right, down, normal])
    return Pose(center, rot)


def average_point(points: np.ndarray) -> np.ndarray:
    points = _arr(points, dtype=np.float64).reshape(-1, 2)
    return points.mean(axis=0)
