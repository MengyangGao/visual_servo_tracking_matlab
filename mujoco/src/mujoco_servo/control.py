from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import dls_solve, interaction_matrix, pose_error
from .types import CameraIntrinsics, Pose


@dataclass(slots=True)
class PoseBasedServo:
    position_gain: float = 2.8
    orientation_gain: float = 2.0
    damping: float = 1e-3

    def camera_twist(self, current: Pose, target: Pose) -> np.ndarray:
        pos_err, rot_err = pose_error(current, target)
        twist = np.concatenate([self.position_gain * pos_err, self.orientation_gain * rot_err])
        return twist


@dataclass(slots=True)
class ImageBasedServo:
    gain: float = 0.85
    damping: float = 1e-3

    def camera_twist(self, current_px: np.ndarray, desired_px: np.ndarray, intrinsics: CameraIntrinsics, depth: np.ndarray | float) -> np.ndarray:
        current_px = np.asarray(current_px, dtype=np.float64).reshape(-1, 2)
        desired_px = np.asarray(desired_px, dtype=np.float64).reshape(-1, 2)
        if current_px.shape != desired_px.shape:
            raise ValueError("Current and desired image points must have the same shape.")
        error = np.column_stack([
            (current_px[:, 0] - desired_px[:, 0]) / intrinsics.fx,
            (current_px[:, 1] - desired_px[:, 1]) / intrinsics.fy,
        ]).reshape(-1)
        jac = interaction_matrix(current_px, depth, intrinsics)
        twist = -self.gain * dls_solve(jac, error, damping=self.damping)
        return twist


def camera_twist_to_joint_velocity(jacobian: np.ndarray, camera_twist_world: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    return dls_solve(jacobian, np.asarray(camera_twist_world, dtype=np.float64).reshape(-1), damping=damping)


def pose_target_from_board(
    board_pose: Pose,
    distance_m: float,
    world_up: np.ndarray | None = None,
) -> Pose:
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    board_normal = board_pose.rotation[:, 2]
    camera_pos = board_pose.position + board_normal * float(distance_m)
    from .geometry import look_at

    return look_at(camera_pos, board_pose.position, world_up)


def board_feature_error(current_points_px: np.ndarray, desired_points_px: np.ndarray, intrinsics: CameraIntrinsics) -> float:
    current_points_px = np.asarray(current_points_px, dtype=np.float64).reshape(-1, 2)
    desired_points_px = np.asarray(desired_points_px, dtype=np.float64).reshape(-1, 2)
    if current_points_px.shape != desired_points_px.shape:
        raise ValueError("Current and desired board features must have the same shape.")
    diff = np.column_stack(
        [
            (current_points_px[:, 0] - desired_points_px[:, 0]) / intrinsics.fx,
            (current_points_px[:, 1] - desired_points_px[:, 1]) / intrinsics.fy,
        ]
    )
    return float(np.linalg.norm(diff))
