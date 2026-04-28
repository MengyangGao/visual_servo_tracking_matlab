from __future__ import annotations

import math

import numpy as np


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        if fallback is None:
            return np.zeros_like(value)
        return normalize(np.asarray(fallback, dtype=float))
    return value / norm


def damped_pseudo_inverse(matrix: np.ndarray, damping: float) -> np.ndarray:
    jac = np.asarray(matrix, dtype=float)
    rows = jac.shape[0]
    lam2 = float(damping) ** 2
    return jac.T @ np.linalg.inv(jac @ jac.T + lam2 * np.eye(rows))


def rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    mat = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(mat))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ],
            dtype=float,
        )
    index = int(np.argmax(np.diag(mat)))
    if index == 0:
        s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        quat = np.array(
            [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ],
            dtype=float,
        )
    elif index == 1:
        s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        quat = np.array(
            [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ],
            dtype=float,
        )
    else:
        s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        quat = np.array(
            [
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )
    return quat / max(float(np.linalg.norm(quat)), 1e-9)


def look_at_xyaxes(camera_pos: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize(np.asarray(target_pos, dtype=float) - np.asarray(camera_pos, dtype=float), np.array([1.0, 0.0, 0.0]))
    up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
    right = normalize(np.cross(forward, up_hint), np.array([1.0, 0.0, 0.0]))
    up = normalize(np.cross(right, forward), np.array([0.0, 0.0, 1.0]))
    return right, up


def clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= float(max_norm) or norm < 1e-12:
        return value
    return value * (float(max_norm) / norm)


def rotation_error_vector(desired_world_from_body: np.ndarray, current_world_from_body: np.ndarray) -> np.ndarray:
    error = np.asarray(desired_world_from_body, dtype=float).reshape(3, 3) @ np.asarray(current_world_from_body, dtype=float).reshape(3, 3).T
    cos_angle = np.clip((np.trace(error) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-7:
        return np.zeros(3, dtype=float)
    axis = np.array(
        [
            error[2, 1] - error[1, 2],
            error[0, 2] - error[2, 0],
            error[1, 0] - error[0, 1],
        ],
        dtype=float,
    )
    axis = axis / max(2.0 * np.sin(angle), 1e-9)
    return axis * angle


def tool_z_facing_rotation(forward_world: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    z_axis = normalize(forward_world, np.array([1.0, 0.0, 0.0]))
    up = normalize(np.array([0.0, 0.0, 1.0]) if up_hint is None else up_hint, np.array([0.0, 0.0, 1.0]))
    x_axis = normalize(np.cross(up, z_axis), np.array([1.0, 0.0, 0.0]))
    y_axis = normalize(np.cross(z_axis, x_axis), np.array([0.0, 0.0, 1.0]))
    return np.column_stack([x_axis, y_axis, z_axis])
