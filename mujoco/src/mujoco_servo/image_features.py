from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import BoardConfig
from .geometry import board_corner_points, board_pose_from_center, project_points
from .types import CameraIntrinsics, Pose


def make_charuco_board(board_cfg: BoardConfig) -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board_cfg.dictionary_name))
    return cv2.aruco.CharucoBoard(board_cfg.pattern_dims, board_cfg.square_length_m, board_cfg.marker_length_m, dictionary)


def render_charuco_board_image(board_cfg: BoardConfig, output_path: Path) -> np.ndarray:
    board = make_charuco_board(board_cfg)
    size = tuple(int(v) for v in board_cfg.image_size_px)
    image = board.generateImage(size, marginSize=board_cfg.margin_px, borderBits=board_cfg.border_bits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return image


def board_outer_corners(board_cfg: BoardConfig) -> np.ndarray:
    return board_corner_points(board_cfg.width_m, board_cfg.height_m)


def board_pose_at(center: np.ndarray, normal: np.ndarray = np.array([0.0, 0.0, 1.0]), up_hint: np.ndarray = np.array([0.0, 1.0, 0.0])) -> Pose:
    return board_pose_from_center(center, normal, up_hint)


def project_board_corners(board_cfg: BoardConfig, intrinsics: CameraIntrinsics, camera_pose: Pose, board_pose: Pose) -> np.ndarray:
    corners_local = board_outer_corners(board_cfg)
    corners_world = (board_pose.rotation @ corners_local.T).T + board_pose.position[None, :]
    return project_points(corners_world, intrinsics, camera_pose)


def box_xyxy_to_corners(box_xyxy: np.ndarray) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float64).reshape(4)
    x0, y0, x1, y1 = box
    return np.array(
        [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ],
        dtype=np.float64,
    )


def centered_box_corners(image_shape: tuple[int, int], box_size_px: float, aspect_ratio: float = 1.0) -> np.ndarray:
    height, width = int(image_shape[0]), int(image_shape[1])
    half_w = 0.5 * float(box_size_px) * float(aspect_ratio)
    half_h = 0.5 * float(box_size_px)
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    return np.array(
        [
            [cx - half_w, cy - half_h],
            [cx + half_w, cy - half_h],
            [cx + half_w, cy + half_h],
            [cx - half_w, cy + half_h],
        ],
        dtype=np.float64,
    )


def bbox_from_mask(mask: np.ndarray) -> np.ndarray | None:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max())
    y1 = float(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def mask_area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(np.asarray(mask) > 0))


def draw_feature_overlay(image: np.ndarray, points_px: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), label: str | None = None) -> np.ndarray:
    out = image.copy()
    pts = np.asarray(points_px, dtype=np.int32).reshape(-1, 1, 2)
    if pts.shape[0] >= 2:
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(out, tuple(int(v) for v in p), 4, color, -1, cv2.LINE_AA)
    if label and len(pts) > 0:
        cv2.putText(out, label, tuple(pts.reshape(-1, 2)[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def charuco_detect(image_bgr: np.ndarray, board_cfg: BoardConfig) -> tuple[np.ndarray, np.ndarray]:
    board = make_charuco_board(board_cfg)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, *_ = detector.detectBoard(image_bgr)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)
    return np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2), np.asarray(charuco_ids, dtype=np.int32).reshape(-1)


def charuco_pose_from_image(
    image_bgr: np.ndarray,
    board_cfg: BoardConfig,
    intrinsics: CameraIntrinsics,
) -> tuple[Pose | None, np.ndarray, np.ndarray]:
    board = make_charuco_board(board_cfg)
    corners, ids = charuco_detect(image_bgr, board_cfg)
    if len(ids) < 4:
        return None, corners, ids

    obj_points, img_points = board.matchImagePoints(corners, ids)
    obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
    if len(obj_points) < 4:
        return None, corners, ids

    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        np.array([[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        intrinsics.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, corners, ids
    rot, _ = cv2.Rodrigues(rvec)
    return Pose(tvec.reshape(3), rot), corners, ids


def calibrate_from_charuco_images(
    images_bgr: Iterable[np.ndarray],
    board_cfg: BoardConfig,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    board = make_charuco_board(board_cfg)
    all_obj: list[np.ndarray] = []
    all_img: list[np.ndarray] = []
    detector = cv2.aruco.CharucoDetector(board)
    for image in images_bgr:
        corners, ids, *_ = detector.detectBoard(image)
        if corners is None or ids is None or len(ids) < 4:
            continue
        obj_points, img_points = board.matchImagePoints(corners, ids)
        obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
        img_points = np.asarray(img_points, dtype=np.float32).reshape(-1, 2)
        if len(obj_points) >= 4:
            all_obj.append(obj_points)
            all_img.append(img_points)

    if not all_obj:
        raise RuntimeError("No ChArUco observations collected.")

    camera_matrix = np.array([[900.0, 0.0, image_size[1] / 2.0], [0.0, 900.0, image_size[0] / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.zeros(5, dtype=np.float64)
    rms, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        all_obj,
        all_img,
        (image_size[1], image_size[0]),
        camera_matrix,
        distortion,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    return camera_matrix, distortion.reshape(-1), float(rms), len(all_obj)
