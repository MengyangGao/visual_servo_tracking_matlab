from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .geometry import board_corner_points, project_points
from .types import CameraIntrinsics, Detection, Pose


@dataclass(slots=True)
class VideoRecorder:
    path: Path
    fps: float
    frame_size: tuple[int, int]
    enabled: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.writer = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, float(self.fps), tuple(int(v) for v in self.frame_size))

    def write(self, image_bgr: np.ndarray) -> None:
        if self.writer is not None:
            self.writer.write(image_bgr)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def bgr_canvas(width: int, height: int, color: tuple[int, int, int] = (245, 245, 245)) -> np.ndarray:
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = np.array(color, dtype=np.uint8)
    return canvas


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int] = (0, 200, 80), alpha: float = 0.35) -> np.ndarray:
    if mask is None:
        return image_bgr
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    out = image_bgr.copy()
    mask_bool = mask_arr > 0
    if not np.any(mask_bool):
        return out
    overlay = np.zeros_like(out)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    out[mask_bool] = cv2.addWeighted(out[mask_bool], 1.0 - alpha, overlay[mask_bool], alpha, 0.0)
    return out


def overlay_detection(image_bgr: np.ndarray, detection: Detection | None, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    out = image_bgr.copy()
    if detection is None:
        return out
    if detection.mask is not None:
        out = overlay_mask(out, detection.mask, color=(0, 180, 90), alpha=0.30)
    x0, y0, x1, y1 = detection.box.astype(int)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    label = f"{detection.label} {detection.score:.2f}".strip()
    cv2.putText(out, label, (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def overlay_points(image_bgr: np.ndarray, points_px: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), closed: bool = True) -> np.ndarray:
    out = image_bgr.copy()
    pts = np.asarray(points_px, dtype=np.int32).reshape(-1, 1, 2)
    if len(pts) == 0:
        return out
    if len(pts) > 1:
        cv2.polylines(out, [pts], closed, color, 2, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(out, tuple(int(v) for v in p), 4, color, -1, cv2.LINE_AA)
    return out


def mask_thumbnail(mask: np.ndarray | None, size: tuple[int, int] = (128, 96), label: str = "MASK") -> np.ndarray | None:
    if mask is None:
        return None
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("Mask thumbnail expects a 2D mask.")
    thumb_w, thumb_h = int(size[0]), int(size[1])
    thumb = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
    thumb[:] = np.array((18, 18, 24), dtype=np.uint8)
    resized = cv2.resize((mask_arr > 0).astype(np.uint8) * 255, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
    thumb[resized > 0] = np.array((0, 200, 90), dtype=np.uint8)
    cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1), (240, 240, 240), 1, cv2.LINE_AA)
    cv2.rectangle(thumb, (0, 0), (thumb_w - 1, 24), (28, 30, 40), -1)
    cv2.putText(thumb, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 1, cv2.LINE_AA)
    return thumb


def overlay_thumbnail(
    image_bgr: np.ndarray,
    thumbnail_bgr: np.ndarray | None,
    *,
    corner: str = "lower_right",
    margin: int = 12,
) -> np.ndarray:
    if thumbnail_bgr is None:
        return image_bgr
    out = image_bgr.copy()
    thumb = np.asarray(thumbnail_bgr, dtype=np.uint8)
    if thumb.ndim != 3 or thumb.shape[2] != 3:
        raise ValueError("Thumbnail must be a BGR image with shape (H, W, 3).")
    thumb_h, thumb_w = thumb.shape[:2]
    height, width = out.shape[:2]
    if corner == "lower_right":
        x0 = max(0, width - thumb_w - margin)
        y0 = max(0, height - thumb_h - margin)
    elif corner == "lower_left":
        x0 = max(0, margin)
        y0 = max(0, height - thumb_h - margin)
    elif corner == "upper_right":
        x0 = max(0, width - thumb_w - margin)
        y0 = max(0, margin)
    elif corner == "upper_left":
        x0 = max(0, margin)
        y0 = max(0, margin)
    else:
        raise ValueError(f"Unknown thumbnail corner: {corner}")
    x1 = min(width, x0 + thumb_w)
    y1 = min(height, y0 + thumb_h)
    if x1 <= x0 or y1 <= y0:
        return out
    out[y0:y1, x0:x1] = thumb[: y1 - y0, : x1 - x0]
    return out


def world_map_panel(
    target_positions: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    ee_positions: list[np.ndarray] | tuple[np.ndarray, ...] | None,
    *,
    current_target: np.ndarray | None = None,
    current_ee: np.ndarray | None = None,
    center_xy: tuple[float, float] = (0.34, 0.0),
    extent_xy: tuple[float, float] = (0.90, 0.70),
    size: tuple[int, int] = (320, 240),
    phase: str = "",
    position_error_norm: float = 0.0,
    feature_error_norm: float = 0.0,
    title: str = "TOP-DOWN MAP",
) -> np.ndarray:
    width, height = int(size[0]), int(size[1])
    panel = np.empty((height, width, 3), dtype=np.uint8)
    panel[:] = np.array((14, 18, 26), dtype=np.uint8)

    margin_x = 18
    title_h = 28
    margin_top = title_h + 10
    margin_bottom = 20
    margin_y = 16
    map_left = margin_x
    map_top = margin_top
    map_right = width - margin_x
    map_bottom = height - margin_bottom
    map_w = max(1, map_right - map_left)
    map_h = max(1, map_bottom - map_top)

    cx, cy = float(center_xy[0]), float(center_xy[1])
    ex, ey = float(extent_xy[0]), float(extent_xy[1])
    x0 = cx - 0.5 * ex
    x1 = cx + 0.5 * ex
    y0 = cy - 0.5 * ey
    y1 = cy + 0.5 * ey

    def world_to_px(point_xy: np.ndarray) -> tuple[int, int]:
        point = np.asarray(point_xy, dtype=np.float64).reshape(2)
        u = map_left + (point[0] - x0) / max(1e-9, x1 - x0) * map_w
        v = map_top + (y1 - point[1]) / max(1e-9, y1 - y0) * map_h
        return int(round(u)), int(round(v))

    # background and frame
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (220, 220, 220), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (width - 1, title_h), (30, 36, 46), -1)
    cv2.putText(panel, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2, cv2.LINE_AA)

    # workspace frame
    tl = (map_left, map_top)
    br = (map_right, map_bottom)
    cv2.rectangle(panel, tl, br, (90, 100, 112), 1, cv2.LINE_AA)

    # center axes / grid
    axis_x = world_to_px(np.array([cx, cy], dtype=np.float64))[0]
    axis_y = world_to_px(np.array([cx, cy], dtype=np.float64))[1]
    cv2.line(panel, (axis_x, map_top), (axis_x, map_bottom), (64, 88, 125), 1, cv2.LINE_AA)
    cv2.line(panel, (map_left, axis_y), (map_right, axis_y), (64, 88, 125), 1, cv2.LINE_AA)

    # axis hints
    cv2.arrowedLine(panel, (map_left + 10, map_bottom - 10), (map_left + 52, map_bottom - 10), (72, 190, 255), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(panel, (map_left + 10, map_bottom - 10), (map_left + 10, map_bottom - 52), (80, 230, 120), 2, cv2.LINE_AA, tipLength=0.18)
    cv2.putText(panel, "x", (map_left + 56, map_bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (72, 190, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "y", (map_left + 2, map_bottom - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 120), 1, cv2.LINE_AA)

    def draw_trail(points: list[np.ndarray] | tuple[np.ndarray, ...] | None, color: tuple[int, int, int]) -> None:
        if points is None:
            return
        pts = [np.asarray(point, dtype=np.float64).reshape(-1)[:2] for point in points]
        if len(pts) < 2:
            return
        for start, end in zip(pts[:-1], pts[1:]):
            cv2.line(panel, world_to_px(start), world_to_px(end), color, 2, cv2.LINE_AA)

    draw_trail(target_positions, (0, 165, 255))
    draw_trail(ee_positions, (255, 210, 80))

    current_target_xy = None if current_target is None else np.asarray(current_target, dtype=np.float64).reshape(-1)[:2]
    current_ee_xy = None if current_ee is None else np.asarray(current_ee, dtype=np.float64).reshape(-1)[:2]
    if current_target_xy is None and target_positions:
        current_target_xy = np.asarray(target_positions[-1], dtype=np.float64).reshape(-1)[:2]
    if current_ee_xy is None and ee_positions:
        current_ee_xy = np.asarray(ee_positions[-1], dtype=np.float64).reshape(-1)[:2]

    if current_target_xy is not None and current_ee_xy is not None:
        cv2.arrowedLine(panel, world_to_px(current_target_xy), world_to_px(current_ee_xy), (80, 80, 255), 3, cv2.LINE_AA, tipLength=0.18)

    if current_target_xy is not None:
        tx, ty = world_to_px(current_target_xy)
        cv2.circle(panel, (tx, ty), 6, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.circle(panel, (tx, ty), 9, (250, 250, 250), 1, cv2.LINE_AA)
        cv2.putText(panel, "T", (tx + 8, ty - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

    if current_ee_xy is not None:
        exy, eyy = world_to_px(current_ee_xy)
        cv2.circle(panel, (exy, eyy), 7, (255, 210, 80), -1, cv2.LINE_AA)
        cv2.circle(panel, (exy, eyy), 10, (250, 250, 250), 1, cv2.LINE_AA)
        cv2.putText(panel, "E", (exy + 8, eyy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 210, 80), 2, cv2.LINE_AA)

    lines = [
        f"phase: {phase}" if phase else "phase: -",
        f"pos err {position_error_norm:.3f} m",
        f"feat err {feature_error_norm:.4f}",
    ]
    y = height - 36
    for line in lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
        y += 16

    return panel


def overlay_pose_axes(image_bgr: np.ndarray, points_px: np.ndarray, color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    return overlay_points(image_bgr, points_px, color=color, closed=True)


def synthesize_board_view(
    board_texture_bgr: np.ndarray,
    board_pose: Pose,
    camera_pose: Pose,
    intrinsics: CameraIntrinsics,
    board_width_m: float,
    board_height_m: float,
    background_color: tuple[int, int, int] = (244, 244, 244),
) -> tuple[np.ndarray, np.ndarray]:
    height, width = intrinsics.height, intrinsics.width
    canvas = bgr_canvas(width, height, color=background_color)
    corners_local = board_corner_points(board_width_m, board_height_m)
    corners_world = (board_pose.rotation @ corners_local.T).T + board_pose.position[None, :]
    corners_px = project_points(corners_world, intrinsics, camera_pose)

    source = np.array(
        [
            [0.0, 0.0],
            [board_texture_bgr.shape[1] - 1.0, 0.0],
            [board_texture_bgr.shape[1] - 1.0, board_texture_bgr.shape[0] - 1.0],
            [0.0, board_texture_bgr.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    destination = corners_px.astype(np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(board_texture_bgr, homography, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpPerspective(
        np.ones(board_texture_bgr.shape[:2], dtype=np.uint8) * 255,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    canvas[mask > 0] = warped[mask > 0]
    return canvas, corners_px


class MuJoCoWorldRenderer:
    def __init__(self, model: mujoco.MjModel, height: int, width: int, camera_name: str):
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera_name = camera_name

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self.renderer.update_scene(data, camera=self.camera_name)
        image = self.renderer.render()
        if image.ndim == 3 and image.shape[-1] == 3:
            return np.asarray(image, dtype=np.uint8)
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        self.renderer.close()
