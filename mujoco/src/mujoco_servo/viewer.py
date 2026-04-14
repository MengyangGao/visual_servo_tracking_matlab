from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from .startup import maybe_reexec_under_mjpython
from .types import Pose


def running_under_mjpython() -> bool:
    return getattr(mujoco.viewer, "_MJPYTHON", None) is not None


def _as_points(points: Sequence[np.ndarray] | None) -> list[np.ndarray]:
    if points is None:
        return []
    return [np.asarray(point, dtype=np.float64).reshape(3) for point in points]


def _init_sphere_geom(geom: mujoco.MjvGeom, point: np.ndarray, rgba: np.ndarray, radius: float) -> None:
    size = np.array([radius, radius, radius], dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, point, mat, rgba)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def _init_line_geom(geom: mujoco.MjvGeom, start: np.ndarray, end: np.ndarray, rgba: np.ndarray, width: float) -> None:
    size = np.zeros(3, dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, size, start, mat, rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, width, start, end)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def _init_arrow_geom(geom: mujoco.MjvGeom, start: np.ndarray, end: np.ndarray, rgba: np.ndarray, width: float) -> None:
    size = np.zeros(3, dtype=np.float64)
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW1, size, start, mat, rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW1, width, start, end)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR


def _init_box_geom(geom: mujoco.MjvGeom, center: np.ndarray, size: np.ndarray, rgba: np.ndarray, geom_type: mujoco.mjtGeom) -> None:
    mat = np.eye(3, dtype=np.float64).reshape(-1)
    mujoco.mjv_initGeom(geom, geom_type, np.asarray(size, dtype=np.float64).reshape(3), np.asarray(center, dtype=np.float64).reshape(3), mat, rgba)
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR


@dataclass(slots=True)
class InteractiveViewer:
    handle: mujoco.viewer.Handle

    def set_camera(self, *, lookat: np.ndarray, distance: float, azimuth: float, elevation: float) -> None:
        with self.handle.lock():
            cam = self.handle.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.fixedcamid = -1
            cam.trackbodyid = -1
            cam.lookat[:] = np.asarray(lookat, dtype=np.float64).reshape(3)
            cam.distance = float(distance)
            cam.azimuth = float(azimuth)
            cam.elevation = float(elevation)
        self.handle.sync()

    def set_status(self, lines: Sequence[str]) -> None:
        if not lines:
            self.handle.clear_texts()
            return
        text = "\n".join(str(line) for line in lines if str(line))
        self.handle.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))

    def _fit_panel(self, image_bgr: np.ndarray, max_panel_w: int, max_panel_h: int) -> tuple[int, int, np.ndarray]:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Viewer image panel expects a BGR image with shape (H, W, 3).")
        image_h, image_w = image.shape[:2]
        if image_w <= 0 or image_h <= 0:
            raise ValueError("Viewer image panel expects a non-empty BGR image.")
        panel_w = max(1, min(int(max_panel_w), int(round(max_panel_h * image_w / image_h)) if image_h > 0 else int(max_panel_w)))
        panel_h = max(1, int(round(panel_w * image_h / image_w)))
        if panel_h > max_panel_h:
            panel_h = max(1, int(max_panel_h))
            panel_w = max(1, int(round(panel_h * image_w / image_h)))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_AREA if (panel_w < image_w or panel_h < image_h) else cv2.INTER_LINEAR)
        return panel_w, panel_h, np.ascontiguousarray(resized)

    def set_image_panels(self, images_bgr: Sequence[np.ndarray | None], *, margin: int = 16, gap: int = 12) -> None:
        valid_images = [image for image in images_bgr if image is not None]
        if not valid_images:
            self.handle.clear_images()
            return
        viewport = self.handle.viewport
        if viewport is None:
            return

        panels: list[tuple[mujoco.MjrRect, np.ndarray]] = []
        max_panel_w = max(1, min(320, viewport.width // 4, viewport.width - 2 * margin))
        if len(valid_images) == 1:
            panel_w, panel_h, resized = self._fit_panel(valid_images[0], max_panel_w, max(1, min(180, viewport.height // 4, viewport.height - 2 * margin)))
            left = max(0, viewport.width - panel_w - margin)
            bottom = max(0, margin)
            panels.append((mujoco.MjrRect(left, bottom, panel_w, panel_h), resized))
        else:
            total_height = max(1, viewport.height - 2 * margin - gap * (len(valid_images) - 1))
            per_panel_h = max(1, total_height // len(valid_images))
            current_top = viewport.height - margin
            for image_bgr in valid_images:
                panel_w, panel_h, resized = self._fit_panel(image_bgr, max_panel_w, per_panel_h)
                left = max(0, viewport.width - panel_w - margin)
                bottom = max(0, current_top - panel_h)
                panels.append((mujoco.MjrRect(left, bottom, panel_w, panel_h), resized))
                current_top = bottom - gap

        self.handle.set_images(panels if len(panels) > 1 else panels[0])

    def set_image_panel(self, image_bgr: np.ndarray | None, *, margin: int = 16) -> None:
        self.set_image_panels([image_bgr], margin=margin)

    def set_motion_traces(
        self,
        target_positions: Sequence[np.ndarray] | None = None,
        ee_positions: Sequence[np.ndarray] | None = None,
        *,
        trail_length: int = 12,
        current_target: np.ndarray | None = None,
        current_ee: np.ndarray | None = None,
        target_pose: Pose | None = None,
        ee_pose: Pose | None = None,
        workspace_center: np.ndarray | None = None,
        workspace_size: np.ndarray | None = None,
    ) -> None:
        user_scn = self.handle.user_scn
        if user_scn is None:
            return

        target = _as_points(target_positions)[-max(0, int(trail_length)) :]
        ee = _as_points(ee_positions)[-max(0, int(trail_length)) :]
        if not target and not ee:
            with self.handle.lock():
                user_scn.ngeom = 0
            return

        target_color = np.array([1.0, 0.62, 0.12, 0.95], dtype=np.float32)
        ee_color = np.array([0.20, 0.86, 1.00, 0.95], dtype=np.float32)
        target_head_color = np.array([1.0, 0.28, 0.12, 1.0], dtype=np.float32)
        ee_head_color = np.array([0.18, 0.48, 1.0, 1.0], dtype=np.float32)
        connector_color = np.array([1.0, 0.25, 0.20, 0.96], dtype=np.float32)
        target_axis_colors = (
            np.array([1.0, 0.20, 0.20, 0.95], dtype=np.float32),
            np.array([0.20, 1.0, 0.20, 0.95], dtype=np.float32),
            np.array([0.20, 0.35, 1.0, 0.95], dtype=np.float32),
        )
        ee_axis_colors = (
            np.array([1.0, 0.52, 0.10, 0.85], dtype=np.float32),
            np.array([0.20, 0.95, 0.85, 0.85], dtype=np.float32),
            np.array([0.75, 0.35, 1.0, 0.85], dtype=np.float32),
        )

        with self.handle.lock():
            maxgeom = int(user_scn.maxgeom)
            geom_idx = 0

            def _reserve(count: int) -> bool:
                return geom_idx + count <= maxgeom

            def _write_line(start: np.ndarray, end: np.ndarray, rgba: np.ndarray, width: float) -> None:
                nonlocal geom_idx
                if geom_idx >= maxgeom:
                    return
                geom = user_scn.geoms[geom_idx]
                _init_line_geom(geom, start, end, rgba, width)
                user_scn.geomorder[geom_idx] = geom_idx
                geom_idx += 1

            def _write_sphere(point: np.ndarray, rgba: np.ndarray, radius: float) -> None:
                nonlocal geom_idx
                if geom_idx >= maxgeom:
                    return
                geom = user_scn.geoms[geom_idx]
                _init_sphere_geom(geom, point, rgba, radius)
                user_scn.geomorder[geom_idx] = geom_idx
                geom_idx += 1

            def _write_arrow(start: np.ndarray, end: np.ndarray, rgba: np.ndarray, width: float) -> None:
                nonlocal geom_idx
                if geom_idx >= maxgeom:
                    return
                geom = user_scn.geoms[geom_idx]
                _init_arrow_geom(geom, start, end, rgba, width)
                user_scn.geomorder[geom_idx] = geom_idx
                geom_idx += 1

            def _write_box(center: np.ndarray, size: np.ndarray, rgba: np.ndarray, geom_type: mujoco.mjtGeom) -> None:
                nonlocal geom_idx
                if geom_idx >= maxgeom:
                    return
                geom = user_scn.geoms[geom_idx]
                _init_box_geom(geom, center, size, rgba, geom_type)
                user_scn.geomorder[geom_idx] = geom_idx
                geom_idx += 1

            user_scn.ngeom = 0

            if workspace_center is not None and workspace_size is not None:
                center = np.asarray(workspace_center, dtype=np.float64).reshape(3)
                size = np.asarray(workspace_size, dtype=np.float64).reshape(3)
                size = np.array([max(0.01, float(size[0]) * 0.5), max(0.01, float(size[1]) * 0.5), max(0.002, float(size[2]) * 0.5)], dtype=np.float64)
                plane_rgba = np.array([0.12, 0.24, 0.45, 0.18], dtype=np.float32)
                frame_rgba = np.array([0.30, 0.60, 0.95, 0.55], dtype=np.float32)
                _write_box(center, size, plane_rgba, mujoco.mjtGeom.mjGEOM_BOX)
                _write_box(center, size, frame_rgba, mujoco.mjtGeom.mjGEOM_LINEBOX)

            def _write_pose_axes(pose: Pose, axis_scale: float, colors: tuple[np.ndarray, np.ndarray, np.ndarray], axis_width: float) -> None:
                origin = np.asarray(pose.position, dtype=np.float64).reshape(3)
                rot = np.asarray(pose.rotation, dtype=np.float64).reshape(3, 3)
                for axis_idx, rgba in enumerate(colors):
                    direction = rot[:, axis_idx]
                    _write_arrow(origin, origin + direction * float(axis_scale), rgba, axis_width)

            for trail, rgba, head_rgba in ((target, target_color, target_head_color), (ee, ee_color, ee_head_color)):
                if len(trail) >= 2:
                    for start, end in zip(trail[:-1], trail[1:]):
                        if not _reserve(1):
                            break
                        _write_line(start, end, rgba, 2.5)
                if trail:
                    if _reserve(1):
                        _write_sphere(trail[-1], head_rgba, 0.019)

            if current_target is not None and current_ee is not None and _reserve(1):
                _write_line(np.asarray(current_target, dtype=np.float64), np.asarray(current_ee, dtype=np.float64), connector_color, 4.0)

            if target_pose is not None:
                _write_pose_axes(target_pose, 0.045, target_axis_colors, 0.004)
                if _reserve(1):
                    _write_sphere(np.asarray(target_pose.position, dtype=np.float64), target_head_color, 0.022)
            elif current_target is not None and _reserve(1):
                _write_sphere(np.asarray(current_target, dtype=np.float64), target_head_color, 0.022)

            if ee_pose is not None:
                _write_pose_axes(ee_pose, 0.040, ee_axis_colors, 0.0035)
                if _reserve(1):
                    _write_sphere(np.asarray(ee_pose.position, dtype=np.float64), ee_head_color, 0.020)
            elif current_ee is not None and _reserve(1):
                _write_sphere(np.asarray(current_ee, dtype=np.float64), ee_head_color, 0.020)

            user_scn.ngeom = geom_idx

    def sync(self) -> None:
        self.handle.sync()

    def is_running(self) -> bool:
        return self.handle.is_running()

    def close(self) -> None:
        self.handle.close()


def launch_interactive_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    lookat: Sequence[float] = (0.34, 0.0, 0.30),
    distance: float = 2.0,
    azimuth: float = 140.0,
    elevation: float = -25.0,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> InteractiveViewer:
    handle = mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
    )
    viewer = InteractiveViewer(handle=handle)
    viewer.set_camera(lookat=np.asarray(lookat, dtype=np.float64), distance=distance, azimuth=azimuth, elevation=elevation)
    return viewer


def initial_viewer_pose(task: str, live_mode: bool = False) -> tuple[np.ndarray, float, float, float]:
    if live_mode:
        return np.array([0.34, 0.0, 0.42], dtype=np.float64), 2.3, 96.0, -15.0
    if task == "t2-eye":
        return np.array([0.34, 0.0, 0.42], dtype=np.float64), 1.4, 98.0, -13.0
    if task == "t3-ibvs":
        return np.array([0.34, 0.0, 0.42], dtype=np.float64), 2.4, 96.0, -15.0
    return np.array([0.34, 0.0, 0.42], dtype=np.float64), 2.7, 96.0, -15.0
