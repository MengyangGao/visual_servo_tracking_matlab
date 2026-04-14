from __future__ import annotations

from collections import deque
import warnings
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .camera import VideoCaptureStream, maybe_resize
from .config import AppConfig, build_default_config
from .control import ImageBasedServo, PoseBasedServo, board_feature_error, camera_twist_to_joint_velocity
from .geometry import board_pose_from_center, compose, inverse_pose, look_at, normalize
from .image_features import bbox_from_mask, box_xyxy_to_corners, calibrate_from_charuco_images, centered_box_corners, mask_area, mask_centroid
from .perception import CharucoTracker, GroundingDinoDetector, OpenVocabularyTracker, Sam2MaskGenerator
from .rendering import MuJoCoWorldRenderer, VideoRecorder, mask_thumbnail, overlay_detection, overlay_points, overlay_thumbnail, synthesize_board_view, world_map_panel
from .robot import PandaRobot
from .scene import build_scene
from .viewer import initial_viewer_pose, launch_interactive_viewer, running_under_mjpython
from .types import Pose, RunSummary, TrackingSample


def _desired_camera_pose_for_target(board_pose: Pose, camera_distance_m: float) -> Pose:
    from .geometry import look_at

    camera_pos = board_pose.position + board_pose.rotation[:, 2] * float(camera_distance_m)
    return look_at(camera_pos, board_pose.position, np.array([0.0, 0.0, 1.0], dtype=np.float64))


def _desired_hand_pose_for_target(board_pose: Pose, camera_distance_m: float, camera_offset: Pose) -> Pose:
    return compose(_desired_camera_pose_for_target(board_pose, camera_distance_m), inverse_pose(camera_offset))


def _pixel_to_camera_point(pixel_xy: np.ndarray, depth_m: float, intrinsics) -> np.ndarray:
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    x = (pixel_xy[0] - intrinsics.cx) / intrinsics.fx * float(depth_m)
    y = (pixel_xy[1] - intrinsics.cy) / intrinsics.fy * float(depth_m)
    return np.array([x, y, float(depth_m)], dtype=np.float64)


def _ray_plane_intersection_world(pixel_xy: np.ndarray, camera_pose: Pose, intrinsics, plane_z: float) -> np.ndarray:
    pixel_xy = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    x = (pixel_xy[0] - intrinsics.cx) / intrinsics.fx
    y = (pixel_xy[1] - intrinsics.cy) / intrinsics.fy
    ray_cam = normalize(np.array([x, y, 1.0], dtype=np.float64))
    ray_world = camera_pose.rotation @ ray_cam
    denom = float(ray_world[2])
    if abs(denom) < 1e-9:
        raise ValueError("Camera ray is parallel to the tracking plane.")
    t = (float(plane_z) - float(camera_pose.position[2])) / denom
    if t <= 0.0:
        raise ValueError("Tracking plane lies behind the camera ray.")
    return camera_pose.position + t * ray_world


def _lowpass_vector(previous: np.ndarray | None, current: np.ndarray, alpha: float, max_step: float | None = None) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    if previous is None:
        return current.copy()
    previous = np.asarray(previous, dtype=np.float64).reshape(-1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = previous + alpha * (current - previous)
    if max_step is None:
        return blended
    max_step = float(max_step)
    delta = np.clip(blended - previous, -max_step, max_step)
    return previous + delta


_PREVIEW_BORDER_X = 24
_PREVIEW_BORDER_Y = 30


def _detection_box_and_center(det) -> tuple[np.ndarray, np.ndarray, float]:
    box = np.asarray(det.box, dtype=np.float64).reshape(4)
    if det.mask is not None:
        mask_box = bbox_from_mask(det.mask)
        if mask_box is not None:
            box = mask_box
        centroid = mask_centroid(det.mask)
        if centroid is not None:
            center = centroid
        else:
            center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
        area = max(1.0, float(mask_area(det.mask)))
        return box, center, area
    center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5], dtype=np.float64)
    area = max(1.0, float((box[2] - box[0]) * (box[3] - box[1])))
    return box, center, area


def _generate_target_path(cfg: AppConfig, motion_time_s: float, moving: bool) -> Pose:
    base = cfg.target_center_m.copy()
    if moving:
        amp_x, amp_y = cfg.sim.target_motion_radius_m
        center = base + np.array(
            [
                amp_x * np.sin(cfg.sim.target_motion_speed * motion_time_s),
                amp_y * np.sin(cfg.sim.target_motion_speed * 0.8 * motion_time_s),
                0.0,
            ],
            dtype=np.float64,
        )
    else:
        center = base
    return board_pose_from_center(center, np.array([1.0, 0.0, 0.0], dtype=np.float64))


def _arm_joint_targets_from_twist(robot: PandaRobot, twist_world: np.ndarray, damping: float, step_dt: float) -> np.ndarray:
    jac = robot.body_jacobian(robot.ee_body_name)[:, : len(robot.arm_joint_names)]
    dq = camera_twist_to_joint_velocity(jac, twist_world, damping=damping)
    return robot.current_arm_qpos() + dq * float(step_dt)


def _save_summary_csv(samples: list[TrackingSample], output_dir: Path) -> None:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "phase", "position_error_norm", "feature_error_norm", "pixel_error_norm"])
        for sample in samples:
            writer.writerow([sample.time_s, sample.phase, sample.position_error_norm, sample.feature_error_norm, sample.pixel_error_norm])


def _plot_metrics(samples: list[TrackingSample], output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not samples:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    times = np.array([s.time_s for s in samples], dtype=np.float64)
    pos_err = np.array([s.position_error_norm for s in samples], dtype=np.float64)
    feat_err = np.array([s.feature_error_norm for s in samples], dtype=np.float64)
    pix_err = np.array([s.pixel_error_norm for s in samples], dtype=np.float64)

    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    ax[0].plot(times, pos_err, color="#1f77b4")
    ax[0].set_ylabel("pos [m]")
    ax[1].plot(times, feat_err, color="#ff7f0e")
    ax[1].set_ylabel("feature")
    ax[2].plot(times, pix_err, color="#2ca02c")
    ax[2].set_ylabel("pixel")
    ax[2].set_xlabel("time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / f"{title}_metrics.png", dpi=160)
    plt.close(fig)


def _mocap_board_color(board_image: np.ndarray) -> np.ndarray:
    if board_image.ndim == 2:
        return cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
    return board_image


def _viewer_status_lines(mode: str, phase: str, prompt: str | None, detection_label: str | None, position_error_norm: float, feature_error_norm: float, joint_positions: np.ndarray | None = None) -> list[str]:
    lines = [
        f"mode: {mode}",
        f"phase: {phase}",
        "drag mouse to rotate/zoom/pan",
    ]
    if prompt:
        lines.append(f"prompt: {prompt}")
    if detection_label:
        lines.append(f"target: {detection_label}")
    lines.append(f"position error: {position_error_norm:.3f} m")
    lines.append(f"feature error: {feature_error_norm:.4f}")
    if joint_positions is not None and len(joint_positions):
        joints = ", ".join(f"{float(v):+.2f}" for v in np.asarray(joint_positions, dtype=np.float64)[:3])
        lines.append(f"q[:3]: {joints}")
    return lines


def _workspace_map_extent(radius_xy: tuple[float, float]) -> tuple[float, float]:
    radius_x = float(abs(radius_xy[0]))
    radius_y = float(abs(radius_xy[1]))
    span_x = max(0.90, 2.2 * radius_x + 0.25)
    span_y = max(0.70, 2.2 * radius_y + 0.25)
    return span_x, span_y


def _preview_panel(image_bgr: np.ndarray, title: str, subtitle: str, info_lines: list[str]) -> np.ndarray:
    panel = np.asarray(image_bgr, dtype=np.uint8).copy()
    panel = cv2.copyMakeBorder(panel, 18, 12, 12, 12, cv2.BORDER_CONSTANT, value=(26, 26, 30))
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (235, 235, 235), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 38), (34, 38, 48), -1)
    cv2.putText(panel, title, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    y = 74
    for line in info_lines:
        cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
        y += 20
    return panel


def _launch_viewer_if_available(model: mujoco.MjModel, data: mujoco.MjData, *, task: str, live_mode: bool) -> object | None:
    try:
        lookat, distance, azimuth, elevation = initial_viewer_pose(task, live_mode=live_mode)
        return launch_interactive_viewer(
            model,
            data,
            lookat=lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            show_left_ui=True,
            show_right_ui=True,
        )
    except Exception as exc:
        warnings.warn(f"Interactive MuJoCo viewer unavailable, falling back to offscreen preview: {exc}", RuntimeWarning)
        return None


def run_simulation(cfg: AppConfig | None = None, task: str = "t2-fixed", display: bool | None = None) -> RunSummary:
    cfg = build_default_config() if cfg is None else cfg
    if display is not None:
        cfg.sim.display = bool(display)
    cfg.paths.ensure()

    scene = build_scene(cfg)
    robot = PandaRobot(scene.model, scene.data)
    pose_servo = PoseBasedServo(position_gain=cfg.sim.position_gain, orientation_gain=cfg.sim.orientation_gain, damping=cfg.sim.ik_damping)
    image_servo = ImageBasedServo(gain=cfg.sim.ibvs_gain, damping=cfg.sim.ibvs_damping)
    viewer_session = _launch_viewer_if_available(scene.model, scene.data, task=task, live_mode=False) if cfg.sim.display else None
    show_cv2_windows = cfg.sim.display and not running_under_mjpython()
    world_renderer = None
    if cfg.sim.render_world and (viewer_session is None or cfg.sim.record_video):
        try:
            world_renderer = MuJoCoWorldRenderer(scene.model, cfg.sim.world_camera_height, cfg.sim.world_camera_width, cfg.sim.world_camera_name)
        except Exception:
            world_renderer = None

    recorder = None
    if cfg.sim.record_video:
        if world_renderer is not None:
            frame_size = (cfg.sim.world_camera_width, cfg.sim.world_camera_height)
        else:
            frame_size = (cfg.camera.width + _PREVIEW_BORDER_X, cfg.camera.height + _PREVIEW_BORDER_Y)
        recorder = VideoRecorder(cfg.paths.results_dir / f"{task}.mp4", fps=1.0 / cfg.sim.dt, frame_size=frame_size)

    static_target_pose = _generate_target_path(cfg, 0.0, moving=False)
    target_motion_enabled = False
    target_motion_start_time_s = 0.0
    aligned_stable_frames = 0
    target_pose = static_target_pose
    robot.set_target_pose(target_pose)
    mujoco.mj_forward(scene.model, scene.data)

    samples: list[TrackingSample] = []
    board_bgr = _mocap_board_color(scene.board_image)
    smoothed_q_target: np.ndarray | None = None
    target_trail: deque[np.ndarray] = deque(maxlen=24)
    ee_trail: deque[np.ndarray] = deque(maxlen=24)
    workspace_center_xy = (float(cfg.target_center_m[0]), float(cfg.target_center_m[1]))
    workspace_extent_xy = _workspace_map_extent(cfg.sim.target_motion_radius_m)
    workspace_plane_center = np.array([cfg.target_center_m[0], cfg.target_center_m[1], cfg.sim.target_height_m], dtype=np.float64)
    workspace_plane_size = np.array([workspace_extent_xy[0], workspace_extent_xy[1], 0.004], dtype=np.float64)

    for step_idx in range(cfg.sim.steps):
        sim_time_s = step_idx * cfg.sim.dt
        if target_motion_enabled:
            motion_time_s = max(0.0, sim_time_s - target_motion_start_time_s)
            target_pose = _generate_target_path(cfg, motion_time_s, moving=True)
        else:
            target_pose = static_target_pose
        robot.set_target_pose(target_pose)
        mujoco.mj_forward(scene.model, scene.data)

        ee_pose = robot.current_ee_pose()
        desired_camera_pose = _desired_camera_pose_for_target(target_pose, cfg.sim.target_offset_m)
        desired_hand_pose = _desired_hand_pose_for_target(target_pose, cfg.sim.target_offset_m, cfg.eye_camera_offset)
        current_camera_pose = ee_pose

        current_view, current_points_px = synthesize_board_view(
            board_texture_bgr=board_bgr,
            board_pose=target_pose,
            camera_pose=current_camera_pose,
            intrinsics=cfg.camera,
            board_width_m=cfg.board.width_m,
            board_height_m=cfg.board.height_m,
        )
        _desired_view, desired_points_px = synthesize_board_view(
            board_texture_bgr=board_bgr,
            board_pose=target_pose,
            camera_pose=desired_camera_pose,
            intrinsics=cfg.camera,
            board_width_m=cfg.board.width_m,
            board_height_m=cfg.board.height_m,
        )

        if task in {"t2-fixed", "t2-eye"}:
            twist_world = pose_servo.camera_twist(ee_pose, desired_hand_pose)
            q_target = _arm_joint_targets_from_twist(robot, twist_world, cfg.sim.ik_damping, cfg.sim.dt)
        elif task == "t3-ibvs":
            depth_est = max(0.25, float(np.linalg.norm(desired_camera_pose.position - target_pose.position)))
            camera_twist = image_servo.camera_twist(current_points_px, desired_points_px, cfg.camera, depth=depth_est)
            twist_world = np.concatenate(
                [
                    current_camera_pose.rotation @ camera_twist[:3],
                    current_camera_pose.rotation @ camera_twist[3:],
                ]
            )
            q_target = _arm_joint_targets_from_twist(robot, twist_world, cfg.sim.ik_damping, cfg.sim.dt)
        else:
            raise ValueError(f"Unknown simulation task '{task}'.")

        q_target = _lowpass_vector(
            smoothed_q_target,
            q_target,
            cfg.sim.joint_target_smoothing_alpha,
            cfg.sim.joint_target_max_step_rad,
        )
        smoothed_q_target = q_target
        robot.apply_joint_targets(q_target, gripper_ctrl=cfg.sim.gripper_ctrl)
        mujoco.mj_step(scene.model, scene.data)

        ee_pose = robot.current_ee_pose()
        position_error_norm = float(np.linalg.norm(ee_pose.position - desired_hand_pose.position))
        pixel_error_norm = float(np.linalg.norm(current_points_px - desired_points_px))
        feature_error_norm = board_feature_error(current_points_px, desired_points_px, cfg.camera)
        target_trail.append(target_pose.position.copy())
        ee_trail.append(ee_pose.position.copy())
        if not target_motion_enabled:
            aligned = position_error_norm < cfg.sim.align_position_threshold_m and feature_error_norm < cfg.sim.align_feature_threshold
            if aligned:
                aligned_stable_frames += 1
            else:
                aligned_stable_frames = 0
            if aligned_stable_frames >= cfg.sim.align_hold_frames or step_idx >= cfg.sim.align_timeout_frames:
                target_motion_enabled = True
                target_motion_start_time_s = sim_time_s + cfg.sim.dt
                aligned_stable_frames = 0
        phase = "tracking" if target_motion_enabled else "aligning"
        samples.append(
            TrackingSample(
                time_s=step_idx * cfg.sim.dt,
                target_pose=target_pose,
                camera_pose=current_camera_pose,
                ee_pose=ee_pose,
                joint_positions=robot.current_arm_qpos(),
                joint_targets=q_target,
                phase=phase,
                pixel_error_norm=pixel_error_norm,
                position_error_norm=position_error_norm,
                feature_error_norm=feature_error_norm,
            )
        )

        camera_preview = overlay_points(current_view.copy(), desired_points_px, color=(0, 165, 255))
        camera_preview = overlay_points(camera_preview, current_points_px, color=(0, 255, 0))
        camera_preview = _preview_panel(
            camera_preview,
            "SIM SENSOR VIEW",
            f"{task} / {samples[-1].time_s:.2f}s",
            [
                f"phase: {phase}",
                f"pos err {position_error_norm:.3f} m",
                f"feat err {feature_error_norm:.4f}",
                "green = detected corners",
                "orange = goal box",
            ],
        )
        map_panel = world_map_panel(
            list(target_trail),
            list(ee_trail),
            current_target=target_pose.position,
            current_ee=ee_pose.position,
            center_xy=workspace_center_xy,
            extent_xy=workspace_extent_xy,
            phase=phase,
            position_error_norm=position_error_norm,
            feature_error_norm=feature_error_norm,
        )

        if viewer_session is not None:
            viewer_session.set_motion_traces(
                list(target_trail),
                list(ee_trail),
                current_target=target_pose.position,
                current_ee=ee_pose.position,
                target_pose=target_pose,
                ee_pose=ee_pose,
                workspace_center=workspace_plane_center,
                workspace_size=workspace_plane_size,
            )
            viewer_session.set_status(_viewer_status_lines(f"simulation/{task}", phase, None, "servo target", position_error_norm, feature_error_norm, robot.current_arm_qpos()))
            viewer_session.set_image_panels([camera_preview, map_panel])
            viewer_session.sync()
            if not viewer_session.is_running():
                break

        if cfg.sim.display or recorder is not None:
            world_image = None
            if world_renderer is not None:
                world_image = world_renderer.render(scene.data)
                if world_image.ndim == 3 and world_image.shape[-1] == 3:
                    world_image = cv2.cvtColor(world_image, cv2.COLOR_RGB2BGR)
                cv2.putText(world_image, f"{task} t={samples[-1].time_s:.2f}s pos={position_error_norm:.3f}m", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
            if recorder is not None:
                if world_image is not None:
                    recorder.write(world_image)
                else:
                    recorder.write(camera_preview)
            if show_cv2_windows:
                if viewer_session is None and world_image is not None:
                    cv2.imshow("world", world_image)
                cv2.imshow("camera", camera_preview)
                cv2.imshow("map", map_panel)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    if recorder is not None:
        recorder.close()
    if viewer_session is not None:
        viewer_session.close()
    if world_renderer is not None:
        world_renderer.close()
    if show_cv2_windows:
        cv2.destroyAllWindows()

    result_dir = cfg.paths.results_dir / task
    plot_dir = cfg.paths.plots_dir / task
    _save_summary_csv(samples, result_dir)
    _plot_metrics(samples, plot_dir, task)
    return RunSummary(name=task, output_dir=result_dir, samples=samples)


def run_live_camera(
    cfg: AppConfig | None = None,
    prompt: str | None = None,
    source: str | None = None,
    display: bool | None = None,
    max_frames: int | None = None,
    capture: VideoCaptureStream | None = None,
    tracker: OpenVocabularyTracker | None = None,
    detector: GroundingDinoDetector | None = None,
    segmenter: Sam2MaskGenerator | None = None,
) -> RunSummary:
    cfg = build_default_config() if cfg is None else cfg
    if prompt is not None:
        cfg.live.prompt = prompt
    if source is not None:
        cfg.live.source = source
    if display is not None:
        cfg.live.display = bool(display)
    cfg.paths.ensure()

    scene = build_scene(cfg)
    robot = PandaRobot(scene.model, scene.data)
    pose_servo = PoseBasedServo(position_gain=cfg.sim.position_gain, orientation_gain=cfg.sim.orientation_gain, damping=cfg.sim.ik_damping)
    viewer_session = _launch_viewer_if_available(scene.model, scene.data, task="live-camera", live_mode=True) if cfg.live.display else None
    show_cv2_windows = cfg.live.display and not running_under_mjpython()
    world_renderer = None
    if cfg.live.render_world and viewer_session is None:
        try:
            world_renderer = MuJoCoWorldRenderer(scene.model, cfg.sim.world_camera_height, cfg.sim.world_camera_width, cfg.sim.world_camera_name)
        except Exception:
            world_renderer = None

    if capture is None:
        if cfg.live.source == "camera":
            capture_source: int | str = cfg.live.device_index
        elif cfg.live.source == "video":
            if not cfg.live.video_path:
                raise ValueError("video_path is required when source='video'.")
            capture_source = cfg.live.video_path
        else:
            capture_source = cfg.live.source
        capture = VideoCaptureStream(capture_source, backend=cfg.live.camera_backend, width=cfg.live.camera_width, height=cfg.live.camera_height)

    if tracker is None:
        if detector is None:
            detector = GroundingDinoDetector(model_id=cfg.live.model_id, box_threshold=cfg.live.box_threshold, text_threshold=cfg.live.text_threshold, device=cfg.live.device)
        if segmenter is None and cfg.live.use_sam2:
            try:
                segmenter = Sam2MaskGenerator(model_id=cfg.live.sam_model_id, device=cfg.live.device)
            except Exception as exc:
                warnings.warn(f"SAM2 unavailable, continuing with Grounding DINO only: {exc}", RuntimeWarning)
                segmenter = None
        tracker = OpenVocabularyTracker(detector=detector, segmenter=segmenter, max_side=cfg.live.inference_max_side)
    recorder = None
    if cfg.live.record_video:
        recorder = VideoRecorder(
            cfg.paths.results_dir / "live_camera.mp4",
            fps=30.0,
            frame_size=(cfg.live.camera_width + _PREVIEW_BORDER_X, cfg.live.camera_height + _PREVIEW_BORDER_Y),
        )

    samples: list[TrackingSample] = []
    frame_limit = cfg.live.max_frames if max_frames is None else int(max_frames)
    idx = 0
    smoothed_object_world: np.ndarray | None = None
    smoothed_q_target: np.ndarray | None = None
    target_trail: deque[np.ndarray] = deque(maxlen=24)
    ee_trail: deque[np.ndarray] = deque(maxlen=24)
    detection_streak = 0
    aligned_stable_frames = 0
    current_target_world: np.ndarray | None = None
    workspace_center_xy = (float(cfg.target_center_m[0]), float(cfg.target_center_m[1]))
    workspace_extent_xy = _workspace_map_extent(cfg.sim.target_motion_radius_m)
    workspace_plane_center = np.array([cfg.target_center_m[0], cfg.target_center_m[1], cfg.sim.target_height_m], dtype=np.float64)
    workspace_plane_size = np.array([workspace_extent_xy[0], workspace_extent_xy[1], 0.004], dtype=np.float64)
    while True:
        if frame_limit > 0 and idx >= frame_limit:
            break
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame = maybe_resize(frame, width=cfg.live.camera_width, height=cfg.live.camera_height)

        det = tracker.observe(frame, cfg.live.prompt)
        current_hand_pose = robot.current_ee_pose()
        desired_hand_pose = current_hand_pose.copy()
        current_camera_pose = compose(current_hand_pose, cfg.eye_camera_offset)
        current_points_px = np.empty((0, 2), dtype=np.float64)
        desired_points_px = np.empty((0, 2), dtype=np.float64)
        feature_error_norm = 0.0
        q_target = robot.current_arm_qpos()
        phase = "search"

        if det is not None:
            detection_streak += 1
            current_box, center_px, area_px = _detection_box_and_center(det)
            desired_box_size_px = max(48.0, cfg.live.target_box_fraction * float(min(frame.shape[:2])))
            desired_points_px = centered_box_corners(frame.shape[:2], desired_box_size_px)
            current_points_px = box_xyxy_to_corners(current_box)
            try:
                object_world = _ray_plane_intersection_world(center_px, cfg.fixed_camera_pose, cfg.camera, cfg.sim.target_height_m)
            except ValueError:
                desired_area_px = max(1.0, float(desired_box_size_px * desired_box_size_px))
                depth_est = float(np.clip(cfg.live.tracking_depth_m * np.sqrt(area_px / desired_area_px), 0.20, 2.50))
                center_cam = _pixel_to_camera_point(center_px, depth_est, cfg.camera)
                object_world = cfg.fixed_camera_pose.transform_point(center_cam)
            object_world[2] = cfg.sim.target_height_m
            object_world = _lowpass_vector(smoothed_object_world, object_world, cfg.live.detection_smoothing_alpha)
            smoothed_object_world = object_world
            current_target_world = object_world.copy()
            target_trail.append(object_world.copy())
            if detection_streak <= cfg.live.acquire_hold_frames:
                phase = "acquiring"
                desired_hand_pose = current_hand_pose.copy()
                q_target = robot.current_arm_qpos()
                smoothed_q_target = q_target
            else:
                horizontal_offset = current_hand_pose.position - object_world
                horizontal_offset[2] = 0.0
                if np.linalg.norm(horizontal_offset) < 1e-6:
                    horizontal_offset = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                horizontal_offset = normalize(horizontal_offset)
                desired_camera_pos = object_world + horizontal_offset * cfg.live.standoff_m
                desired_camera_pos[2] = object_world[2]
                desired_camera_pose = look_at(desired_camera_pos, object_world, np.array([0.0, 0.0, 1.0], dtype=np.float64))
                desired_hand_pose = compose(desired_camera_pose, inverse_pose(cfg.eye_camera_offset))
                camera_twist = pose_servo.camera_twist(current_hand_pose, desired_hand_pose)
                twist_world = np.concatenate(
                    [
                        current_hand_pose.rotation @ camera_twist[:3],
                        current_hand_pose.rotation @ camera_twist[3:],
                    ]
                )
                q_target = _arm_joint_targets_from_twist(robot, twist_world, cfg.sim.ik_damping, 1.0 / 30.0)
                q_target = _lowpass_vector(
                    smoothed_q_target,
                    q_target,
                    cfg.live.joint_target_smoothing_alpha,
                    cfg.live.joint_target_max_step_rad,
                )
                smoothed_q_target = q_target
        else:
            detection_streak = 0
            aligned_stable_frames = 0
            current_target_world = None
            desired_hand_pose = current_hand_pose.copy()
            smoothed_q_target = q_target

        if len(current_points_px) and len(desired_points_px):
            feature_error_norm = board_feature_error(current_points_px, desired_points_px, cfg.camera)

        robot.apply_joint_targets(q_target, gripper_ctrl=cfg.sim.gripper_ctrl)
        mujoco.mj_step(scene.model, scene.data)

        ee_pose = robot.current_ee_pose()
        ee_trail.append(ee_pose.position.copy())
        position_error_norm = float(np.linalg.norm(ee_pose.position - desired_hand_pose.position))
        if det is None:
            phase = "search"
        elif detection_streak <= cfg.live.acquire_hold_frames:
            phase = "acquiring"
        else:
            aligned = position_error_norm < cfg.live.align_position_threshold_m and feature_error_norm < cfg.live.align_feature_threshold
            if aligned:
                aligned_stable_frames += 1
            else:
                aligned_stable_frames = 0
            phase = "tracking" if aligned_stable_frames >= cfg.live.align_hold_frames or detection_streak >= cfg.live.align_timeout_frames else "aligning"
        if viewer_session is not None:
            viewer_session.set_status(
                _viewer_status_lines(
                    "live-camera",
                    phase,
                    cfg.live.prompt,
                    det.label if det is not None else None,
                    position_error_norm,
                    feature_error_norm,
                    robot.current_arm_qpos(),
                )
            )
            if not viewer_session.is_running():
                break
        overlay = overlay_detection(frame, det)
        if len(current_points_px):
            overlay = overlay_points(overlay, current_points_px, color=(0, 255, 0))
        if len(desired_points_px):
            overlay = overlay_points(overlay, desired_points_px, color=(0, 165, 255))
        if det is not None and det.mask is not None:
            overlay = overlay_thumbnail(overlay, mask_thumbnail(det.mask), corner="lower_right", margin=14)
        overlay = _preview_panel(
            overlay,
            "GROUNDING DINO + SAM2",
            cfg.live.prompt,
            [
                f"phase: {phase}",
                f"pos err {position_error_norm:.3f} m",
                f"feat err {feature_error_norm:.4f}",
                f"detection: {det.label if det is not None else 'none'}",
                "green = detection",
                "orange = goal box",
            ],
        )
        map_panel = world_map_panel(
            list(target_trail),
            list(ee_trail),
            current_target=current_target_world,
            current_ee=ee_pose.position,
            center_xy=workspace_center_xy,
            extent_xy=workspace_extent_xy,
            phase=phase,
            position_error_norm=position_error_norm,
            feature_error_norm=feature_error_norm,
        )

        samples.append(
            TrackingSample(
                time_s=idx / 30.0,
                target_pose=desired_hand_pose,
                camera_pose=current_camera_pose,
                ee_pose=ee_pose,
                joint_positions=robot.current_arm_qpos(),
                joint_targets=q_target,
                phase=phase,
                pixel_error_norm=float(np.linalg.norm(current_points_px - desired_points_px)) if len(current_points_px) and len(desired_points_px) else 0.0,
                position_error_norm=position_error_norm,
                feature_error_norm=feature_error_norm,
                detection=det,
            )
        )

        if recorder is not None:
            recorder.write(overlay)
        if viewer_session is not None:
            viewer_session.set_motion_traces(
                list(target_trail),
                list(ee_trail),
                current_target=current_target_world,
                current_ee=ee_pose.position,
                target_pose=None,
                ee_pose=ee_pose,
                workspace_center=workspace_plane_center,
                workspace_size=workspace_plane_size,
            )
            viewer_session.set_image_panels([overlay, map_panel])
            viewer_session.sync()
        if show_cv2_windows:
            cv2.imshow("live_camera", overlay)
            cv2.imshow("world_map", map_panel)
            if viewer_session is None and world_renderer is not None:
                world_image = world_renderer.render(scene.data)
                if world_image.ndim == 3 and world_image.shape[-1] == 3:
                    world_image = cv2.cvtColor(world_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("world", world_image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        idx += 1

    if capture is not None:
        capture.close()
    if recorder is not None:
        recorder.close()
    if world_renderer is not None:
        world_renderer.close()
    if viewer_session is not None:
        viewer_session.close()
    if show_cv2_windows:
        cv2.destroyAllWindows()

    out_dir = cfg.paths.results_dir / "live_camera"
    _save_summary_csv(samples, out_dir)
    _plot_metrics(samples, cfg.paths.plots_dir / "live_camera", "live_camera")
    return RunSummary(name="live_camera", output_dir=out_dir, samples=samples)


def run_calibration(cfg: AppConfig | None = None, source: str | None = None, frame_count: int = 24) -> tuple[np.ndarray, np.ndarray, float, int]:
    cfg = build_default_config() if cfg is None else cfg
    if source is not None:
        cfg.live.source = source
    cfg.paths.ensure()

    if cfg.live.source == "camera":
        capture_source: int | str = cfg.live.device_index
    elif cfg.live.source == "video":
        if not cfg.live.video_path:
            raise ValueError("video_path is required when source='video'.")
        capture_source = cfg.live.video_path
    else:
        capture_source = cfg.live.source

    capture = VideoCaptureStream(capture_source, backend=cfg.live.camera_backend, width=cfg.live.camera_width, height=cfg.live.camera_height)
    frames: list[np.ndarray] = []
    tracker = CharucoTracker(cfg.board)
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame = maybe_resize(frame, width=cfg.live.camera_width, height=cfg.live.camera_height)
        frames.append(frame)
        if cfg.live.display:
            _, obs = tracker.estimate_pose(frame, cfg.camera)
            preview = frame.copy()
            if obs is not None:
                preview = overlay_points(preview, obs.points_px, color=(0, 255, 0))
            cv2.imshow("calibration", preview)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.close()
    if cfg.live.display:
        cv2.destroyAllWindows()
    if not frames:
        raise RuntimeError("No frames collected for calibration.")

    camera_matrix, distortion, rms, used = calibrate_from_charuco_images(frames, cfg.board, (cfg.live.camera_height, cfg.live.camera_width))
    out_dir = cfg.paths.calibration_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "camera_calibration.npz", camera_matrix=camera_matrix, distortion=distortion, rms=rms, used=used)
    return camera_matrix, distortion, rms, used
