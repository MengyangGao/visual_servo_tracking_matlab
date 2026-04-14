from __future__ import annotations

import argparse
import sys

from .config import build_default_config
from .startup import maybe_reexec_under_mjpython
from .runtime import run_calibration, run_live_camera, run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mujoco-servo", description="MuJoCo + Python visual servo demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("sim", help="Run simulation demos")
    sim.add_argument("--task", choices=["t2-fixed", "t2-eye", "t3-ibvs"], default="t2-fixed")
    sim.add_argument("--no-display", action="store_true", help="Disable GUI windows")
    sim.add_argument(
        "--render-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable offscreen world rendering for fallback/recording",
    )
    sim.add_argument("--steps", type=int, default=None, help="Override number of simulation steps")
    sim.add_argument("--record-video", action="store_true", help="Record a video to outputs")

    camera = subparsers.add_parser("camera", help="Run live camera tracking")
    camera.add_argument("--source", choices=["camera", "video"], default="camera")
    camera.add_argument("--device-index", type=int, default=0)
    camera.add_argument("--video-path", type=str, default=None)
    camera.add_argument("--backend", type=str, default=None, help="OpenCV capture backend, e.g. avfoundation, v4l2, dshow, auto")
    camera.add_argument("--prompt", type=str, default="charuco board")
    camera.add_argument("--model-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    camera.add_argument("--sam-model-id", type=str, default="facebook/sam2.1-hiera-tiny")
    camera.add_argument("--inference-max-side", type=int, default=None, help="Resize the camera frame before open-vocab inference")
    camera.add_argument("--target-box-fraction", type=float, default=None, help="Desired object box size as a fraction of the shorter image side")
    camera.add_argument("--tracking-depth", type=float, default=None, help="Fallback depth proxy used only if plane intersection fails")
    camera.add_argument("--standoff", type=float, default=None, help="Desired same-plane standoff from the tracked object")
    camera.add_argument("--max-frames", type=int, default=None, help="Limit live mode to a finite number of frames for testing")
    camera.add_argument("--no-sam", action="store_true", help="Disable SAM2 mask refinement and use Grounding DINO boxes only")
    camera.add_argument(
        "--render-world",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable offscreen world rendering for fallback/recording",
    )
    camera.add_argument("--no-display", action="store_true", help="Disable GUI windows")
    camera.add_argument("--record-video", action="store_true", help="Record a video to outputs")

    calib = subparsers.add_parser("calibrate", help="Collect ChArUco calibration images")
    calib.add_argument("--source", choices=["camera", "video"], default="camera")
    calib.add_argument("--device-index", type=int, default=0)
    calib.add_argument("--video-path", type=str, default=None)
    calib.add_argument("--backend", type=str, default=None, help="OpenCV capture backend, e.g. avfoundation, v4l2, dshow, auto")
    calib.add_argument("--frames", type=int, default=24)
    calib.add_argument("--no-display", action="store_true", help="Disable GUI windows")

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    maybe_reexec_under_mjpython(raw_argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    cfg = build_default_config()

    if args.command == "sim":
        cfg.sim.display = not args.no_display
        cfg.sim.render_world = args.render_world
        cfg.sim.steps = args.steps if args.steps is not None else cfg.sim.steps
        cfg.sim.record_video = args.record_video
        run_simulation(cfg=cfg, task=args.task, display=cfg.sim.display)
        return 0

    if args.command == "camera":
        cfg.live.display = not args.no_display
        cfg.live.source = args.source
        cfg.live.device_index = args.device_index
        cfg.live.video_path = args.video_path
        if args.backend is not None:
            cfg.live.camera_backend = args.backend
        cfg.live.prompt = args.prompt
        cfg.live.model_id = args.model_id
        cfg.live.sam_model_id = args.sam_model_id
        if args.inference_max_side is not None:
            cfg.live.inference_max_side = args.inference_max_side
        if args.target_box_fraction is not None:
            cfg.live.target_box_fraction = args.target_box_fraction
        if args.tracking_depth is not None:
            cfg.live.tracking_depth_m = args.tracking_depth
        if args.standoff is not None:
            cfg.live.standoff_m = args.standoff
        if args.no_sam:
            cfg.live.use_sam2 = False
        cfg.live.render_world = args.render_world
        cfg.live.record_video = args.record_video
        run_live_camera(cfg=cfg, prompt=args.prompt, source=args.source, display=cfg.live.display, max_frames=args.max_frames)
        return 0

    if args.command == "calibrate":
        cfg.live.display = not args.no_display
        cfg.live.source = args.source
        cfg.live.device_index = args.device_index
        cfg.live.video_path = args.video_path
        if args.backend is not None:
            cfg.live.camera_backend = args.backend
        run_calibration(cfg=cfg, source=args.source, frame_count=args.frames)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
