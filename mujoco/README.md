# MuJoCo Vision Servo

This branch is the Python + MuJoCo rebuild of the vision-servo project.

## What it does

- Uses Franka Panda from `reference/mujoco_menagerie` when available.
- Runs a continuous visual-servo loop instead of one-shot detections.
- Keeps the target moving in simulation so the follow behavior is visible.
- Uses image-space corners plus a standoff pose target, so the end-effector stays facing the target at roughly 30 cm.
- Supports simulation and real-camera modes with the same controller.
- Supports `oracle`, `heuristic`, and `grounded-sam2` perception backends.

## Perception backends

- `oracle`: deterministic backend for simulation and tests.
- `heuristic`: prompt-guided classical fallback.
- `grounded-sam2`: Grounding DINO + SAM 2 when open-vocabulary weights are available.

Vision presets:

- `default`: `IDEA-Research/grounding-dino-base` + `facebook/sam2.1-hiera-base-plus`
- `small`: `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-small`
- `lite`: `IDEA-Research/grounding-dino-tiny` + `facebook/sam2.1-hiera-tiny`

If SAM 2 is not available, the backend falls back to a box-mask mode so the controller still runs.

## Install

```bash
conda run -n mujoco python -m pip install -e .
```

To enable the open-vocabulary backend:

```bash
conda run -n mujoco python -m pip install -e ".[open-vocab]"
```

If you have a local checkout of `reference/Grounded-SAM-2`, the backend can use it automatically. Otherwise set `MUJOCO_SERVO_SAM2_REPO` and `MUJOCO_SERVO_SAM2_CHECKPOINT`.

The first `grounded-sam2` run downloads the Grounding DINO weights and the SAM 2 checkpoint into `mujoco/outputs/hf_cache`.

## Run

Simulation smoke test with a moving target:

```bash
conda run -n mujoco python -m mujoco_servo sim --prompt "cup" --backend oracle --steps 240
```

Simulation with open-vocabulary perception:

```bash
conda run -n mujoco python -m mujoco_servo sim --prompt "cup" --backend grounded-sam2 --vision-preset lite --steps 240
```

Real-camera mode:

```bash
mjpython -m mujoco_servo camera --prompt "cup" --backend grounded-sam2 --vision-preset lite --run-mode manual
```

The camera mode uses the system camera and the fixed MuJoCo world viewer. The robot view follows the end-effector and the detected target so motion stays visible.

GUI:

```bash
conda run -n mujoco python -m mujoco_servo gui
```

List available cameras:

```bash
conda run -n mujoco python -m mujoco_servo cameras
```

## Notes

- The controller uses image corners, filtered detections, and a standoff pose target.
- The simulation target moves continuously so the follow behavior is easy to see.
- On macOS, camera access still depends on system permission prompts.
- If the live camera path is unavailable, the simulation path remains the primary regression test.
