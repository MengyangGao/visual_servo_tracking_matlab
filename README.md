# Robotics Vision Projects

This repository is an exploration of visual servo tracking, implemented with PUMA560 robotic arm in Matlab, and Franka Panda robotic arm in Mujoco, 

- `matlab/`: the MATLAB implementation, both eye-in-hand and fixed-eye two views.
- `mujoco/`: the MuJoCo + Python implementation, with general perception tools like grounding dino and grounded sam 2.

## MATLAB

The MATLAB project covers:

- `T1`: ChArUco-based camera calibration
- `T2`: position-based tracking with fixed-camera and eye-in-hand simulation modes
- `T3`: feature-based tracking with an IBVS control loop
- real-camera follow and IBVS demos that reuse the saved calibration parameters

Technical report:

- [`matlab/report.md`](matlab/report.md)

Main entry point:

```matlab
addpath(genpath(pwd));
results = run_demo();
```

Refresh the real-camera calibration parameters:

```matlab
addpath(genpath(pwd));
results = run_live_camera_calibration();
```

Run the real-camera follow or IBVS demos:

```matlab
addpath(genpath(pwd));
follow = run_real_camera_follow();
ibvs = run_real_camera_ibvs();
```

Manual Start/Stop mode:

```matlab
follow = run_real_camera_follow('RunMode', 'manual');
ibvs = run_real_camera_ibvs('RunMode', 'manual');
```

Public MATLAB assets:

- Printable ChArUco board: [`matlab/assets/charuco_board_printable.png`](matlab/assets/charuco_board_printable.png), [`matlab/assets/charuco_board_printable.pdf`](matlab/assets/charuco_board_printable.pdf)
- Saved camera parameters: [`matlab/assets/cameraParams.mat`](matlab/assets/cameraParams.mat)
- Technical report assets: [`matlab/assets/report/`](matlab/assets/report/)

Board parameters:

- pattern: `7 x 5`
- dictionary: `DICT_4X4_1000`
- checker size: `30 mm`
- marker size: `22.5 mm`
- image size: `2942 x 2102`

To regenerate or modify the board:

- edit `cfg.calibration.charuco` and `cfg.realCamera.charuco` in [`matlab/src/config.m`](matlab/src/config.m)
- keep those two sections aligned so the simulation and real-camera paths use the same board
- run `run_charuco_board_asset()` to refresh the printable PNG/PDF in `matlab/assets/`
- run `run_live_camera_calibration()` if you also want to refresh the saved camera parameters

Results and logs are generated locally under `matlab/results/`.

## MuJoCo

The MuJoCo project uses a continuous visual-servo loop with a three-panel dashboard:

- MuJoCo world
- robot follow view
- camera / detection view

It supports:

- simulation and real-camera modes
- `oracle`, `heuristic`, and `grounded-sam2` perception backends
- `default`, `small`, and `lite` vision presets
- CLI target prompts such as `cup`, `phone`, `mouse`, and `apple`
- `auto` and `manual` run modes

Install:

```bash
conda run -n mujoco python -m pip install -e .
```

Enable the open-vocabulary backend:

```bash
conda run -n mujoco python -m pip install -e ".[open-vocab]"
```

Run simulation:

```bash
conda run -n mujoco python -m mujoco_servo sim --prompt "cup" --backend oracle --steps 240
```

Run simulation with open-vocabulary perception:

```bash
conda run -n mujoco python -m mujoco_servo sim --prompt "cup" --backend grounded-sam2 --vision-preset lite --steps 240
```

Run real-camera mode on macOS:

```bash
mjpython -m mujoco_servo camera --prompt "cup" --backend grounded-sam2 --vision-preset lite --run-mode manual
```

GUI launcher:

```bash
conda run -n mujoco python -m mujoco_servo gui
```

Available cameras:

```bash
conda run -n mujoco python -m mujoco_servo cameras
```

The first `grounded-sam2` run downloads the Grounding DINO weights and the SAM 2 checkpoint into `mujoco/outputs/hf_cache`.

## Repository layout

- `matlab/`: MATLAB implementation and report
- `mujoco/`: MuJoCo + Python rebuild
- `reference/`: local reference checkouts

