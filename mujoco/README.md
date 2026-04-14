# MuJoCo Vision Servo

Python + MuJoCo reimplementation of the vision servo project.

## What is covered

- Franka Panda simulation using the upstream `reference/mujoco_menagerie/franka_emika_panda` assets
- position-based visual servoing in simulation
- eye-in-hand visual servoing in simulation
- IBVS using four board corners
- live camera tracking on MacBook camera first
- open-vocabulary object detection via Grounding DINO tiny/small style models
- optional SAM2 box-conditioned mask refinement
- ChArUco calibration and board-based feature extraction
- a side-on tracking setup where the simulation target moves in a horizontal plane and the robot keeps a readable lateral motion in the viewer

## Layout

- `src/mujoco_servo/`: runtime, control, perception, scene, and CLI
- `tests/`: focused unit and smoke tests
- `outputs/`: generated runs, plots, videos, and temporary scene files

## Run

Install the package into the `mujoco` Conda environment:

```bash
conda run -n mujoco python -m pip install -e ./mujoco
```

That editable install is the cleanest way to get the `mujoco-servo` command
and to make code changes visible immediately. If you only want to run the
source tree without installing it, use:

```bash
conda run -n mujoco bash -lc 'PYTHONPATH=mujoco/src python -m mujoco_servo sim --task t2-fixed'
```

Then run a simulation demo:

```bash
conda run -n mujoco mujoco-servo sim --task t2-fixed
```

This opens the standard MuJoCo viewer with mouse drag/zoom/rotate controls.
On macOS the CLI auto-relaunches under `mjpython` so the viewer works from a
normal terminal command. The offscreen world render is kept only for fallback
and recording.

Run the eye-in-hand simulation:

```bash
conda run -n mujoco mujoco-servo sim --task t2-eye
```

Run IBVS in simulation:

```bash
conda run -n mujoco mujoco-servo sim --task t3-ibvs
```

Run live camera tracking:

```bash
conda run -n mujoco mujoco-servo camera --source camera --backend avfoundation --prompt "cup"
```

The live camera mode also opens the standard MuJoCo viewer so you can see the
robot motion while the object is being tracked. The live controller currently
uses plane-only tracking on a fixed horizontal workspace, so the arm moves
laterally instead of diving toward the floor. On macOS the OpenCV preview
window is suppressed because `cv2.imshow` is not stable under `mjpython`; the
SAM/DINO detection overlay is shown as a compact image panel inside the
MuJoCo viewer, including a mask thumbnail when SAM2 returns one, and the same
overlay is still written to recorded output. The viewer status text also shows
the current servo phase so the sequence reads as: detection, alignment, then
tracking.

Run camera calibration:

```bash
conda run -n mujoco mujoco-servo calibrate --source camera --backend avfoundation
```

Useful live-camera options:

- `--model-id IDEA-Research/grounding-dino-tiny`
- `--sam-model-id facebook/sam2.1-hiera-tiny`
- `--no-sam` to disable SAM2 if you want faster box-only tracking
- `--inference-max-side 960` to reduce open-vocabulary inference cost
- `--max-frames 10` for smoke tests or short runs

The live camera path uses the external camera as the visual input and drives the simulated Panda toward the detected object. The object pose is approximate and is driven by the detection box/mask geometry plus a configurable depth proxy, which is enough for a practical visual-servo demo without a real robot arm.

## Notes

- The Panda XML/MJCF/URDF files under `reference/mujoco_menagerie/franka_emika_panda` are treated as read-only inputs.
- The simulation scene is assembled in Python by wrapping the upstream Panda model.
- The live camera path is modular and can be extended to USB cameras and video files.
