from __future__ import annotations

import mujoco
import numpy as np

from mujoco_servo.config import build_default_config
from mujoco_servo.robot import PandaRobot
from mujoco_servo.scene import build_scene


def test_scene_builds_and_exposes_robot_names() -> None:
    cfg = build_default_config()
    scene = build_scene(cfg)
    mujoco.mj_forward(scene.model, scene.data)
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "hand") >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "servo_target") >= 0
    assert mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_CAMERA, cfg.sim.world_camera_name) >= 0
    target_body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "servo_target")
    assert np.isclose(scene.data.xpos[target_body_id][2], cfg.sim.target_height_m)
    assert np.allclose(scene.data.xmat[target_body_id].reshape(3, 3)[:, 2], np.array([1.0, 0.0, 0.0]), atol=1e-6)
    robot = PandaRobot(scene.model, scene.data)
    assert robot.current_arm_qpos().shape == (7,)
