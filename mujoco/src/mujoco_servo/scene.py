from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .config import AppConfig, BoardConfig
from .geometry import board_pose_from_center, mujoco_camera_pose_from_internal_pose, rotation_matrix_to_quat_wxyz
from .image_features import render_charuco_board_image
from .types import Pose


@dataclass(slots=True)
class SceneBundle:
    model: mujoco.MjModel
    data: mujoco.MjData
    board_image: np.ndarray
    board_texture_path: Path
    scene_xml_path: Path
    board_cfg: BoardConfig
    target_body_name: str = "servo_target"
    world_camera_name: str = "world_camera"


def _xml_escape(path: Path) -> str:
    return str(path).replace("&", "&amp;")


def _resolve_panda_xml(cfg: AppConfig) -> Path:
    source = cfg.paths.panda_xml
    destination = cfg.paths.resolved_panda_xml
    asset_dir = source.parent / "assets"

    text = source.read_text(encoding="utf-8")

    def _rewrite_file_attr(match: re.Match[str]) -> str:
        file_path = Path(match.group(1))
        if file_path.is_absolute():
            return match.group(0)
        return f'file="{(asset_dir / file_path).as_posix()}"'

    resolved = re.sub(r'file="([^"]+)"', _rewrite_file_attr, text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists() or destination.read_text(encoding="utf-8") != resolved:
        destination.write_text(resolved, encoding="utf-8")
    return destination


def _build_scene_xml(cfg: AppConfig) -> str:
    render_charuco_board_image(cfg.board, cfg.paths.board_texture_png)
    panda_xml = _resolve_panda_xml(cfg)

    target_half_w = cfg.board.width_m * 0.5
    target_half_h = cfg.board.height_m * 0.5
    target_thickness = cfg.board.thickness_m * 0.5
    target_pose = board_pose_from_center(cfg.target_center_m, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    target_quat = rotation_matrix_to_quat_wxyz(target_pose.rotation)
    cam = mujoco_camera_pose_from_internal_pose(cfg.fixed_camera_pose)
    quat = rotation_matrix_to_quat_wxyz(cam.rotation)

    return f"""
<mujoco model="panda_servo_scene">
  <include file="{_xml_escape(panda_xml)}"/>

  <statistic center="0.3 0.0 0.4" extent="1.2"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.12 0.18 0.24 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.32 0.48 0.68" rgb2="0.02 0.02 0.04" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.22 0.26 0.30" rgb2="0.14 0.16 0.18"
      markrgb="0.82 0.82 0.82" width="512" height="512"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0.12"/>
    <texture type="2d" name="charuco_board" file="{_xml_escape(cfg.paths.board_texture_png)}"/>
    <material name="charuco_board_mat" texture="charuco_board" texuniform="true" reflectance="0.05"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.6" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <camera name="{cfg.sim.world_camera_name}" pos="{cam.position[0]} {cam.position[1]} {cam.position[2]}"
      quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"/>
    <body name="{cfg.sim.world_camera_name}_marker" pos="{cam.position[0]} {cam.position[1]} {cam.position[2]}">
      <site name="{cfg.sim.world_camera_name}_site" size="0.01" rgba="0.1 0.7 0.9 1"/>
    </body>
    <body name="servo_target" mocap="true" pos="{target_pose.position[0]} {target_pose.position[1]} {target_pose.position[2]}" quat="{target_quat[0]} {target_quat[1]} {target_quat[2]} {target_quat[3]}">
      <geom name="servo_target_plate" type="box" size="{target_half_w} {target_half_h} {target_thickness}"
        material="charuco_board_mat" rgba="1 1 1 1"/>
      <site name="servo_target_center" pos="0 0 0" size="0.004" rgba="1 0.2 0.2 1"/>
      <site name="servo_target_normal" pos="0 0 {target_thickness + 0.002}" size="0.003" rgba="0.2 0.8 1 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build_scene(cfg: AppConfig) -> SceneBundle:
    cfg.paths.ensure()
    board_image = render_charuco_board_image(cfg.board, cfg.paths.board_texture_png)
    xml_text = _build_scene_xml(cfg)
    cfg.paths.generated_scene_xml.parent.mkdir(parents=True, exist_ok=True)
    cfg.paths.generated_scene_xml.write_text(xml_text, encoding="utf-8")
    model = mujoco.MjModel.from_xml_path(str(cfg.paths.generated_scene_xml))
    data = mujoco.MjData(model)
    return SceneBundle(
        model=model,
        data=data,
        board_image=board_image,
        board_texture_path=cfg.paths.board_texture_png,
        scene_xml_path=cfg.paths.generated_scene_xml,
        board_cfg=cfg.board,
    )


def set_mocap_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pose: Pose) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"Body '{body_name}' is not a mocap body.")
    data.mocap_pos[mocap_id] = pose.position
    quat = rotation_matrix_to_quat_wxyz(pose.rotation)
    data.mocap_quat[mocap_id] = quat


def get_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> Pose:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return Pose(np.array(data.xpos[body_id], dtype=np.float64), np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3))


def get_body_jacobian(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return np.vstack([jacp, jacr])


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    qpos = np.array(model.key_qpos[key_id], dtype=np.float64)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
