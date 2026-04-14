from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .geometry import clamp_to_limits, dls_solve, pose_error
from .scene import get_body_jacobian, get_body_pose, reset_to_home, set_mocap_pose
from .types import Pose


@dataclass(slots=True)
class PandaRobot:
    model: mujoco.MjModel
    data: mujoco.MjData
    ee_body_name: str = "hand"
    target_body_name: str = "servo_target"
    gripper_ctrl_name: str = "actuator8"
    gripper_open_ctrl: float = 255.0
    arm_joint_names: list[str] = field(init=False)
    arm_ctrl_names: list[str] = field(init=False)
    arm_joint_ids: list[int] = field(init=False)
    arm_dof_ids: list[int] = field(init=False)
    arm_qpos_ids: list[int] = field(init=False)
    arm_ctrl_ids: list[int] = field(init=False)
    gripper_ctrl_id: int = field(init=False)
    ee_body_id: int = field(init=False)
    target_body_id: int = field(init=False)
    arm_lower: np.ndarray = field(init=False)
    arm_upper: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.arm_joint_names = [f"joint{i}" for i in range(1, 8)]
        self.arm_ctrl_names = [f"actuator{i}" for i in range(1, 8)]
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.arm_joint_names]
        self.arm_dof_ids = [self.model.jnt_dofadr[jid] for jid in self.arm_joint_ids]
        self.arm_qpos_ids = [self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids]
        self.arm_ctrl_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.arm_ctrl_names]
        self.gripper_ctrl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.gripper_ctrl_name)
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_body_name)
        self.arm_lower = np.array([self.model.jnt_range[jid][0] for jid in self.arm_joint_ids], dtype=np.float64)
        self.arm_upper = np.array([self.model.jnt_range[jid][1] for jid in self.arm_joint_ids], dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        reset_to_home(self.model, self.data)
        self.set_gripper(self.gripper_open_ctrl)

    def set_gripper(self, ctrl_value: float | None = None) -> None:
        self.data.ctrl[self.gripper_ctrl_id] = self.gripper_open_ctrl if ctrl_value is None else float(ctrl_value)

    def current_arm_qpos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.arm_qpos_ids], dtype=np.float64).copy()

    def current_ee_pose(self) -> Pose:
        return get_body_pose(self.model, self.data, self.ee_body_name)

    def current_target_pose(self) -> Pose:
        return get_body_pose(self.model, self.data, self.target_body_name)

    def body_jacobian(self, body_name: str | None = None) -> np.ndarray:
        return get_body_jacobian(self.model, self.data, body_name or self.ee_body_name)

    def set_target_pose(self, pose: Pose) -> None:
        set_mocap_pose(self.model, self.data, self.target_body_name, pose)

    def arm_ctrl_from_qpos(self, qpos_target: np.ndarray) -> None:
        qpos_target = np.asarray(qpos_target, dtype=np.float64).reshape(-1)
        qpos_target = clamp_to_limits(qpos_target, self.arm_lower, self.arm_upper)
        for idx, q in zip(self.arm_ctrl_ids, qpos_target):
            self.data.ctrl[idx] = float(q)
        self.set_gripper()

    def solve_pose_ik(
        self,
        target_pose: Pose,
        damping: float = 1e-3,
        position_gain: float = 2.0,
        orientation_gain: float = 1.5,
        iterations: int = 8,
    ) -> np.ndarray:
        q = self.current_arm_qpos().copy()
        for _ in range(iterations):
            current = self.current_ee_pose()
            pos_err, rot_err = pose_error(current, target_pose)
            error = np.concatenate([position_gain * pos_err, orientation_gain * rot_err])
            jac = self.body_jacobian(self.ee_body_name)[:, : len(self.arm_joint_names)]
            dq = dls_solve(jac, error, damping=damping)
            q = clamp_to_limits(q + dq, self.arm_lower, self.arm_upper)
            self.data.qpos[self.arm_qpos_ids] = q
            mujoco.mj_forward(self.model, self.data)
        return q

    def apply_joint_targets(self, qpos_target: np.ndarray, gripper_ctrl: float | None = None) -> None:
        self.arm_ctrl_from_qpos(qpos_target)
        if gripper_ctrl is not None:
            self.set_gripper(gripper_ctrl)

    def step(self, nstep: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=nstep)
