
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────────────────────
# URDF generation for a 12-DOF serial-chain robot arm
# ──────────────────────────────────────────────────────────────────────────────

_JOINT_AXES = [
    "0 0 1",  # 1  – base yaw
    "0 1 0",  # 2  – shoulder pitch
    "1 0 0",  # 3  – shoulder roll
    "0 0 1",  # 4  – upper-arm twist
    "0 1 0",  # 5  – elbow pitch
    "1 0 0",  # 6  – elbow roll
    "0 0 1",  # 7  – forearm twist
    "0 1 0",  # 8  – wrist pitch
    "1 0 0",  # 9  – wrist roll
    "0 0 1",  # 10 – second wrist twist
    "0 1 0",  # 11 – second wrist pitch
    "0 0 1",  # 12 – tool rotation
]

_LINK_COLORS = [
    "0.2 0.4 0.8 1",
    "0.2 0.6 0.8 1",
    "0.3 0.7 0.5 1",
    "0.5 0.7 0.3 1",
    "0.7 0.6 0.2 1",
    "0.8 0.4 0.2 1",
    "0.8 0.2 0.3 1",
    "0.6 0.2 0.7 1",
    "0.4 0.2 0.8 1",
    "0.2 0.3 0.8 1",
    "0.2 0.5 0.6 1",
    "0.3 0.6 0.4 1",
]

_LINK_LEN  = 0.07   # m
_LINK_R    = 0.015  # m
_LINK_MASS = 0.4    # kg


def _link_xml(name: str, color: str, length: float, radius: float, mass: float) -> str:
    half = length / 2
    ixx = mass * (3 * radius**2 + length**2) / 12
    izz = mass * radius**2 / 2
    return f"""
  <link name="{name}">
    <visual>
      <origin xyz="0 0 {half}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
      <material name="m_{name}"><color rgba="{color}"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 {half}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
    </collision>
    <inertial>
      <mass value="{mass}"/>
      <origin xyz="0 0 {half}"/>
      <inertia ixx="{ixx:.6f}" ixy="0" ixz="0"
               iyy="{ixx:.6f}" iyz="0" izz="{izz:.6f}"/>
    </inertial>
  </link>"""


def _joint_xml(name, parent, child, origin_z, axis, lower=-3.14, upper=3.14) -> str:
    return f"""
  <joint name="{name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="0 0 {origin_z}" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit lower="{lower}" upper="{upper}" effort="100" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>"""


def generate_robot_urdf(path: str) -> None:
    """Write a 12-DOF serial-chain robot URDF to *path*."""
    lines = ['<?xml version="1.0"?>', '<robot name="arm_12dof">']

    # Base
    lines.append("""
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.05"/></geometry>
      <material name="grey"><color rgba="0.4 0.4 0.4 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.05"/></geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.025"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>""")

    prev = "base_link"
    prev_z = 0.05
    for i in range(12):
        link_name  = f"link_{i}"
        joint_name = f"joint_{i}"
        lines.append(_link_xml(link_name, _LINK_COLORS[i], _LINK_LEN, _LINK_R, _LINK_MASS))
        lines.append(_joint_xml(joint_name, prev, link_name, prev_z, _JOINT_AXES[i]))
        prev   = link_name
        prev_z = _LINK_LEN

    lines.append("</robot>")
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class RobotArmEnv(gym.Env):
    """
    12-DOF serial-chain robot arm in PyBullet.

    Observations (48-dim):
      joint_positions (12) | joint_velocities (12) | ee_pos (3) |
      ee_euler (3) | goal (3) | obstacle_positions (5×3=15)

    Actions (12-dim):
      target joint positions scaled to [-π, π]

    Reward shaping:
      -dist_to_goal
      +50 success bonus
      -100 collision penalty (terminates)
      -proximity penalty when EE < safety_margin from any obstacle
      -0.01 × sum(joint_vel²)   (smoothness)
      -0.05 step penalty         (efficiency)
    """

    metadata = {"render_modes": ["human"]}

    NUM_JOINTS    = 12
    NUM_OBSTACLES = 5
    SAFETY_MARGIN = 0.15   # m
    GOAL_THRESH   = 0.05   # m – success radius
    OBS_DIM       = NUM_JOINTS * 2 + 3 + 3 + 3 + NUM_OBSTACLES * 3  # = 48

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 500,
        render_fps: float = 20.0,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.render_fps  = max(1.0, float(render_fps))
        self._render_dt  = 1.0 / self.render_fps
        self._physics_client: int | None = None
        self.steps = 0

        # Build URDF once alongside this file
        self._urdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "robot_12dof.urdf")
        if not os.path.exists(self._urdf_path):
            generate_robot_urdf(self._urdf_path)

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.NUM_JOINTS,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

        self._obs_positions: list[np.ndarray] = []
        self._obs_ids: list[int] = []
        self.goal = np.zeros(3, dtype=np.float32)
        self.robot: int = -1
        self.plane: int = -1
        self._goal_vis: int = -1

    # ── internal helpers ──────────────────────────────────────────────────────

    def _connect(self):
        if self.render_mode == "human":
            client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=client)
            p.resetDebugVisualizerCamera(1.5, 45, -30, [0, 0, 0.3],
                                         physicsClientId=client)
        else:
            client = p.connect(p.DIRECT)
        self._physics_client = client
        p.resetSimulation(physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        return client

    def _cid(self) -> int:
        return self._physics_client  # type: ignore[return-value]

    def _controllable_joints(self) -> list[int]:
        joints = []
        for i in range(p.getNumJoints(self.robot, physicsClientId=self._cid())):
            info = p.getJointInfo(self.robot, i, physicsClientId=self._cid())
            if info[2] != p.JOINT_FIXED:
                joints.append(i)
        return joints

    def _load_world(self):
        cid = self._cid()
        self.plane = p.loadURDF("plane.urdf", physicsClientId=cid)
        self.robot  = p.loadURDF(
            self._urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=cid,
        )
        self._joint_idx = self._controllable_joints()
        assert len(self._joint_idx) == self.NUM_JOINTS, (
            f"Expected 12 joints, got {len(self._joint_idx)}"
        )

    def _reset_world(self):
        cid = self._cid()
        p.resetSimulation(physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(1.5, 45, -30, [0, 0, 0.3],
                                         physicsClientId=cid)

    def _spawn_obstacles(self):
        cid = self._cid()
        self._obs_ids.clear()
        self._obs_positions.clear()
        rng = np.random.default_rng()

        for i in range(self.NUM_OBSTACLES):
            # Keep obstacles away from base center and goal area
            while True:
                pos = rng.uniform([-0.45, -0.45, 0.05], [0.45, 0.45, 0.55])
                if np.linalg.norm(pos[:2]) > 0.12:
                    break

            if i % 2 == 0:
                r   = rng.uniform(0.04, 0.09)
                col = p.createCollisionShape(p.GEOM_SPHERE, radius=r,
                                             physicsClientId=cid)
                vis = p.createVisualShape(p.GEOM_SPHERE, radius=r,
                                          rgbaColor=[0.9, 0.2, 0.2, 0.85],
                                          physicsClientId=cid)
            else:
                he  = rng.uniform(0.03, 0.07, 3)
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he,
                                             physicsClientId=cid)
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=he,
                                          rgbaColor=[0.9, 0.6, 0.1, 0.85],
                                          physicsClientId=cid)

            oid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=pos.tolist(),
                physicsClientId=cid,
            )
            self._obs_ids.append(oid)
            self._obs_positions.append(pos.astype(np.float32))

    def _sample_goal(self) -> np.ndarray:
        rng = np.random.default_rng()
        while True:
            g = rng.uniform([-0.4, -0.4, 0.15], [0.4, 0.4, 0.65]).astype(np.float32)
            # keep goal away from obstacles
            if all(np.linalg.norm(g - op) > 0.12 for op in self._obs_positions):
                return g

    def _add_goal_marker(self):
        cid = self._cid()
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                  rgbaColor=[0.1, 0.9, 0.1, 0.6],
                                  physicsClientId=cid)
        self._goal_vis = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=self.goal.tolist(),
            physicsClientId=cid,
        )

    def _get_ee_pos(self) -> np.ndarray:
        state = p.getLinkState(self.robot, self._joint_idx[-1],
                               physicsClientId=self._cid())
        return np.array(state[0], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        cid = self._cid()
        js  = p.getJointStates(self.robot, self._joint_idx, physicsClientId=cid)
        jpos = np.array([s[0] for s in js], dtype=np.float32)
        jvel = np.array([s[1] for s in js], dtype=np.float32)

        ee_state = p.getLinkState(self.robot, self._joint_idx[-1], physicsClientId=cid)
        ee_pos   = np.array(ee_state[0], dtype=np.float32)
        ee_euler = np.array(p.getEulerFromQuaternion(ee_state[1]), dtype=np.float32)

        obs_flat = np.concatenate(self._obs_positions) if self._obs_positions \
            else np.zeros(self.NUM_OBSTACLES * 3, dtype=np.float32)

        return np.concatenate([jpos, jvel, ee_pos, ee_euler, self.goal, obs_flat])

    def _check_collision(self) -> bool:
        cid = self._cid()
        for oid in self._obs_ids:
            if p.getContactPoints(self.robot, oid, physicsClientId=cid):
                return True

        # Base link (index -1) is mounted at the floor by design and may touch
        # the plane; only treat non-base floor contacts as collisions.
        plane_contacts = p.getContactPoints(self.robot, self.plane, physicsClientId=cid)
        for c in plane_contacts:
            if c[3] != -1:
                return True

        # self-collision (skip same-link pairs)
        contacts = p.getContactPoints(self.robot, self.robot, physicsClientId=cid)
        for c in contacts:
            if abs(c[3] - c[4]) > 1:   # non-adjacent links
                return True
        return False

    def _min_obs_dist(self, ee_pos: np.ndarray) -> float:
        if not self._obs_positions:
            return float("inf")
        return min(float(np.linalg.norm(ee_pos - op)) for op in self._obs_positions)

    # ── gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):  # noqa: ARG002
        super().reset(seed=seed)

        if self._physics_client is None:
            self._connect()
        else:
            self._reset_world()

        self._load_world()
        self._spawn_obstacles()
        self.goal  = self._sample_goal()
        self.steps = 0

        if self.render_mode == "human":
            self._add_goal_marker()

        # Randomise starting joint positions slightly
        cid = self._cid()
        for jidx in self._joint_idx:
            init = np.random.uniform(-0.2, 0.2)
            p.resetJointState(self.robot, jidx, targetValue=init,
                              physicsClientId=cid)

        # Settle
        for _ in range(10):
            p.stepSimulation(physicsClientId=cid)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        cid     = self._cid()
        action  = np.clip(action, -1, 1).astype(np.float64)
        targets = (action * np.pi).tolist()

        p.setJointMotorControlArray(
            self.robot,
            self._joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targets,
            positionGains=[0.3] * self.NUM_JOINTS,
            velocityGains=[0.1] * self.NUM_JOINTS,
            physicsClientId=cid,
        )

        for _ in range(5):
            p.stepSimulation(physicsClientId=cid)

        if self.render_mode == "human":
            # Pace control loop in GUI mode so movement remains readable.
            time.sleep(self._render_dt)

        obs      = self._get_obs()
        ee_pos   = obs[self.NUM_JOINTS * 2 : self.NUM_JOINTS * 2 + 3]
        jvel     = obs[self.NUM_JOINTS : self.NUM_JOINTS * 2]

        dist      = float(np.linalg.norm(ee_pos - self.goal))
        collision = self._check_collision()
        min_dist  = self._min_obs_dist(ee_pos)
        success   = dist < self.GOAL_THRESH

        # ── reward ───────────────────────────────────────────────────────────
        reward  = -dist                                          # goal distance
        reward -= 0.05                                           # time step
        reward -= 0.01 * float(np.sum(jvel ** 2))               # smoothness

        if success:
            reward += 50.0

        if collision:
            reward -= 100.0

        if min_dist < self.SAFETY_MARGIN and not collision:
            reward -= (self.SAFETY_MARGIN - min_dist) * 25.0    # proximity

        # ── termination ───────────────────────────────────────────────────────
        terminated = bool(success or collision)
        truncated  = bool(self.steps >= self.max_steps)

        info = {
            "collision":    collision,
            "success":      success,
            "dist_to_goal": dist,
            "min_obs_dist": min_dist,
        }

        self.steps += 1
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass  # GUI mode streams automatically when render_mode="human"

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
            self._physics_client = None
