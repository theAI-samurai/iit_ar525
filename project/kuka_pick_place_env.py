import os
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


class KukaPickPlaceEnv(gym.Env):
    """
    Kuka iiwa pick-and-place task: pick a cube from the left and place it on the right.

    Observation (28-dim):
      joint_pos(7) | joint_vel(7) | ee_pos(3) | obj_pos(3) |
      ee_to_obj(3) | obj_to_goal(3) | [gripper_cmd, is_grasped](2)

    Action (4-dim):
      ee_delta_xyz(3) in [-1,1] scaled to meters + gripper(1)
      gripper > 0 closes, <= 0 opens

        Optional strict two-phase curriculum:
            - Phase 1: fixed object for first N episodes.
            - Phase 2: randomized object around start pose.
    """

    metadata = {"render_modes": ["human"]}

    NUM_JOINTS = 7
    ACT_DIM = 4
    OBS_DIM = 28
    EE_LINK_IDX = 6

    TABLE_H = 0.30
    BLOCK_HALF = 0.025
    BLOCK_Z = TABLE_H + BLOCK_HALF

    OBJ_START = np.array([0.50, -0.18, BLOCK_Z], dtype=np.float32)
    OBJ_GOAL = np.array([0.50, 0.18, BLOCK_Z], dtype=np.float32)
    PHASE1_GOAL_Y_OFFSET = 0.12

    GRASP_DIST = 0.07
    PLACE_DIST = 0.07
    LIFT_HEIGHT = TABLE_H + 0.12

    JOINT_LIMITS = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05], dtype=np.float32)
    EE_DELTA_MAX = np.array([0.03, 0.03, 0.03], dtype=np.float32)

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 400,
        strict_two_phase_curriculum: bool = False,
        fixed_phase_episodes: int = 300,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.strict_two_phase_curriculum = strict_two_phase_curriculum
        self.fixed_phase_episodes = max(1, int(fixed_phase_episodes))

        self._physics_client: int | None = None
        self.steps = 0
        self._episode_idx = 0

        self._kuka_uid = -1
        self._object_uid = -1
        self._plane_uid = -1
        self._table_uid = -1
        self._goal_vis = -1

        self._is_grasped = False
        self._grasp_cid = None
        self._gripper_cmd = 0.0
        self._goal_pos = self.OBJ_GOAL.copy()

        self._stage = 0
        self._reached_grasp_zone = False
        self._reached_lift = False

        self._prev_dist_ee_obj = 0.0
        self._prev_dist_obj_goal = 0.0

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.ACT_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32)

    def _cid(self) -> int:
        return self._physics_client  # type: ignore[return-value]

    def _connect(self):
        if self.render_mode == "human":
            client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.3,
                cameraYaw=55,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 0.35],
                physicsClientId=client,
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=client)
        else:
            client = p.connect(p.DIRECT)

        self._physics_client = client
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(1 / 240.0, physicsClientId=client)
        return client

    def _load_world(self):
        cid = self._cid()
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

        self._plane_uid = p.loadURDF("plane.urdf", physicsClientId=cid)

        t_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.5, 0.5, self.TABLE_H / 2], physicsClientId=cid
        )
        t_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, self.TABLE_H / 2],
            rgbaColor=[0.85, 0.75, 0.60, 1],
            physicsClientId=cid,
        )
        self._table_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=t_col,
            baseVisualShapeIndex=t_vis,
            basePosition=[0, 0, self.TABLE_H / 2],
            physicsClientId=cid,
        )

        kuka_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self._kuka_uid = p.loadURDF(
            kuka_path,
            basePosition=[0, 0, self.TABLE_H],
            useFixedBase=True,
            physicsClientId=cid,
        )

        b_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[self.BLOCK_HALF] * 3, physicsClientId=cid
        )
        b_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.BLOCK_HALF] * 3,
            rgbaColor=[0.85, 0.15, 0.15, 1],
            physicsClientId=cid,
        )
        self._object_uid = p.createMultiBody(
            baseMass=0.08,
            baseCollisionShapeIndex=b_col,
            baseVisualShapeIndex=b_vis,
            basePosition=self.OBJ_START.tolist(),
            physicsClientId=cid,
        )

        if self.render_mode == "human":
            g_vis = p.createVisualShape(
                p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.1, 0.9, 0.1, 0.45], physicsClientId=cid
            )
            self._goal_vis = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=g_vis,
                basePosition=self.OBJ_GOAL.tolist(),
                physicsClientId=cid,
            )

    def _reset_grasp(self):
        if self._is_grasped and self._grasp_cid is not None:
            try:
                p.removeConstraint(self._grasp_cid, physicsClientId=self._cid())
            except Exception:
                pass
        self._is_grasped = False
        self._grasp_cid = None
        self._gripper_cmd = 0.0

    def _get_ee_pos(self) -> np.ndarray:
        state = p.getLinkState(self._kuka_uid, self.EE_LINK_IDX, physicsClientId=self._cid())
        return np.array(state[0], dtype=np.float32)

    def _get_obj_pos(self) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(self._object_uid, physicsClientId=self._cid())
        return np.array(pos, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        cid = self._cid()
        js = [p.getJointState(self._kuka_uid, j, physicsClientId=cid) for j in range(self.NUM_JOINTS)]
        jpos = np.array([s[0] for s in js], dtype=np.float32)
        jvel = np.array([s[1] for s in js], dtype=np.float32)
        ee_pos = self._get_ee_pos()
        obj_pos = self._get_obj_pos()

        return np.concatenate(
            [
                jpos,
                jvel,
                ee_pos,
                obj_pos,
                obj_pos - ee_pos,
                self._goal_pos - obj_pos,
                [self._gripper_cmd, float(self._is_grasped)],
            ]
        )

    def _update_gripper(self, cmd: float):
        self._gripper_cmd = cmd
        cid = self._cid()
        dist = float(np.linalg.norm(self._get_ee_pos() - self._get_obj_pos()))
        want_close = cmd > 0

        # Exploration helper near the object.
        if dist < (self.GRASP_DIST * 0.75):
            want_close = True

        if want_close and not self._is_grasped and dist < self.GRASP_DIST:
            self._grasp_cid = p.createConstraint(
                self._kuka_uid,
                self.EE_LINK_IDX,
                self._object_uid,
                -1,
                p.JOINT_FIXED,
                [0, 0, 0],
                [0, 0, 0.03],
                [0, 0, 0],
                physicsClientId=cid,
            )
            self._is_grasped = True

        elif cmd <= 0 and self._is_grasped:
            try:
                p.removeConstraint(self._grasp_cid, physicsClientId=cid)
            except Exception:
                pass
            self._grasp_cid = None
            self._is_grasped = False

    def _check_collision(self) -> bool:
        cid = self._cid()
        num_links = p.getNumJoints(self._kuka_uid, physicsClientId=cid)

        for link in range(1, num_links):
            for body in (self._table_uid, self._plane_uid):
                pts = p.getContactPoints(
                    bodyA=self._kuka_uid,
                    bodyB=body,
                    linkIndexA=link,
                    physicsClientId=cid,
                )
                if pts and any(abs(pt[9]) > 2.0 for pt in pts):
                    return True

        pts = p.getContactPoints(bodyA=self._kuka_uid, bodyB=self._kuka_uid, physicsClientId=cid)
        for pt in pts:
            if abs(pt[3] - pt[4]) > 1:
                return True
        return False

    def _compute_reward(
        self,
        ee_pos: np.ndarray,
        obj_pos: np.ndarray,
        jvel: np.ndarray,
        jtorque: np.ndarray,
    ) -> tuple[float, bool, dict]:
        dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        dist_obj_goal = float(np.linalg.norm(obj_pos - self._goal_pos))
        obj_z = float(obj_pos[2])

        if not self._is_grasped:
            r_dist = (self._prev_dist_ee_obj - dist_ee_obj) * 5.0
        elif not self._reached_lift:
            r_dist = (self._prev_dist_obj_goal - dist_obj_goal) * 5.0
        else:
            # Strong progress signal once the block is airborne.
            r_dist = (self._prev_dist_obj_goal - dist_obj_goal) * 20.0

        self._prev_dist_ee_obj = dist_ee_obj
        self._prev_dist_obj_goal = dist_obj_goal

        r_stage = 0.0
        if not self._reached_grasp_zone and dist_ee_obj < self.GRASP_DIST:
            r_stage += 2.0
            self._reached_grasp_zone = True
            self._stage = max(self._stage, 1)

        if self._is_grasped and self._stage < 2:
            r_stage += 5.0
            self._stage = 2

        if self._is_grasped and obj_z >= self.LIFT_HEIGHT and not self._reached_lift:
            r_stage += 5.0
            self._reached_lift = True
            self._stage = max(self._stage, 3)

        # Require the block to have been lifted to prevent knock exploits.
        success = dist_obj_goal < self.PLACE_DIST and self._reached_lift
        r_success = 100.0 if success else 0.0

        collision = self._check_collision()
        r_collision = -10.0 if collision else 0.0

        r_velocity = -0.0002 * float(np.sum(jvel ** 2))
        r_torque = -0.00005 * float(np.sum(np.abs(jtorque)))
        r_time = -0.05

        if not self._is_grasped:
            r_dense = -0.3 * dist_ee_obj
            r_lift = 0.0
        elif not self._reached_lift:
            # Grasped but not yet lifted — keep lifting incentive.
            r_dense = -0.8 * dist_obj_goal
            r_lift = 2.0 * max(0.0, obj_z - self.BLOCK_Z)
        else:
            # Lifted — kill lift reward, strongly penalise goal distance.
            r_dense = -4.0 * dist_obj_goal
            r_lift = 0.0

        total = r_dist + r_dense + r_lift + r_stage + r_success + r_collision + r_velocity + r_torque + r_time

        components = {
            "r_distance": round(r_dist, 4),
            "r_dense": round(r_dense, 4),
            "r_lift": round(r_lift, 4),
            "r_stage": round(r_stage, 4),
            "r_success": round(r_success, 4),
            "r_collision": round(r_collision, 4),
            "r_energy": round(r_velocity + r_torque, 4),
            "r_time": round(r_time, 4),
        }
        return total, success, components

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._physics_client is not None:
            self._reset_grasp()
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass

        self._connect()
        self._load_world()
        self._reset_grasp()

        self.steps = 0
        self._episode_idx += 1
        self._stage = 0
        self._reached_grasp_zone = False
        self._reached_lift = False

        block_start = self.OBJ_START.copy()
        if self.strict_two_phase_curriculum and self._episode_idx <= self.fixed_phase_episodes:
            # Phase 1: fixed start to simplify early grasp/lift discovery.
            self._goal_pos = block_start.copy()
            self._goal_pos[1] += self.PHASE1_GOAL_Y_OFFSET
        else:
            # Phase 2: randomized start to improve robustness and generalization.
            rng = np.random.default_rng(seed)
            noise_scale = 0.03
            noise = rng.uniform(-noise_scale, noise_scale, size=2).astype(np.float32)
            block_start[:2] += noise
            self._goal_pos = self.OBJ_GOAL.copy()

        if self.render_mode == "human" and self._goal_vis != -1:
            p.resetBasePositionAndOrientation(
                self._goal_vis,
                self._goal_pos.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self._cid(),
            )

        p.resetBasePositionAndOrientation(
            self._object_uid,
            block_start.tolist(),
            [0, 0, 0, 1],
            physicsClientId=self._cid(),
        )

        pregrasp = [float(block_start[0]), float(block_start[1]), float(block_start[2] + 0.10)]
        ik = p.calculateInverseKinematics(
            self._kuka_uid,
            self.EE_LINK_IDX,
            pregrasp,
            physicsClientId=self._cid(),
        )
        for j in range(self.NUM_JOINTS):
            val = float(np.clip(ik[j], -self.JOINT_LIMITS[j], self.JOINT_LIMITS[j]))
            p.resetJointState(self._kuka_uid, j, targetValue=val, physicsClientId=self._cid())

        for _ in range(20):
            p.stepSimulation(physicsClientId=self._cid())

        ee_pos = self._get_ee_pos()
        obj_pos = self._get_obj_pos()
        self._prev_dist_ee_obj = float(np.linalg.norm(ee_pos - obj_pos))
        self._prev_dist_obj_goal = float(np.linalg.norm(obj_pos - self._goal_pos))

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        cid = self._cid()

        ee_pos = self._get_ee_pos().astype(np.float64)
        delta_xyz = action[:3] * self.EE_DELTA_MAX
        ee_target = ee_pos + delta_xyz

        ee_target[0] = np.clip(ee_target[0], 0.30, 0.72)
        ee_target[1] = np.clip(ee_target[1], -0.32, 0.32)
        ee_target[2] = np.clip(ee_target[2], self.BLOCK_Z + 0.01, 0.95)

        ik = p.calculateInverseKinematics(
            self._kuka_uid,
            self.EE_LINK_IDX,
            ee_target.tolist(),
            maxNumIterations=40,
            residualThreshold=1e-4,
            physicsClientId=cid,
        )
        targets = np.clip(np.array(ik[: self.NUM_JOINTS]), -self.JOINT_LIMITS, self.JOINT_LIMITS).tolist()

        p.setJointMotorControlArray(
            self._kuka_uid,
            list(range(self.NUM_JOINTS)),
            p.POSITION_CONTROL,
            targetPositions=targets,
            positionGains=[0.5] * self.NUM_JOINTS,
            velocityGains=[0.1] * self.NUM_JOINTS,
            physicsClientId=cid,
        )

        self._update_gripper(float(action[3]))

        for _ in range(10):
            p.stepSimulation(physicsClientId=cid)

        self.steps += 1
        obs = self._get_obs()
        ee_pos = obs[self.NUM_JOINTS * 2 : self.NUM_JOINTS * 2 + 3]
        obj_pos = obs[self.NUM_JOINTS * 2 + 3 : self.NUM_JOINTS * 2 + 6]

        js = [p.getJointState(self._kuka_uid, j, physicsClientId=cid) for j in range(self.NUM_JOINTS)]
        jvel = np.array([s[1] for s in js], dtype=np.float32)
        jtorque = np.array([s[3] for s in js], dtype=np.float32)

        reward, success, components = self._compute_reward(ee_pos, obj_pos, jvel, jtorque)

        info = {
            "success": success,
            "is_grasped": self._is_grasped,
            "stage": self._stage,
            "dist_obj_goal": float(np.linalg.norm(obj_pos - self._goal_pos)),
            "dist_ee_obj": float(np.linalg.norm(ee_pos - obj_pos)),
            "collision": components["r_collision"] < 0,
            **components,
        }

        terminated = bool(success)
        truncated = bool(self.steps >= self.max_steps)
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self._reset_grasp()
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
            self._physics_client = None
