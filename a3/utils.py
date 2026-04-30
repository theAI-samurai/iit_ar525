"""
utils.py — Shared utilities for Assignment 3: Biped 1 m Platform Jump.

Contains
--------
  - SAC_CONFIG          Hyperparameters for Soft Actor-Critic (YOU will tune these)
  - Training constants  TOTAL_TIMESTEPS, EVAL_FREQ, EVAL_EPISODES, ROBOT_MASS_KG
  - RewardPlotCallback  Records episode rewards and saves a plot after training
  - BipedJumpEnv        Gymnasium environment — provided, do not modify
"""

# ===========================================================================
# Hyperparameters  (edit these for Task 3)
# ===========================================================================

# ============================================================
# TODO: Set the total number of training timesteps (e.g. 1_000_000).
# ============================================================
TOTAL_TIMESTEPS = 1_000_000

# ============================================================
# TODO: Set how often (in steps) the evaluator runs during training (e.g. 10_000).
# ============================================================
EVAL_FREQ = 10_000

# ============================================================
# TODO: Set the max steps per episode — must match BipedJumpEnv.max_steps (600).
# ============================================================
MAX_EPISODE_STEPS = 600

# ---------------------------------------------------------------------------
# SAC  (Soft Actor-Critic) — the only algorithm used in this assignment
# ---------------------------------------------------------------------------
# ============================================================
# TODO: Fill in SAC_CONFIG with your chosen hyperparameters.
#       Required keys: policy, learning_rate, buffer_size, batch_size,
#                      tau, gamma, ent_coef, verbose.
# ============================================================
SAC_CONFIG = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    buffer_size   = 1_000_000,
    batch_size    = 512,
    tau           = 0.005,
    gamma         = 0.99,
    ent_coef      = "auto",
    train_freq    = (4, "step"),
    gradient_steps= 4,
    learning_starts = 2_000,
    policy_kwargs = dict(net_arch=[256, 256, 256]),
    verbose       = 1,
)

# ---------------------------------------------------------------------------
# Evaluation / metric settings  (do not change)
# ---------------------------------------------------------------------------
EVAL_EPISODES = 10
ROBOT_MASS_KG = 2.05   # used to compute Cost of Transport (CoT)


# ===========================================================================
# RewardPlotCallback
# ===========================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless training
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class RewardPlotCallback(BaseCallback):
    """Records episode rewards during training and saves a plot at the end."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [0.0]), dtype=np.float32)
        dones = np.asarray(self.locals.get("dones", [False]))

        self._current_episode_reward += float(np.mean(rewards))
        if bool(np.any(dones)):
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True   # returning False would stop training

    def plot_rewards(self, save_path="reward_curve_sac.png"):
        if not self.episode_rewards:
            print("No episode rewards recorded yet.")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, alpha=0.6, label="Episode Reward")

        window = 20
        if len(self.episode_rewards) >= window:
            rolling = [
                sum(self.episode_rewards[max(0, i - window):i]) / min(i, window)
                for i in range(1, len(self.episode_rewards) + 1)
            ]
            plt.plot(rolling, color="red", linewidth=2, label=f"{window}-ep Rolling Avg")

        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("SAC Training Reward Curve — Biped 1 m Jump")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Reward plot saved to {save_path}")


# ===========================================================================
# BipedJumpEnv  — provided environment, do not modify
# ===========================================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
if not os.path.isdir(_ASSET_DIR):
    _ASSET_DIR = os.path.join(os.path.dirname(__file__), "assest")


class BipedJumpEnv(gym.Env):
    """
    Task: the biped robot spawns on top of a 1 m tall platform and must
    jump off, then land upright on the ground below.

    Phases
    ------
    1. On platform  
    2. In flight    
    3. Landing      

   
    """

    PLATFORM_H = 1.0          # top surface height (m)
    SPAWN_Z    = 1.0 + 0.81   # robot COM at spawn  (platform top + standing height)
    GROUND_Z   = 0.81         # robot COM when standing on flat ground

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        cid = p.connect(p.GUI if render else p.DIRECT)
        self.physics_client = cid

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        self.timestep = 1.0 / 50.0
        p.setTimeStep(self.timestep, physicsClientId=cid)

        self.max_steps         = MAX_EPISODE_STEPS
        self.step_counter      = 0
        self.land_stable_steps = 0

        # Ground plane
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, physicsClientId=cid)

        # stair.urdf step is at local xyz="0 1.6 0.29", size "5 1 0.08".
        # With 180-deg Z rotation: world_step_centre = (urdf_x, urdf_y-1.6, urdf_z+0.29).
        # To land step-top at z=1.0 directly under robot (world y=0):
        #   urdf_y = 1.6,  urdf_z = 0.67  →  step top = 0.67+0.29+0.04 = 1.00 m
        self.platform_id = p.loadURDF(
            os.path.join(_ASSET_DIR, "stair.urdf"),
            [0, 1.6, 0.67],
            p.getQuaternionFromEuler([0, 0, -3.1416]),
            useFixedBase=True,
            physicsClientId=cid,
        )

        # Robot
        urdf_path = os.path.join(_ASSET_DIR, "biped_.urdf")
        self.robot_id = p.loadURDF(urdf_path, [0, 0, self.SPAWN_Z],
                                    useFixedBase=False, physicsClientId=cid)
        p.changeDynamics(self.robot_id, -1,
                         linearDamping=0.5, angularDamping=0.5,
                         physicsClientId=cid)

        # Joint discovery
        self.joint_indices   = []
        self.joint_limits    = []
        self.left_foot_link  = 2
        self.right_foot_link = 5

        for i in range(p.getNumJoints(self.robot_id, physicsClientId=cid)):
            ji = p.getJointInfo(self.robot_id, i, physicsClientId=cid)
            if ji[2] == p.JOINT_REVOLUTE:
                self.joint_indices.append(i)
                self.joint_limits.append((ji[8], ji[9]))
            if b"left_foot"  in ji[12]: self.left_foot_link  = i
            if b"right_foot" in ji[12]: self.right_foot_link = i

        p.changeDynamics(self.robot_id, self.left_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)
        p.changeDynamics(self.robot_id, self.right_foot_link,
                         lateralFriction=2.0, physicsClientId=cid)

        self.n_actuated = len(self.joint_indices)

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.n_actuated,), dtype=np.float32)
        obs_dim  = self.n_actuated * 2 + 3 + 3 + 3 + 2 + 1 + 1
        obs_high = np.full(obs_dim, np.finfo(np.float32).max, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.prev_z     = self.SPAWN_Z
        self.has_landed = False
        self._initial_base_pos = np.array([0.0, 0.0, self.SPAWN_Z], dtype=np.float32)
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.reached_floor = False
        self.floor_stable_steps = 0
        self.upright_floor_steps = 0
        self.edge_control_steps = 0
        self.just_landed = False
        self.steps_since_landing = 0
        self.floor_entry_y = 0.0
        self.prev_action = np.zeros(self.n_actuated, dtype=np.float32)
        self.platform_edge_y = 0.48
        self.edge_brace_y = 0.12
        self.landing_recovery_steps = 45
        self.edge_brace_action = self._action_from_joint_targets(
            [0.05, -0.75, 0.65, 0.05, -0.75, 0.65]
        )
        self.edge_step_action = self._action_from_joint_targets(
            [0.18, -0.55, 0.42, 0.18, -0.55, 0.42]
        )
        self.landing_brace_action = self._action_from_joint_targets(
            [0.08, -0.70, 0.62, 0.08, -0.70, 0.62]
        )
        self.standing_action = self._action_from_joint_targets(
            [0.08, -0.08, 0.02, 0.08, -0.08, 0.02]
        )
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, self.SPAWN_Z],
            [0, 0, 0, 1],
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.robot_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.physics_client,
        )

        for j in self.joint_indices:
            p.resetJointState(
                self.robot_id,
                j,
                targetValue=0.0,
                targetVelocity=0.0,
                physicsClientId=self.physics_client,
            )

        self.step_counter = 0
        self.land_stable_steps = 0
        self.prev_z = self.SPAWN_Z
        self.has_landed = False
        self._initial_base_pos = np.array(self.robot_current_position(), dtype=np.float32)
        self.prev_x = float(self._initial_base_pos[0])
        self.prev_y = float(self._initial_base_pos[1])
        self.reached_floor = False
        self.floor_stable_steps = 0
        self.upright_floor_steps = 0
        self.edge_control_steps = 0
        self.just_landed = False
        self.steps_since_landing = 0
        self.floor_entry_y = float(self._initial_base_pos[1])
        self.prev_action = np.zeros(self.n_actuated, dtype=np.float32)

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    def _action_from_joint_targets(self, targets):
        """Convert joint target angles to the normalized action range."""
        targets = np.asarray(targets[:self.n_actuated], dtype=np.float32)
        action = np.zeros(self.n_actuated, dtype=np.float32)
        for i, target in enumerate(targets):
            lo, hi = self.joint_limits[i]
            if hi <= lo:
                continue
            action[i] = (2.0 * (float(target) - lo) / (hi - lo)) - 1.0
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    def _get_obs(self):
        joint_states = p.getJointStates(
            self.robot_id,
            self.joint_indices,
            physicsClientId=self.physics_client,
        )
        joint_pos = np.array([s[0] for s in joint_states], dtype=np.float32)
        joint_vel = np.array([s[1] for s in joint_states], dtype=np.float32)

        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        euler = p.getEulerFromQuaternion(orn)

        left_contact = float(
            len(
                p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.plane_id,
                    linkIndexA=self.left_foot_link,
                    physicsClientId=self.physics_client,
                )
            )
            > 0
        )
        right_contact = float(
            len(
                p.getContactPoints(
                    bodyA=self.robot_id,
                    bodyB=self.plane_id,
                    linkIndexA=self.right_foot_link,
                    physicsClientId=self.physics_client,
                )
            )
            > 0
        )

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                np.array(pos, dtype=np.float32),
                np.array(euler, dtype=np.float32),
                np.array(lin_vel, dtype=np.float32),
                np.array([left_contact, right_contact], dtype=np.float32),
                np.array([self.step_counter / self.max_steps], dtype=np.float32),
                np.array([float(self.has_landed)], dtype=np.float32),
            ]
        ).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    def _compute_reward(self, pos, orn, lin_vel, left_contact, right_contact):
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        # Upright penalty (roll/pitch only — NOT applied to lateral lin_vel).
        tilt_penalty = 1.2 * (abs(roll) + abs(pitch))
        # Penalise sideways drift (X axis), never Y (forward).
        side_penalty = 0.3 * abs(float(lin_vel[0]))

        on_platform = pos[2] > (self.GROUND_Z + 0.55)
        near_floor  = pos[2] < (self.GROUND_Z + 0.18)

        fwd_vel = float(lin_vel[1])      # +Y is forward

        # Phase 1 — on platform: reward forward velocity strongly.
        if not self.reached_floor and on_platform:
            y_to_edge = self.platform_edge_y - float(pos[1])
            if y_to_edge > 0.16:
                reward = 7.0 * fwd_vel
            else:
                reward = 0.5 * max(0.0, fwd_vel)
                reward += 4.0 * max(0.0, 0.35 - (abs(roll) + abs(pitch)))
                reward -= 4.0 * max(0.0, abs(fwd_vel) - 0.20)
                reward -= 5.0 * max(0.0, (abs(roll) + abs(pitch)) - 0.25)
            reward += 0.05 if (left_contact and right_contact) else -0.05

        # Phase 2 — in descent from platform to floor.
        elif not self.reached_floor and (not on_platform):
            vz = float(lin_vel[2])
            descend_progress = max(0.0, float(self.prev_z - pos[2]))
            reward = 4.0 * descend_progress   # reward descent progress strongly
            reward += 1.0 * max(0.0, -vz)    # encourage downward motion
            reward += 3.0 * fwd_vel           # forward motion, but avoid over-dive at edge
            if near_floor and (left_contact or right_contact):
                reward += 0.5
                reward -= 1.4 * max(0.0, abs(vz) - 1.0)   # penalize hard touchdown speed
                reward += 0.6 * max(0.0, 0.32 - (abs(roll) + abs(pitch)))

        # Phase 3 — on floor: walk, higher forward velocity scale.
        else:
            post_land_steps = max(0, self.steps_since_landing)
            y_step_progress = float(pos[1] - self.prev_y)
            post_land_distance = float(pos[1] - self.floor_entry_y)
            if post_land_steps < self.landing_recovery_steps:
                reward = 1.5 * max(0.0, fwd_vel)
                reward += 2.0 * max(0.0, 0.45 - (abs(roll) + abs(pitch)))
                reward -= 1.0 * max(0.0, abs(fwd_vel) - 0.55)
            elif post_land_steps < self.landing_recovery_steps + 20:
                reward = 6.0 * fwd_vel
            else:
                reward = 12.0 * fwd_vel
            progress_scale = 6.0 if post_land_steps < self.landing_recovery_steps else 22.0
            reward += progress_scale * y_step_progress
            reward += 0.12 * float(left_contact) + 0.12 * float(right_contact)
            if self.floor_stable_steps >= 10:
                reward += 0.2
            if post_land_steps < self.landing_recovery_steps and (left_contact or right_contact):
                reward += 0.8 * max(0.0, 0.35 - (abs(roll) + abs(pitch)))
            if left_contact and right_contact:
                reward += 0.4 * max(0.0, 0.45 - (abs(roll) + abs(pitch)))
            if post_land_steps >= 25 and fwd_vel < 0.05:
                reward -= 0.8
            if post_land_steps >= 60 and post_land_distance < 0.12:
                reward -= 1.2

        if self.just_landed:
            reward += 8.0

        reward -= tilt_penalty + side_penalty
        return reward

    # ------------------------------------------------------------------
    def get_joint_indices(self):
        return list(self.joint_indices)

    def robot_initial_position(self):
        return tuple(self._initial_base_pos)

    def robot_current_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        return tuple(pos)

    def _damp_touchdown_velocity(self, orn, lin_vel, ang_vel, contact_count):
        if contact_count <= 0:
            return

        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        lin = np.array(lin_vel, dtype=np.float32)
        ang = np.array(ang_vel, dtype=np.float32)

        contact_scale = 0.35 if contact_count == 1 else 0.22
        lin[0] *= 0.35
        lin[1] *= 0.35
        if lin[2] < 0.0:
            lin[2] *= 0.12

        ang[0] = (ang[0] * contact_scale) - (3.2 * roll)
        ang[1] = (ang[1] * contact_scale) - (3.2 * pitch)
        ang[2] *= 0.35

        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=lin.tolist(),
            angularVelocity=ang.tolist(),
            physicsClientId=self.physics_client,
        )

    def _damp_descent_velocity(self, orn, lin_vel, ang_vel):
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        lin = np.array(lin_vel, dtype=np.float32)
        ang = np.array(ang_vel, dtype=np.float32)

        lin[0] *= 0.55
        lin[1] *= 0.45
        if lin[2] < -0.75:
            lin[2] = -0.75
        ang[0] = (0.28 * ang[0]) - (2.4 * roll)
        ang[1] = (0.28 * ang[1]) - (2.4 * pitch)
        ang[2] *= 0.50

        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=lin.tolist(),
            angularVelocity=ang.tolist(),
            physicsClientId=self.physics_client,
        )

    def _damp_edge_velocity(self, orn, lin_vel, ang_vel, forward_cap=0.12, forward_min=None):
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        lin = np.array(lin_vel, dtype=np.float32)
        ang = np.array(ang_vel, dtype=np.float32)

        lin[0] *= 0.45
        lin[1] = min(float(lin[1]), forward_cap)
        if forward_min is not None:
            lin[1] = max(float(lin[1]), forward_min)
        if lin[2] < 0.0:
            lin[2] *= 0.80
        ang[0] = (0.18 * ang[0]) - (3.8 * roll)
        ang[1] = (0.22 * ang[1]) - (3.0 * pitch)
        ang[2] *= 0.45

        p.resetBaseVelocity(
            self.robot_id,
            linearVelocity=lin.tolist(),
            angularVelocity=ang.tolist(),
            physicsClientId=self.physics_client,
        )

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.robot_id,
            physicsClientId=self.physics_client,
        )
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        on_platform_ctrl = base_pos[2] > (self.GROUND_Z + 0.55)
        edge_phase = (
            (not self.reached_floor)
            and on_platform_ctrl
            and base_pos[1] > self.edge_brace_y
        )
        near_floor_ctrl = base_pos[2] < (self.GROUND_Z + 0.36)
        descending_fast = lin_vel[2] < -0.6
        landing_recovery_ctrl = (
            self.reached_floor
            and self.steps_since_landing < self.landing_recovery_steps
        )
        landing_phase = (
            landing_recovery_ctrl
            or near_floor_ctrl
            or (base_pos[2] < (self.PLATFORM_H + 0.35) and descending_fast)
        )
        if edge_phase:
            self.edge_control_steps += 1
        elif self.reached_floor or base_pos[1] < self.edge_brace_y:
            self.edge_control_steps = 0

        control_force = 78.0 if landing_phase else (95.0 if edge_phase else 35.0)
        action_alpha = 0.18 if landing_phase else (0.12 if edge_phase else 0.45)
        smoothed_action = (action_alpha * action) + ((1.0 - action_alpha) * self.prev_action)

        if landing_phase:
            if not self.reached_floor:
                smoothed_action = self.landing_brace_action.copy()
                self._damp_descent_velocity(base_orn, lin_vel, ang_vel)
            elif landing_recovery_ctrl:
                # Hold the crouch after impact, then stand up slowly.
                recovery_progress = min(1.0, self.steps_since_landing / max(1, self.landing_recovery_steps))
                brace_action = (
                    (1.0 - recovery_progress) * self.landing_brace_action
                    + recovery_progress * self.standing_action
                )
                smoothed_action = (0.30 * smoothed_action) + (0.70 * brace_action)
        elif edge_phase:
            if self.edge_control_steps < 25:
                smoothed_action = self.edge_brace_action.copy()
                self._damp_edge_velocity(base_orn, lin_vel, ang_vel, forward_cap=0.12)
            else:
                smoothed_action = self.edge_step_action.copy()
                self._damp_edge_velocity(
                    base_orn,
                    lin_vel,
                    ang_vel,
                    forward_cap=0.35,
                    forward_min=0.20,
                )

        self.prev_action = smoothed_action.astype(np.float32)

        for i, j in enumerate(self.joint_indices):
            lo, hi = self.joint_limits[i]
            target = lo + 0.5 * (smoothed_action[i] + 1.0) * (hi - lo)
            p.setJointMotorControl2(
                self.robot_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=control_force,
                physicsClientId=self.physics_client,
            )

        p.stepSimulation(physicsClientId=self.physics_client)
        if self.render_mode:
            time.sleep(self.timestep)

        self.step_counter += 1

        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self.physics_client)
        post_step_edge_phase = (
            (not self.reached_floor)
            and pos[2] > (self.GROUND_Z + 0.42)
            and pos[1] > self.edge_brace_y
        )
        if post_step_edge_phase:
            if self.edge_control_steps < 25:
                self._damp_edge_velocity(orn, lin_vel, ang_vel, forward_cap=0.12)
            else:
                self._damp_edge_velocity(
                    orn,
                    lin_vel,
                    ang_vel,
                    forward_cap=0.35,
                    forward_min=0.20,
                )
            pos, orn = p.getBasePositionAndOrientation(
                self.robot_id,
                physicsClientId=self.physics_client,
            )
            lin_vel, ang_vel = p.getBaseVelocity(
                self.robot_id,
                physicsClientId=self.physics_client,
            )

        post_step_landing_phase = (
            (not self.reached_floor)
            and pos[2] < (self.GROUND_Z + 0.55)
            and lin_vel[2] < -0.05
        )
        if post_step_landing_phase:
            self._damp_descent_velocity(orn, lin_vel, ang_vel)
            pos, orn = p.getBasePositionAndOrientation(
                self.robot_id,
                physicsClientId=self.physics_client,
            )
            lin_vel, ang_vel = p.getBaseVelocity(
                self.robot_id,
                physicsClientId=self.physics_client,
            )

        left_contact = len(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=self.left_foot_link,
                physicsClientId=self.physics_client,
            )
        ) > 0
        right_contact = len(
            p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=self.plane_id,
                linkIndexA=self.right_foot_link,
                physicsClientId=self.physics_client,
            )
        ) > 0

        near_floor = pos[2] < (self.GROUND_Z + 0.18)
        self.just_landed = False
        both_feet_contact = left_contact and right_contact
        contact_count = int(left_contact) + int(right_contact)
        if near_floor and contact_count > 0:
            self._damp_touchdown_velocity(orn, lin_vel, ang_vel, contact_count)
            pos, orn = p.getBasePositionAndOrientation(
                self.robot_id,
                physicsClientId=self.physics_client,
            )
            lin_vel, _ = p.getBaseVelocity(
                self.robot_id,
                physicsClientId=self.physics_client,
            )

        if near_floor and both_feet_contact:
            self.floor_stable_steps += 1
            if (not self.reached_floor) and self.floor_stable_steps >= 4:
                self.reached_floor = True
                self.has_landed = True
                self.just_landed = True
                self.steps_since_landing = 0
                self.floor_entry_y = float(pos[1])
        elif near_floor and (left_contact or right_contact):
            self.floor_stable_steps = max(0, self.floor_stable_steps - 1)
        else:
            self.floor_stable_steps = 0

        if self.reached_floor:
            self.steps_since_landing += 1

        reward = self._compute_reward(pos, orn, lin_vel, left_contact, right_contact)
        self.just_landed = False

        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        landing_recovery = self.reached_floor and (self.steps_since_landing < self.landing_recovery_steps)
        tilt_limit = 1.45 if landing_recovery else 1.1
        fallen = abs(roll) > tilt_limit or abs(pitch) > tilt_limit or pos[2] < 0.40
        if fallen:
            reward -= 60.0

        upright_on_floor = (
            self.reached_floor
            and contact_count > 0
            and abs(roll) < 0.80
            and abs(pitch) < 0.80
            and pos[2] > 0.52
        )
        if upright_on_floor:
            self.upright_floor_steps += 1
        else:
            self.upright_floor_steps = 0

        stable_landing = (
            self.reached_floor
            and self.upright_floor_steps >= 12
            and not fallen
        )
        if stable_landing:
            reward += 50.0

        terminated = bool(fallen)
        truncated = bool(self.step_counter >= self.max_steps)

        self.prev_z = pos[2]
        self.prev_x = float(pos[0])
        self.prev_y = float(pos[1])
        obs = self._get_obs()
        info = {
            "is_fall": bool(fallen),
            "is_success": bool((not fallen) and (stable_landing or (truncated and self.reached_floor))),
            "has_landed": bool(self.has_landed),
            "reached_floor": bool(self.reached_floor),
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def close(self):
        p.disconnect(self.physics_client)
