"""
main.py — Assignment 3: Biped RL (1 m Platform Jump with SAC)

Usage examples
--------------
# View the environment (biped + stair in GUI, no model needed):
    python main.py --mode view

# Train SAC (timesteps set in utils.py):
    python main.py --mode train

# Train SAC for a custom number of steps:
    python main.py --mode train --timesteps 500000

# Evaluate the best saved checkpoint (10 episodes, headless):
    python main.py --mode test

# Evaluate with GUI rendering:
    python main.py --mode test --render --episodes 5

# Evaluate a specific model file:
    python main.py --mode test --model_path "models/sac_best/best_model"
"""

import argparse
import os
import time
import multiprocessing

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.env_util import make_vec_env

from utils import (
    BipedJumpEnv, RewardPlotCallback,
    TOTAL_TIMESTEPS, EVAL_FREQ,
    SAC_CONFIG,
    EVAL_EPISODES, ROBOT_MASS_KG,
)

# ── Algorithm registry ────────────────────────────────────────────────────────
# Task registry
TASK_ENV = {
    "jump": BipedJumpEnv,
}

# Register SAC algorithm with its config from utils.py.
ALGO_MAP = {
    "sac": {
        "cls": SAC,
        "config": SAC_CONFIG,
    }
}


# ── Environment Preview ────────────────────────────────────────────────────────
def view():
    """Spawns the biped + stair in GUI mode. Press Ctrl+C to quit."""
    import pybullet_data

    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8, physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)

    assets = os.path.join(os.path.dirname(__file__), "assets")
    if not os.path.isdir(assets):
        assets = os.path.join(os.path.dirname(__file__), "assest")
    p.loadURDF(os.path.join(assets, "biped_.urdf"), [0, 0, 0.81],
               useFixedBase=False, physicsClientId=cid)
    p.loadURDF(os.path.join(assets, "stair.urdf"),  [0, 2, 0],
               p.getQuaternionFromEuler([0, 0, -3.1416]),
               useFixedBase=True, physicsClientId=cid)

    print("[view] Biped + stair spawned. Press Ctrl+C to quit.")
    try:
        while True:
            p.stepSimulation(physicsClientId=cid)
            time.sleep(1 / 240)
    except KeyboardInterrupt:
        pass
    p.disconnect(cid)


# ── Training ──────────────────────────────────────────────────────────────────
def train(timesteps: int, render: bool = False, algo: str = "sac", task: str = "jump"):
    """
    Trains a SAC agent on the 1 m platform jump task and saves the model.

    Steps
    -----
    1. Create training and evaluation environments (wrapped in Monitor).
    2. Instantiate SAC with SAC_CONFIG from utils.py.
    3. Set up EvalCallback (saves best model) and RewardPlotCallback.
    4. Call model.learn() and handle KeyboardInterrupt for crash-saves.
    5. Save the final model and plot the reward curve.
    """
    env_cls = TASK_ENV[task]
    algo_cls = ALGO_MAP[algo]["cls"]
    algo_cfg = ALGO_MAP[algo]["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/sac_best", exist_ok=True)
    os.makedirs("logs/sac_goal", exist_ok=True)
    os.makedirs("logs/sac_eval", exist_ok=True)

    if render:
        n_envs = 1
        train_env = Monitor(env_cls(render=True), filename="logs/sac_monitor.csv")
    else:
        cpu_count = multiprocessing.cpu_count()
        n_envs = max(1, min(4, cpu_count // 2))
        train_env = make_vec_env(env_cls, n_envs=n_envs, env_kwargs={"render": False})
        train_env = VecMonitor(train_env, filename="logs/sac_monitor.csv")
    print(f"[train] Parallel envs: {n_envs}")

    eval_env = Monitor(env_cls(render=False))

    model = algo_cls(
        env=train_env,
        tensorboard_log="logs/sac_goal/",
        device=device,
        **algo_cfg,
    )

    reward_cb = RewardPlotCallback()
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/sac_best/",
        log_path="logs/sac_eval/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    try:
        model.learn(total_timesteps=timesteps, callback=[reward_cb, eval_cb])
    except KeyboardInterrupt:
        print("\n[train] Interrupted. Saving crash checkpoint...")
        model.save("models/sac_biped_crashsave")
    finally:
        model.save("models/sac_biped_goal")
        reward_cb.plot_rewards("reward_curve_sac.png")
        train_env.close()
        eval_env.close()


# ── Evaluation ────────────────────────────────────────────────────────────────
def test(model_path: str, episodes: int, render: bool, task: str = "jump"):
    """
    Loads a trained SAC model and evaluates it for a given number of episodes.

    Metrics reported per episode
    ----------------------------
    - Steps taken
    - Total reward
    - Energy consumed  (sum of |torque × velocity| × dt)
    - Distance travelled (Euclidean, spawn → landing)

    Summary metrics printed at the end
    -----------------------------------
    - Average reward
    - Fall rate  (%)
    - Average distance (m)
    - Average energy (J)
    - Cost of Transport (CoT) = Energy / (mass × g × distance)
    """
    DT = 1.0 / 50.0   # simulation timestep (must match utils.py)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Using device: {device}")

    env_cls = TASK_ENV[task]
    env = env_cls(render=render)
    model = SAC.load(model_path, env=env, device=device)

    joint_indices = env.get_joint_indices()
    total_energy, total_distance, total_reward, fall_count = 0.0, 0.0, 0.0, 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        is_fall = False

        ep_reward = 0.0
        ep_energy = 0.0
        steps = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            joint_states = p.getJointStates(env.robot_id, joint_indices, physicsClientId=env.physics_client)
            torques = np.array([s[3] for s in joint_states], dtype=np.float32)
            joint_vels = np.array([s[1] for s in joint_states], dtype=np.float32)
            ep_energy += float(np.sum(np.abs(torques * joint_vels)) * DT)

            obs, reward, terminated, truncated, info = env.step(action)
            is_fall = is_fall or bool(info.get("is_fall", False))
            ep_reward += reward
            steps += 1

        start_pos = np.array(env.robot_initial_position(), dtype=np.float32)
        end_pos = np.array(env.robot_current_position(), dtype=np.float32)
        ep_distance = float(np.linalg.norm(end_pos[:2] - start_pos[:2]))

        ep_fall = is_fall
        if ep_fall:
            fall_count += 1

        total_reward += ep_reward
        total_energy += ep_energy
        total_distance += ep_distance

        print(
            f"[test] Ep {ep:02d}/{episodes} | steps={steps:3d} | "
            f"reward={ep_reward:8.2f} | dist={ep_distance:6.3f} m | "
            f"energy={ep_energy:8.3f} J | fall={ep_fall}"
        )

    n = max(1, episodes)
    avg_reward = total_reward / n
    fall_rate = 100.0 * fall_count / n
    avg_distance = total_distance / n
    avg_energy = total_energy / n
    cot = total_energy / (ROBOT_MASS_KG * 9.81 * total_distance + 1e-8)

    print("\n=== Evaluation Summary ===")
    print(f"Average Reward   : {avg_reward:.3f}")
    print(f"Fall Rate (%)    : {fall_rate:.2f}")
    print(f"Average Distance : {avg_distance:.3f} m")
    print(f"Average Energy   : {avg_energy:.3f} J")
    print(f"Cost of Transport: {cot:.6f}")

    env.close()


# ── CLI entry-point ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Assignment 3 — Biped 1 m Platform Jump (SAC)"
    )
    parser.add_argument("--mode",       choices=["view", "train", "test"], required=True,
                        help="view: preview env  |  train: train SAC  |  test: evaluate")
    parser.add_argument("--algo",       choices=list(ALGO_MAP.keys()), default="sac",
                        help="RL algorithm")
    parser.add_argument("--task",       choices=list(TASK_ENV.keys()), default="jump",
                        help="Task/environment key")
    parser.add_argument("--timesteps",  type=int, default=None,
                        help="Override TOTAL_TIMESTEPS from utils.py")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a saved model (.zip) for --mode test")
    parser.add_argument("--episodes",   type=int, default=EVAL_EPISODES,
                        help=f"Evaluation episodes (default: {EVAL_EPISODES})")
    parser.add_argument("--render",     action="store_true",
                        help="Enable PyBullet GUI")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "view":
        view()

    else:
        if args.mode == "train":
            ts = args.timesteps if args.timesteps else TOTAL_TIMESTEPS
            train(ts, args.render, args.algo, args.task)
        elif args.mode == "test":
            model_path = args.model_path
            if model_path is None:
                model_path = "models/sac_best/best_model"
            test(model_path, args.episodes, args.render, args.task)


if __name__ == "__main__":
    main()
