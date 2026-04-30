
"""
12-DOF Robot Arm RL Training & Evaluation
Algorithms: PPO, TD3, SAC, TRPO
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import TRPO
    HAS_TRPO = True
except ImportError:
    HAS_TRPO = False
    print("[WARNING] sb3-contrib not installed – TRPO will be skipped.")
    print("          Install with: pip install sb3-contrib")

from utils import RobotArmEnv
from kuka_pick_place_env import KukaPickPlaceEnv

MODELS_DIR  = "./models"
RESULTS_DIR = "./results"
LOGS_DIR    = "./logs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    _gpu = torch.cuda.get_device_name(0)
    _vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] {_gpu}  |  VRAM: {_vram:.1f} GB  |  device=cuda")
else:
    print("[GPU] CUDA not available – training on CPU")


# ──────────────────────────────────────────────────────────────────────────────
# Callback: capture episode rewards & metrics during training
# ──────────────────────────────────────────────────────────────────────────────

class TrainingLogger(BaseCallback):
    """Logs mean episode reward, length, success rate, collision rate."""

    def __init__(self, log_freq: int = 2000):
        super().__init__()
        self.log_freq = log_freq
        self.timesteps: list[int]  = []
        self.mean_rewards: list[float] = []
        self.mean_lengths: list[float] = []
        self.success_rates: list[float] = []
        self.collision_rates: list[float] = []
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[float] = []
        self._ep_success: list[float] = []
        self._ep_collision: list[float] = []

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [])
        dones   = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, done in enumerate(dones):
            if done and infos and i < len(infos):
                ep_info = infos[i].get("episode", {})
                if ep_info:
                    self._ep_rewards.append(ep_info["r"])
                    self._ep_lengths.append(ep_info["l"])
                info = infos[i]
                self._ep_success.append(float(info.get("success", False)))
                self._ep_collision.append(float(info.get("collision", False)))

        if self.num_timesteps % self.log_freq == 0 and self._ep_rewards:
            self.timesteps.append(self.num_timesteps)
            self.mean_rewards.append(np.mean(self._ep_rewards[-50:]))
            self.mean_lengths.append(np.mean(self._ep_lengths[-50:]))
            self.success_rates.append(np.mean(self._ep_success[-50:]))
            self.collision_rates.append(np.mean(self._ep_collision[-50:]))

        return True


# ──────────────────────────────────────────────────────────────────────────────
# Make environment factory
# ──────────────────────────────────────────────────────────────────────────────

def make_env(
    render_mode=None,
    max_steps=500,
    render_fps=20.0,
    env_name="arm12dof",
    strict_curriculum: bool = False,
    fixed_phase_episodes: int = 300,
):
    def _init():
        if env_name == "arm12dof":
            env = RobotArmEnv(
                render_mode=render_mode,
                max_steps=max_steps,
                render_fps=render_fps,
            )
        elif env_name == "kuka":
            env = KukaPickPlaceEnv(
                render_mode=render_mode,
                max_steps=max_steps,
                strict_two_phase_curriculum=strict_curriculum,
                fixed_phase_episodes=fixed_phase_episodes,
            )
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        return Monitor(env)
    return _init


# ──────────────────────────────────────────────────────────────────────────────
# Build algorithm models
# ──────────────────────────────────────────────────────────────────────────────

def build_model(algo_name: str, env, tensorboard_log: str | None = None,
                device: str = DEVICE, fast_pickup: bool = False):
    """Return an SB3 model for the given algorithm name.

    fast_pickup=True applies hyperparameters tuned for the Kuka pick-and-place
    task: higher learning rate, smaller buffer, early learning start, faster
    target updates.  Only meaningful for SAC/TD3.
    """
    algo_name = algo_name.upper()

    if algo_name == "PPO":
        return PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            device=device,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    if algo_name == "TD3":
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        return TD3(
            "MlpPolicy", env,
            learning_rate =1e-3  if fast_pickup else 3e-4,
            buffer_size   =200_000 if fast_pickup else 1_000_000,
            batch_size    =256   if fast_pickup else 1024,
            tau           =0.02  if fast_pickup else 0.005,
            gamma         =0.98  if fast_pickup else 0.99,
            learning_starts=500  if fast_pickup else 100,
            train_freq    =1,
            gradient_steps=2     if fast_pickup else 1,
            action_noise=noise,
            policy_kwargs=dict(net_arch=[256, 256]),
            device=device,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    if algo_name == "SAC":
        return SAC(
            "MlpPolicy", env,
            learning_rate =1e-3  if fast_pickup else 3e-4,
            buffer_size   =200_000 if fast_pickup else 1_000_000,
            batch_size    =256   if fast_pickup else 1024,
            tau           =0.02  if fast_pickup else 0.005,
            gamma         =0.98  if fast_pickup else 0.99,
            learning_starts=500  if fast_pickup else 100,
            train_freq    =1,
            gradient_steps=2     if fast_pickup else 1,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]),
            device=device,
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    if algo_name == "TRPO":
        if not HAS_TRPO:
            raise RuntimeError("Install sb3-contrib to use TRPO: pip install sb3-contrib")
        # TRPO with MLP policies is typically faster/more stable on CPU in SB3.
        return TRPO(
            "MlpPolicy", env,
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=1024,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=0.01,
            cg_max_steps=15,
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu",
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

    raise ValueError(f"Unknown algorithm: {algo_name}")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    algo_name: str,
    total_timesteps: int = 200_000,
    render: bool = False,
    n_envs: int = 1,
    render_fps: float = 20.0,
    save_dir: str = MODELS_DIR,
    env_name: str = "arm12dof",
    resume: bool = False,
    strict_curriculum: bool = False,
    fixed_phase_episodes: int = 300,
) -> tuple[Any, TrainingLogger]:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOGS_DIR, "tensorboard"), exist_ok=True)

    render_mode = "human" if render else None

    # PyBullet's C extension is not subprocess-safe (forked processes lose the
    # physics server). DummyVecEnv runs every env in the same process where
    # each env gets its own p.connect() client ID – fully supported by PyBullet.
    n = 1 if render else n_envs
    env_max_steps = 400 if env_name == "kuka" else 500
    env = DummyVecEnv([
        make_env(
            render_mode=render_mode if i == 0 else None,
            max_steps=env_max_steps,
            render_fps=render_fps,
            env_name=env_name,
            strict_curriculum=strict_curriculum,
            fixed_phase_episodes=fixed_phase_episodes,
        )
        for i in range(n)
    ])

    tb_log = os.path.join(LOGS_DIR, "tensorboard")
    model_path = os.path.join(save_dir, f"{algo_name}_final.zip")

    if resume and os.path.exists(model_path):
        cls_map = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
        if HAS_TRPO:
            cls_map["TRPO"] = TRPO
        model = cls_map[algo_name].load(model_path, env=env, device=DEVICE)
        print(f"  Resuming from checkpoint: {model_path}")
    else:
        model = build_model(algo_name, env, tensorboard_log=tb_log,
                            fast_pickup=(env_name == "kuka"))

    logger  = TrainingLogger(log_freq=max(1000, total_timesteps // 100))

    print(f"\n{'='*60}")
    print(f"  Training {algo_name}  |  timesteps={total_timesteps:,}"
          f"  |  envs={1 if render else n_envs}  |  device={DEVICE}")
    print(f"{'='*60}")
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=logger,
        progress_bar=True,
        reset_num_timesteps=not (resume and os.path.exists(model_path)),
    )
    elapsed = time.time() - t0

    path = os.path.join(save_dir, f"{algo_name}_final")
    model.save(path)
    print(f"\n  Saved → {path}.zip   (training time: {elapsed:.1f}s)")

    env.close()
    return model, logger


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    model,
    algo_name: str,
    num_episodes: int = 50,
    render: bool = False,
    render_fps: float = 20.0,
    env_name: str = "arm12dof",
    strict_curriculum: bool = False,
    fixed_phase_episodes: int = 300,
) -> dict[str, float]:
    render_mode = "human" if render else None
    if env_name == "arm12dof":
        env = Monitor(RobotArmEnv(render_mode=render_mode, render_fps=render_fps))
    elif env_name == "kuka":
        env = Monitor(
            KukaPickPlaceEnv(
                render_mode=render_mode,
                max_steps=400,
                strict_two_phase_curriculum=strict_curriculum,
                fixed_phase_episodes=fixed_phase_episodes,
            )
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    successes, collisions, ep_lengths, ep_rewards, obs_dists = [], [], [], [], []
    grasped_eps, lifted_eps = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_r, steps = 0.0, 0
        min_obs_dist = float("inf")
        min_goal_dist = float("inf")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_r     += reward
            steps    += 1
            if env_name == "kuka":
                min_goal_dist = min(min_goal_dist, float(info.get("dist_obj_goal", float("inf"))))
            else:
                min_obs_dist = min(min_obs_dist, info.get("min_obs_dist", float("inf")))

        successes.append(float(info.get("success", False)))
        collisions.append(float(info.get("collision", False)))
        if env_name == "kuka":
            grasped_eps.append(float(info.get("stage", 0) >= 2))
            lifted_eps.append(float(info.get("stage", 0) >= 3))
        ep_lengths.append(steps)
        ep_rewards.append(ep_r)
        if env_name == "kuka":
            # Convert min goal distance to a higher-is-better score in [0, 1].
            obs_dists.append(1.0 / (1.0 + min_goal_dist))
        else:
            obs_dists.append(min_obs_dist)

        if (ep + 1) % 10 == 0:
            print(f"  [{algo_name}] ep {ep+1}/{num_episodes}  "
                  f"reward={ep_r:.1f}  success={info.get('success')}")

    env.close()

    metrics = {
        "success_rate":    np.mean(successes),
        "collision_rate":  np.mean(collisions),
        "avg_length":      np.mean(ep_lengths),
        "avg_reward":      np.mean(ep_rewards),
        "min_obs_dist":    np.mean(obs_dists),
    }
    if env_name == "kuka":
        metrics["grasp_rate"] = np.mean(grasped_eps) if grasped_eps else 0.0
        metrics["lift_rate"] = np.mean(lifted_eps) if lifted_eps else 0.0

    print(f"\n  ── {algo_name} Evaluation ({num_episodes} episodes) ──")
    print(f"  Success rate   : {metrics['success_rate']:.2%}")
    print(f"  Collision rate : {metrics['collision_rate']:.2%}")
    print(f"  Avg ep length  : {metrics['avg_length']:.1f}")
    print(f"  Avg reward     : {metrics['avg_reward']:.2f}")
    if env_name == "kuka":
        print(f"  Avg goal proximity: {metrics['min_obs_dist']:.3f} (higher is better)")
        print(f"  Grasp rate     : {metrics['grasp_rate']:.2%}")
        print(f"  Lift rate      : {metrics['lift_rate']:.2%}")
    else:
        print(f"  Avg min obs dist: {metrics['min_obs_dist']:.3f} m")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "PPO":  "#2196F3",
    "TD3":  "#F44336",
    "SAC":  "#4CAF50",
    "TRPO": "#FF9800",
}


def _smooth(arr, w=5):
    if len(arr) < w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def plot_training_curves(loggers: dict[str, TrainingLogger], out_dir: str, env_name: str = "arm12dof"):
    os.makedirs(out_dir, exist_ok=True)
    task_label = "KUKA Pick-and-Place" if env_name == "kuka" else "12-DOF Robot Arm"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves – {task_label}", fontsize=14, fontweight="bold")

    metrics = [
        ("mean_rewards",    "Mean Episode Reward",     axes[0, 0]),
        ("mean_lengths",    "Mean Episode Length",     axes[0, 1]),
        ("success_rates",   "Success Rate",            axes[1, 0]),
        ("collision_rates", "Collision Rate",          axes[1, 1]),
    ]

    for attr, ylabel, ax in metrics:
        for algo, lg in loggers.items():
            ts   = getattr(lg, "timesteps")
            vals = getattr(lg, attr)
            if not ts:
                continue
            color = COLORS.get(algo, "black")
            ax.plot(ts, _smooth(vals), label=algo, color=color, linewidth=2)
            ax.fill_between(ts, _smooth(vals), alpha=0.1, color=color)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {path}")


def plot_performance_comparison(results: dict[str, dict], out_dir: str, env_name: str = "arm12dof"):
    os.makedirs(out_dir, exist_ok=True)
    task_label = "KUKA Pick-and-Place" if env_name == "kuka" else "12-DOF Robot Arm"
    algos   = list(results.keys())
    colors  = [COLORS.get(a, "steelblue") for a in algos]

    metrics = [
        ("success_rate",   "Success Rate",         True),
        ("collision_rate", "Collision Rate",        True),
        ("avg_reward",     "Average Episode Reward", False),
        ("avg_length",     "Average Episode Length", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Performance Comparison – {task_label}", fontsize=14, fontweight="bold")

    for (key, title, as_pct), ax in zip(metrics, axes.flatten()):
        vals = [results[a][key] for a in algos]
        bars = ax.bar(algos, vals, color=colors, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, vals):
            label = f"{val:.1%}" if as_pct else f"{val:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(abs(v) for v in vals) * 0.01,
                    label, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(title)
        ax.set_ylabel("Rate" if as_pct else title)
        ax.grid(True, axis="y", alpha=0.3)
        if as_pct:
            ax.set_ylim(0, 1.15)

    plt.tight_layout()
    path = os.path.join(out_dir, "performance_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved performance comparison → {path}")


def plot_radar(results: dict[str, dict], out_dir: str):
    """Radar / spider chart comparing algorithms across normalised metrics."""
    os.makedirs(out_dir, exist_ok=True)

    metric_keys   = ["success_rate", "collision_rate", "avg_reward", "avg_length", "min_obs_dist"]
    metric_labels = ["Success\nRate", "Collision\nRate (↓)", "Avg\nReward (↑)",
                     "Ep Length (↓)", "Min Obs\nDist (↑)"]
    N = len(metric_keys)

    # Normalise each metric to [0, 1]; invert lower-is-better metrics
    invert = {k: False for k in metric_keys}
    invert["collision_rate"] = True
    invert["avg_length"]     = True

    all_vals = {k: [results[a][k] for a in results] for k in metric_keys}
    ranges   = {k: (min(all_vals[k]), max(all_vals[k])) for k in metric_keys}

    def normalise(k, v):
        lo, hi = ranges[k]
        if hi == lo:
            return 0.5
        n = (v - lo) / (hi - lo)
        return 1 - n if invert[k] else n

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for algo, res in results.items():
        vals = [normalise(k, res[k]) for k in metric_keys]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=algo, color=COLORS.get(algo, "black"))
        ax.fill(angles, vals, alpha=0.1, color=COLORS.get(algo, "black"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Algorithm Comparison (normalised)", pad=20, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    path = os.path.join(out_dir, "radar_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved radar chart → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(results: dict[str, dict], out_dir: str, env_name: str = "arm12dof"):
    os.makedirs(out_dir, exist_ok=True)
    task_label = "KUKA PICK-AND-PLACE" if env_name == "kuka" else "12-DOF ROBOT ARM"

    # CSV
    csv_path = os.path.join(out_dir, "comparison_report.csv")
    fieldnames = ["Algorithm", "Success Rate", "Collision Rate",
                  "Avg Episode Length", "Avg Reward", "Avg Min Obs Dist (m)"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for algo, m in results.items():
            writer.writerow({
                "Algorithm":           algo,
                "Success Rate":        f"{m['success_rate']:.4f}",
                "Collision Rate":      f"{m['collision_rate']:.4f}",
                "Avg Episode Length":  f"{m['avg_length']:.2f}",
                "Avg Reward":          f"{m['avg_reward']:.4f}",
                "Avg Min Obs Dist (m)":f"{m['min_obs_dist']:.4f}",
            })

    # Text report
    txt_path = os.path.join(out_dir, "comparison_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  {task_label} RL – ALGORITHM COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        hdr = f"{'Algorithm':<8}  {'Success':>8}  {'Collision':>9}  "
        hdr += f"{'Avg Len':>8}  {'Avg Rew':>10}  {'ObsDist':>8}\n"
        f.write(hdr)
        f.write("-" * 70 + "\n")
        for algo, m in results.items():
            f.write(
                f"{algo:<8}  "
                f"{m['success_rate']:>8.2%}  "
                f"{m['collision_rate']:>9.2%}  "
                f"{m['avg_length']:>8.1f}  "
                f"{m['avg_reward']:>10.2f}  "
                f"{m['min_obs_dist']:>8.3f}\n"
            )
        f.write("\n" + "=" * 70 + "\n")

        # Winner per metric
        f.write("\nBEST PER METRIC\n")
        f.write("-" * 40 + "\n")
        metrics_best = {
            "Success Rate":    ("success_rate",   max),
            "Collision Rate":  ("collision_rate",  min),
            "Avg Ep Length":   ("avg_length",      min),
            "Avg Reward":      ("avg_reward",      max),
            "Min Obs Dist":    ("min_obs_dist",    max),
        }
        for label, (key, fn) in metrics_best.items():
            best = fn(results, key=lambda a: results[a][key])
            f.write(f"  {label:<18}: {best}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("  NOTE: Success ↑  Collision ↓  Ep Length ↓  Reward ↑  Obs Dist ↑\n")
        f.write("=" * 70 + "\n")

    print(f"  Saved report → {txt_path}")
    print(f"  Saved CSV    → {csv_path}")

    # Print text report to console
    with open(txt_path) as f:
        print("\n" + f.read())


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

ALL_ALGOS = ["PPO", "TD3", "SAC"] + (["TRPO"] if HAS_TRPO else [])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train & evaluate RL algorithms on a 12-DOF robot arm or KUKA pick-and-place."
    )
    parser.add_argument(
        "--env", default="arm12dof", choices=["arm12dof", "kuka"],
        help="Environment: arm12dof (default) or kuka (pick-and-place)."
    )
    parser.add_argument(
        "--algo", nargs="+", default=ALL_ALGOS,
        choices=["PPO", "TD3", "SAC", "TRPO"],
        help="Algorithm(s) to run. Default: all available.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training timesteps per algorithm (default: 200 000).",
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=50,
        help="Episodes for evaluation (default: 50).",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Show PyBullet GUI during training & evaluation.",
    )
    parser.add_argument(
        "--render_fps", type=float, default=20.0,
        help="GUI pacing in human render mode (lower is slower, default: 20).",
    )
    parser.add_argument(
        "--n_envs", type=int, default=12,
        help="Parallel environments for headless training (default: 8). "
             "Ignored when --render is set.",
    )
    parser.add_argument(
        "--eval_only", action="store_true",
        help="Skip training; load saved models from models/ and evaluate.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue training from existing <ALGO>_final.zip checkpoint if present.",
    )
    parser.add_argument(
        "--strict_curriculum", action="store_true",
        help="Use strict two-phase curriculum in KUKA env: fixed object first, then randomized.",
    )
    parser.add_argument(
        "--fixed_phase_episodes", type=int, default=300,
        help="Number of initial fixed-object episodes for strict KUKA curriculum (default: 300).",
    )
    parser.add_argument(
        "--models_dir", default=MODELS_DIR,
        help="Directory for saving / loading models.",
    )
    parser.add_argument(
        "--results_dir", default=RESULTS_DIR,
        help="Directory for plots and reports.",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    algos  = [a.upper() for a in args.algo]

    # Filter out TRPO if not installed
    if not HAS_TRPO and "TRPO" in algos:
        print("[WARNING] Removing TRPO – sb3-contrib not installed.")
        algos = [a for a in algos if a != "TRPO"]

    if not algos:
        print("No algorithms selected. Exiting.")
        return

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    loggers: dict[str, TrainingLogger] = {}
    results: dict[str, dict]           = {}

    # ── Train ────────────────────────────────────────────────────────────────
    if not args.eval_only:
        for algo in algos:
            model, logger = train(
                algo,
                total_timesteps=args.timesteps,
                render=args.render,
                n_envs=args.n_envs,
                render_fps=args.render_fps,
                save_dir=args.models_dir,
                env_name=args.env,
                resume=args.resume,
                strict_curriculum=args.strict_curriculum,
                fixed_phase_episodes=args.fixed_phase_episodes,
            )
            loggers[algo] = logger

        if loggers:
            print("\n  Generating training curves…")
            plot_training_curves(loggers, args.results_dir, env_name=args.env)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    for algo in algos:
        model_path = os.path.join(args.models_dir, f"{algo}_final.zip")

        if args.eval_only:
            if not os.path.exists(model_path):
                print(f"[SKIP] {model_path} not found.")
                continue
            # Create a temporary env matching the selected environment to load the model
            tmp_env = DummyVecEnv([
                make_env(
                    env_name=args.env,
                    strict_curriculum=args.strict_curriculum,
                    fixed_phase_episodes=args.fixed_phase_episodes,
                )
            ])
            cls_map = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
            if HAS_TRPO:
                cls_map["TRPO"] = TRPO
            model = cls_map[algo].load(model_path, env=tmp_env)
            tmp_env.close()

        print(f"\n  Evaluating {algo}…")
        metrics = evaluate(model, algo,
                   num_episodes=args.eval_episodes,
                   render=args.render,
                   render_fps=args.render_fps,
                   env_name=args.env,
                   strict_curriculum=args.strict_curriculum,
                   fixed_phase_episodes=args.fixed_phase_episodes)
        results[algo] = metrics

    # ── Report & plots ────────────────────────────────────────────────────────
    if results:
        print("\n  Generating performance plots and report…")
        plot_performance_comparison(results, args.results_dir, env_name=args.env)
        plot_radar(results, args.results_dir)
        generate_report(results, args.results_dir, env_name=args.env)

    print("\nDone. Results saved to", args.results_dir)


if __name__ == "__main__":
    main()
