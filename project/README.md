# 12-DOF Robot Arm – Reinforcement Learning with Collision Avoidance

A PyBullet simulation of a **12-degree-of-freedom serial-chain robot arm** trained to reach random 3-D goals while avoiding obstacles, using four RL algorithms: **PPO**, **TD3**, **SAC**, and **TRPO**.

---

## Project Structure

```
RL_project/
├── main.py              # Training, evaluation, plotting, report generation
├── utils.py             # RobotArmEnv (gymnasium.Env), URDF generator
├── robot_12dof.urdf     # Auto-generated on first run (12-joint chain robot)
├── requirements.txt
├── models/              # Saved model checkpoints  (.zip)
├── results/             # Plots and comparison reports
│   ├── training_curves.png
│   ├── performance_comparison.png
│   ├── radar_comparison.png
│   ├── comparison_report.txt
│   └── comparison_report.csv
└── logs/
    └── tensorboard/     # TensorBoard event files
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **TRPO** requires [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib):
> ```bash
> pip install sb3-contrib
> ```

---

## Environment Details

| Property            | Value                                      |
|---------------------|--------------------------------------------|
| DOF                 | 12 (revolute joints, alternating Z/Y/X axes) |
| Action space        | `Box(-1, 1, shape=(12,))` → scaled to `[-π, π]` |
| Observation space   | 48-dim: joint pos/vel · EE pose · goal · obstacle positions |
| Obstacles           | 5 randomly placed spheres and boxes       |
| Goal threshold      | 0.05 m from end-effector to goal          |
| Max episode steps   | 500                                        |

### Reward Function

| Component                  | Value                                     |
|----------------------------|-------------------------------------------|
| Distance to goal           | `-dist` (continuous shaping)              |
| Success bonus              | `+50` when EE within 0.05 m of goal       |
| Collision penalty          | `-100` + episode termination              |
| Proximity penalty          | `-25 × (0.15 - dist)` if EE within 15 cm of any obstacle |
| Smoothness                 | `-0.01 × Σ(joint_vel²)`                  |
| Time step penalty          | `-0.05` per step                          |

---

## Commands

### Train all algorithms (default 200 000 steps each)

```bash
python main.py
```

### Train specific algorithm(s)

```bash
python main.py --env kuka --algo PPO
python main.py --env kuka --algo TD3 SAC
python main.py --env kuka --algo PPO TD3 SAC TRPO
```

### Train with longer horizon

```bash
python main.py --env kuka --timesteps 500000
```

### Continue training from a saved checkpoint (`--resume`)

```bash
# Example: continue all KUKA models from 500k to 1M (add 500k more)
python main.py --env kuka --algo PPO TD3 SAC TRPO --timesteps 500000 --resume --models_dir ./models_kuka --results_dir ./results_kuka

# Continue one model only
python main.py --env kuka --algo SAC --timesteps 500000 --resume --models_dir ./models_kuka --results_dir ./results_kuka
```

### Train with PyBullet GUI (visualisation)

```bash
python main.py --render
```

### Evaluate saved models (skip training)

```bash
python main.py --env kuka --eval_only
```

### Evaluate a single algorithm with rendering

```bash
python main.py --env kuka --algo PPO --eval_only --render --render_fps 8 --eval_episodes 20
```

### Evaluate all algorithms without re-training

```bash
python main.py --env kuka --eval_only --eval_episodes 100
```

### Evaluate all algorithms with rendering and FPS control

```bash
python main.py --env kuka --eval_only --render --render_fps 8 --eval_episodes 50
```

### Evaluate each model with rendering and FPS control

```bash
python main.py --env kuka --algo PPO --eval_only --render --render_fps 8 --eval_episodes 50
python main.py --env kuka --algo TD3 --eval_only --render --render_fps 8 --eval_episodes 50
python main.py --env kuka --algo SAC --eval_only --render --render_fps 8 --eval_episodes 50
python main.py --env kuka --algo TRPO --eval_only --render --render_fps 8 --eval_episodes 50
```

### Full options

```
usage: main.py [-h] [--algo {PPO,TD3,SAC,TRPO} [...]]
               [--env {arm12dof,kuka}] [--timesteps TIMESTEPS]
               [--eval_episodes EVAL_EPISODES]
               [--render] [--render_fps RENDER_FPS] [--n_envs N_ENVS]
               [--eval_only] [--resume]
               [--models_dir MODELS_DIR] [--results_dir RESULTS_DIR]

options:
  --env             Environment to run: arm12dof or kuka
  --algo            Algorithm(s) to run (default: all available)
  --timesteps       Training timesteps per algorithm (default: 200000)
  --eval_episodes   Evaluation episodes (default: 50)
  --render          Show PyBullet GUI window
  --render_fps      GUI pacing in human render mode (default: 20)
  --n_envs          Parallel envs for headless training (default: 8)
  --eval_only       Load saved models, skip training
  --resume          Continue training from existing <ALGO>_final.zip checkpoint
  --models_dir      Where to save/load model .zip files (default: ./models)
  --results_dir     Where to save plots/reports (default: ./results)
```

### TensorBoard monitoring (during or after training)

```bash
tensorboard --logdir logs/tensorboard
```

---

## Outputs

| File | Description |
|------|-------------|
| `results/training_curves.png` | Reward, episode length, success rate, collision rate over training timesteps for all algorithms |
| `results/performance_comparison.png` | Bar charts comparing final metrics across algorithms |
| `results/radar_comparison.png` | Spider/radar chart – multi-metric normalised comparison |
| `results/comparison_report.txt` | Human-readable table with best-per-metric summary |
| `results/comparison_report.csv` | Machine-readable CSV of evaluation metrics |
| `models/<ALGO>_final.zip` | Final trained SB3 model for each algorithm |

---

## Algorithm Hyperparameters

| Parameter         | PPO         | TD3         | SAC         | TRPO        |
|-------------------|-------------|-------------|-------------|-------------|
| Learning rate     | 3e-4        | 3e-4        | 3e-4        | 1e-3        |
| Batch size        | 256         | 256         | 256         | 128         |
| Network arch      | [256, 256]  | [256, 256]  | [256, 256]  | [256, 256]  |
| Gamma             | 0.99        | 0.99        | 0.99        | 0.99        |
| Buffer size       | –           | 500 000     | 500 000     | –           |
| n_steps           | 1024        | –           | –           | 1024        |
| Entropy coef      | 0.01        | –           | auto        | –           |
| Target KL         | –           | –           | –           | 0.01        |
| Action noise      | –           | Normal(0,0.1)| –          | –           |

---

## Notes

- The URDF `robot_12dof.urdf` is generated automatically on first run.
- PyBullet uses **DIRECT** mode by default (headless). Pass `--render` for the GUI.
- Each algorithm trains in a **separate physics server** to avoid state contamination.
- TRPO requires `sb3-contrib`; the script runs without it (just skips TRPO).
