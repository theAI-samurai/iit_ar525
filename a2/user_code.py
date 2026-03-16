"""
Student Template for RL Assignment: Drone Hover Task.

This template provides stubs for:
1. Monte Carlo (MC) Learning
2. Temporal Difference (TD) Learning - Q-Learning

Students should implement the missing parts marked with TODO.

Environment: HoverAviary - Drone must hover at z=1.0
State: 3D position (x, y, z) relative to target [0, 0, 1]
Action: 3 discrete actions (thrust adjustment): -1, 0, +1
Reward: Based on proximity to target position
"""

import numpy as np
import gymnasium as gym
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ========================================
# CONFIGURATION (Students can modify)
# ========================================
NUM_BINS = 20
STATE_DIM = 3
NUM_EPISODES = 10
MAX_STEPS = 240

EPSILON = 0.1
GAMMA = 0.99
ALPHA = 0.1

# ========================================
# HELPER FUNCTIONS (Do not modify)
# ========================================

def discretize_state(state, num_bins=NUM_BINS):
    """Convert continuous state to discrete bins."""
    state = np.asarray(state)
    if state.ndim == 2:
        state = state[0, 0:3]
    else:
        state = state[0:3]

    bounds = np.array([[-1, 1], [-1, 1], [0, 2]])

    discrete = []
    for val, (low, high) in zip(state, bounds):
        val = np.clip(val, low, high)
        normalized = (val - low) / (high - low)
        bin_idx = int(normalized * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)
        discrete.append(bin_idx)

    return tuple(discrete)

def get_action_space_size():
    """Returns the size of the action space."""
    return 3

def action_index_to_value(action_idx):
    """Map action index {0,1,2} to thrust adjustment {-1,0,+1}."""
    return float(action_idx - 1)

def get_q_table_shape():
    """Returns the shape of the Q-table."""
    return (NUM_BINS,) * STATE_DIM + (get_action_space_size(),)

def initialize_q_table():
    """Initialize Q-table with zeros."""
    return np.zeros(get_q_table_shape())

def choose_action(q_table, state, epsilon):
    """Epsilon-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(get_action_space_size())
    return np.argmax(q_table[state])

def extract_position(obs):
    """Extract (x, y, z) from HoverAviary observation."""
    obs_arr = np.asarray(obs)
    if obs_arr.ndim == 2:
        return obs_arr[0, 0:3]
    return obs_arr[0:3]

def format_action(action):
    """Format discrete action index for ONE_D_RPM env.step()."""
    return np.array([[action_index_to_value(action)]], dtype=np.float32)

def evaluate_policy(env, q_table, num_episodes=10):
    """Evaluate learned policy (greedy, no exploration)."""
    rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(extract_position(state))
        total_reward = 0

        for _ in range(MAX_STEPS):
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_state))

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        rewards.append(total_reward)

    return np.mean(rewards), np.std(rewards)

# ========================================
# TODO: MONTE CARLO IMPLEMENTATION
# ========================================

def run_monte_carlo(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA):
    """
    TODO: Implement Monte Carlo Control (first-visit MC).

    Steps:
    1. Generate episodes with epsilon-greedy policy.
    2. Compute discounted returns G_t.
    3. Update Q-values using first-visit MC averaging.
    4. Track episode reward for each episode.

    Returns:
        tuple: (q_table, episode_rewards)
    """
    q_table = initialize_q_table()
    episode_rewards = []
    returns = {}
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(extract_position(state))
        episode_data = []
        total_reward = 0
        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_obs))
            episode_data.append((state, action, reward))
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
        # Process episode
        G = 0
        visited = set()
        for t in reversed(range(len(episode_data))):
            state, action, reward = episode_data[t]
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)
                q_table[state][action] = np.mean(returns[(state, action)])
    return q_table, episode_rewards

# ========================================
# TODO: TD LEARNING IMPLEMENTATION (Q-LEARNING)
# ========================================

def run_q_learning(env, num_episodes=NUM_EPISODES, epsilon=EPSILON, gamma=GAMMA, alpha=ALPHA):
    """
    TODO: Implement Q-Learning (off-policy TD control).

    Update rule:
    Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

    Returns:
        tuple: (q_table, episode_rewards)
    """
    q_table = initialize_q_table()
    episode_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(extract_position(state))
        total_reward = 0
        for step in range(MAX_STEPS):
            action = choose_action(q_table, state, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(format_action(action))
            next_state = discretize_state(extract_position(next_obs))
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        episode_rewards.append(total_reward)
    return q_table, episode_rewards

# ========================================
# MAIN FUNCTION (Do not modify)
# ========================================

def main():
    """Main function to run MC and TD learning experiments."""

    print("=" * 60)
    print("RL Assignment: Monte Carlo vs TD Learning")
    print("Task: Drone Hover at z=1.0")
    print("=" * 60)

    env = HoverAviary(obs=ObservationType.KIN, act=ActionType.ONE_D_RPM, gui=False)
    print("Environment: HoverAviary")
    print(f"Target Position: {env.TARGET_POS}")
    print(f"Episode Length: {MAX_STEPS} steps ({MAX_STEPS/30:.1f} seconds)")
    print()

    print("-" * 40)
    print("Training Monte Carlo...")
    print("-" * 40)
    q_table_mc, rewards_mc = run_monte_carlo(env, num_episodes=NUM_EPISODES)
    mean_mc, std_mc = evaluate_policy(env, q_table_mc)
    print(f"MC Final Evaluation: {mean_mc:.2f} (+/- {std_mc:.2f})")

    print()
    print("-" * 40)
    print("Training Q-Learning...")
    print("-" * 40)
    q_table_td, rewards_td = run_q_learning(env, num_episodes=NUM_EPISODES)
    mean_td, std_td = evaluate_policy(env, q_table_td)
    print(f"TD Final Evaluation: {mean_td:.2f} (+/- {std_td:.2f})")

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Monte Carlo - Final Avg Reward (last 50): {np.mean(rewards_mc[-50:]):.2f}")
    print(f"Q-Learning  - Final Avg Reward (last 50): {np.mean(rewards_td[-50:]):.2f}")
    print()
    print(f"Monte Carlo - Evaluation: {mean_mc:.2f} (+/- {std_mc:.2f})")
    print(f"Q-Learning  - Evaluation: {mean_td:.2f} (+/- {std_td:.2f})")
    print()

    if mean_mc > mean_td:
        print("Monte Carlo performed better!")
    elif mean_td > mean_mc:
        print("Q-Learning performed better!")
    else:
        print("Both performed equally!")

    env.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
