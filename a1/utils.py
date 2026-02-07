"""
==========================================================================
                        UTILS.PY - STUDENT IMPLEMENTATION
==========================================================================
Students must implement the Dynamic Programming algorithms below.

Author: Assignment 1 - AR525
==========================================================================
"""

import numpy as np

class GridEnv:
    
    def __init__(self, rows=5, cols=6, start=0, goal=29):
   
        self.rows = rows
        self.cols = cols
        self.nS = rows * cols
        self.nA = 4
        self.start = start
        self.goal = goal
        self.action_names = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}
        self.P = self._build_dynamics()
    
    def _state_to_pos(self, state):

        return state // self.cols, state % self.cols
    
    def _pos_to_state(self, row, col):

        return row * self.cols + col
    
    def _is_valid_pos(self, row, col):

        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_next_state(self, state, action):
  
        row, col = self._state_to_pos(state)
        
        if action == 0:    # LEFT
            col -= 1
        elif action == 1:  # DOWN
            row += 1
        elif action == 2:  # RIGHT
            col += 1
        elif action == 3:  # UP
            row -= 1
        
        if not self._is_valid_pos(row, col):
            return state
        
        return self._pos_to_state(row, col)
    
    def _build_dynamics(self):

        P = {}
        
        for state in range(self.nS):
            P[state] = {}
            
            for action in range(self.nA):
                next_state = self._get_next_state(state, action)
                
                # ============================================================
                # TODO: Define your reward structure here!
                # ============================================================
              
                
                # TEMPORARY: Default rewards (students should modify this)
                if next_state == self.goal:
                    reward = 100.0
                    done = True
                else:
                    reward = -1.0
                    done = False
                
                P[state][action] = [(1.0, next_state, reward, done)]
        
        return P
    
    # def get_optimal_path(self, policy):
    #
    #
    #         # ============================================================
    #         # TODO: Students can modify this to extract path from policy
    #         # ============================================================
    #
    #
    #     return path

    def get_optimal_path(self, policy):
        path = []
        state = self.start
        path.append(state)

        # Prevent infinite loop in case of bad policy
        seen = set()
        max_steps = self.nS * 2

        while state != self.goal and len(path) < max_steps:
            if state in seen:
                return []  # cycle → bad policy
            seen.add(state)

            action = policy[state]
            next_state = self._get_next_state(state, action)

            # If action didn't change state (wall), stop to avoid loop
            if next_state == state:
                break

            path.append(next_state)
            state = next_state

        if state == self.goal:
            return path
        else:
            return []  # failed to reach goal


# ==========================================================================
#                  DYNAMIC PROGRAMMING ALGORITHMS - TODO
# ==========================================================================

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Iterative policy evaluation
    policy: dict {state: action} or array of shape [nS] with action indices
    Returns: V (np.array of shape [nS])
    """
    V = np.zeros(env.nS)
    V[env.goal] = 0.0

    while True:
        delta = 0

        for s in range(env.nS):
            if s == env.goal:
                continue

            v = V[s]
            # ensure policy indexing works for lists/ndarrays
            a = int(policy[s])

            # Look up transition (in your env there's only one successor per action)
            prob, next_state, reward, done = env.P[s][a][0]

            V[s] = reward + gamma * V[next_state] * (0.0 if done else 1.0)

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def q_from_v(env, V, gamma=0.99):
    Q = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in env.P[s]:
            _, ns, r, done = env.P[s][a][0]
            Q[s, a] = r + gamma * V[ns] * (1 - done)
    return Q


def policy_improvement(env, V, gamma=0.99):
    """
    Greedy policy improvement
    Returns: new_policy [nS] → action index
    """
    policy = np.zeros(env.nS, dtype=int)
    Q = q_from_v(env, V, gamma)

    for s in range(env.nS):
        if s == env.goal:
            continue  # terminal — policy doesn't matter
        policy[s] = np.argmax(Q[s])

    return policy


def policy_iteration(env, gamma=0.99, theta=1e-8):
    """
    Full policy iteration
    Returns: (optimal V, optimal policy)
    """
    # Start with arbitrary policy (e.g. always RIGHT)
    policy = np.ones(env.nS, dtype=int) * 2  # 2 = RIGHT

    iteration = 0

    while True:
        iteration += 1
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)

        # Check if policy stabilized
        if np.array_equal(new_policy, policy):
            print(f"Policy iteration converged after {iteration} policy updates")
            break

        policy = new_policy

    return V, policy


def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Value iteration (Bellman optimality)
    Returns: (optimal V, optimal policy)
    """
    V = np.zeros(env.nS)
    iteration = 0

    while True:
        iteration += 1
        delta = 0

        for s in range(env.nS):
            if s == env.goal:
                continue

            v = V[s]
            # Compute max over actions
            values = []
            for a in range(env.nA):
                _, next_state, reward, done = env.P[s][a][0]
                values.append(reward + gamma * V[next_state] * (not done))

            V[s] = max(values)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            print(f"Value iteration converged after {iteration} sweeps")
            break

    # Extract policy
    policy = np.zeros(env.nS, dtype=int)
    Q = q_from_v(env, V, gamma)
    for s in range(env.nS):
        if s != env.goal:
            policy[s] = np.argmax(Q[s])

    return V, policy





