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
    
    def get_optimal_path(self, policy):

            
            # ============================================================
            # TODO: Students can modify this to extract path from policy
            # ============================================================

        
        return path


# ==========================================================================
#                  DYNAMIC PROGRAMMING ALGORITHMS - TODO
# ==========================================================================

def policy_evaluation(env, policy, gamma=1.0, theta=1e-8):
    ###############################################
    # TODO: Implement Policy Evaluation
    ###############################################
    pass


def q_from_v():

    ###############################################
    # TODO: Implement Q-value computation
    ###############################################
    pass


def policy_improvement():
 
    ###############################################
    # TODO: Implement Policy Improvement
    ###############################################
    pass


def policy_iteration():
 
    ###############################################
    # TODO: Implement Policy Iteration
    ###############################################
    pass


def value_iteration():
 
    ###############################################
    # TODO: Implement Value Iteration
    ###############################################
    pass
