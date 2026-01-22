"""
==========================================================================
                    MAIN.PY - UR5 GRID NAVIGATION
==========================================================================
Students implement DP algorithms in utils.py and run this to see results.

Dependencies:
    - pybullet
    - numpy
    - utils.py

Usage:
    python main.py

Author: Assignment 1 - AR525
==========================================================================
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import sys


from utils import (
    GridEnv,
    policy_iteration,
    value_iteration
)



def state_to_position(state, rows, cols, grid_size=0.10, 
                      table_center=[0, -0.3, 0.65], z_offset=0.10):

    row = state // cols
    col = state % cols
    
    x = table_center[0] + (col - cols/2 + 0.5) * grid_size
    y = table_center[1] + (row - rows/2 + 0.5) * grid_size
    z = table_center[2] + z_offset
    
    return [x, y, z]


def draw_grid_lines(rows, cols, grid_size=0.10, table_center=[0, -0.3, 0.65]):

    line_color = [0, 0, 0]
    line_width = 2
    z = table_center[2] + 0.001
    
    x_start = table_center[0] - (cols/2) * grid_size
    x_end = table_center[0] + (cols/2) * grid_size
    y_start = table_center[1] - (rows/2) * grid_size
    y_end = table_center[1] + (rows/2) * grid_size
    

    for i in range(rows + 1):
        y = y_start + i * grid_size
        p.addUserDebugLine([x_start, y, z], [x_end, y, z], line_color, line_width)

    for j in range(cols + 1):
        x = x_start + j * grid_size
        p.addUserDebugLine([x, y_start, z], [x, y_end, z], line_color, line_width)





if __name__ == "__main__":
    

    ROWS = 5
    COLS = 6
    START = 0
    GOAL = ROWS * COLS - 1
    GAMMA = 0.99
    
    env = GridEnv(rows=ROWS, cols=COLS, start=START, goal=GOAL)

    

    old_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )

    p.loadURDF("plane.urdf")
    
    table_path = os.path.join("assest", "table", "table.urdf")
    p.loadURDF(table_path, [0, -0.3, 0], globalScaling=2.0)
    
    stand_path = os.path.join("assest", "robot_stand.urdf")
    p.loadURDF(stand_path, [0, -0.8, 0], useFixedBase=True)
    
    ur5_path = os.path.join("assest", "ur5.urdf")
    ur5_start_pos = [0, -0.8, 0.65]
    ur5_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    ur5_id = p.loadURDF(ur5_path, ur5_start_pos, ur5_start_orn, useFixedBase=True)
    
    sys.stderr = old_stderr
    
    draw_grid_lines(env.rows, env.cols)
    
  
    all_states = set(range(env.rows * env.cols))
    available_states = list(all_states - {env.start, env.goal})
    
    num_obstacles = min(5, len(available_states))
    if len(available_states) >= num_obstacles:
        obstacle_states = np.random.choice(available_states, num_obstacles, replace=False)
        
        obstacle_path = os.path.join("assest", "cube_and_square", "cube_small_cyan.urdf")
        for obs_state in obstacle_states:
            obs_pos = state_to_position(obs_state, env.rows, env.cols, z_offset=0.025)
            p.loadURDF(obstacle_path, obs_pos)

    grid_size = 0.10
    half = grid_size / 2 * 0.8

    start_pos = state_to_position(env.start, env.rows, env.cols, z_offset=0.005)
    yellow = [1, 1, 0]
    p.addUserDebugLine([start_pos[0]-half, start_pos[1]-half, start_pos[2]], 
                       [start_pos[0]+half, start_pos[1]-half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]+half, start_pos[1]-half, start_pos[2]], 
                       [start_pos[0]+half, start_pos[1]+half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]+half, start_pos[1]+half, start_pos[2]], 
                       [start_pos[0]-half, start_pos[1]+half, start_pos[2]], yellow, 3, 0)
    p.addUserDebugLine([start_pos[0]-half, start_pos[1]+half, start_pos[2]], 
                       [start_pos[0]-half, start_pos[1]-half, start_pos[2]], yellow, 3, 0)
    
   
    goal_pos = state_to_position(env.goal, env.rows, env.cols, z_offset=0.005)
    red = [1, 0, 0]
    p.addUserDebugLine([goal_pos[0]-half, goal_pos[1]-half, goal_pos[2]], 
                       [goal_pos[0]+half, goal_pos[1]-half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]+half, goal_pos[1]-half, goal_pos[2]], 
                       [goal_pos[0]+half, goal_pos[1]+half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]+half, goal_pos[1]+half, goal_pos[2]], 
                       [goal_pos[0]-half, goal_pos[1]+half, goal_pos[2]], red, 3, 0)
    p.addUserDebugLine([goal_pos[0]-half, goal_pos[1]+half, goal_pos[2]], 
                       [goal_pos[0]-half, goal_pos[1]-half, goal_pos[2]], red, 3, 0)
    

    try:
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except:
        pass

    # ============================================================
    # TODO: Implement DP algorithms in utils.py, then add simulation code here
    # ============================================================
