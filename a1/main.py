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
    

    # Let the scene settle briefly before running DP / animation
    for _ in range(240):  # ~1 second at 240 Hz
        p.stepSimulation()
        time.sleep(1./240.)

    # ============================================================
    # TODO: Implement DP algorithms in utils.py, then add simulation code here
    # ============================================================

    # ============================================================
    # Run DP → get optimal policy
    # ============================================================

    print("\nRunning Policy Iteration...")
    V_pi, policy_pi = policy_iteration(env, gamma=GAMMA, theta=1e-6)

    print("\nRunning Value Iteration...")
    V_vi, policy_vi = value_iteration(env, gamma=GAMMA, theta=1e-6)

    # Choose which one to visualize (you can compare both later)
    # For now we use policy iteration result
    V = V_pi
    policy = policy_pi
    method_name = "Policy Iteration"

    # V = V_vi
    # policy = policy_vi
    # method_name = "Value Iteration"

    print(f"\nUsing policy from {method_name}")

    # ============================================================
    # Extract path
    # ============================================================
    path = env.get_optimal_path(policy)

    if not path:
        print("No path found → check your policy / implementation")
        path = [env.start]  # at least show start
    else:
        print(f"Optimal path found ({len(path)} steps):")
        print(" → ".join(map(str, path)))

    # ============================================================
    # Optional: Print value function as text grid
    # ============================================================
    print(f"\nValue function ({method_name}):")
    v_grid = V.reshape(env.rows, env.cols)
    for r in range(env.rows):
        row_str = " ".join(f"{v_grid[r,c]:6.1f}" for c in range(env.cols))
        print(row_str)

    # ============================================================
    # Visualization: heatmap in console is already done above
    # Now move robot + draw green trail in PyBullet
    # ============================================================

    # Colors
    green = [0, 1, 0, 1]          # RGBA
    trail_width = 2.0
    trail_life_time = 0.0         # 0 = permanent

    prev_pos = None

    for i, state in enumerate(path):
        target_pos = state_to_position(
            state, env.rows, env.cols,
            grid_size=0.10,
            table_center=[0, -0.3, 0.65],
            z_offset=0.12               # slightly higher to avoid table
        )

        # Draw green trail (line between consecutive waypoints)
        if prev_pos is not None:
            p.addUserDebugLine(
                prev_pos,
                target_pos,
                lineColorRGB=[0, 1, 0],
                lineWidth=trail_width,
                lifeTime=trail_life_time
            )

        # Move UR5 end-effector to this position
        # We use orientation pointing downwards (typical for pick/place)
        orientation = p.getQuaternionFromEuler([0, np.pi, 0])  # gripper down

        try:
            joint_poses = p.calculateInverseKinematics(
                ur5_id,
                endEffectorLinkIndex=7,           # usually link 7 = tool tip for UR5
                targetPosition=target_pos,
                targetOrientation=orientation,
                maxNumIterations=100,
                residualThreshold=1e-5
            )

            for joint_idx in range(p.getNumJoints(ur5_id)):
                # Skip fixed / passive joints if needed
                joint_info = p.getJointInfo(ur5_id, joint_idx)
                if joint_info[2] != p.JOINT_FIXED:  # only controllable joints
                    p.setJointMotorControl2(
                        ur5_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=joint_poses[joint_idx],
                        force=800,
                        maxVelocity=1.5
                    )

            # Give some time for the robot to reach ≈ position
            for _ in range(240 * 2):   # ≈ 2 seconds at 240 Hz
                p.stepSimulation()
                time.sleep(1./240.)

        except Exception as e:
            print(f"IK failed at state {state}: {e}")
            # continue anyway — show at least the trail

        prev_pos = target_pos

    print("Simulation finished. Robot should have followed the path.")

    # Keep window open until user closes it
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1./240.)
    except:
        pass
