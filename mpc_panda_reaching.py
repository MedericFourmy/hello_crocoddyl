
import time
import numpy as np
import pinocchio as pin
# from bench_croco import MPCBenchmark

np.set_printoptions(precision=4, linewidth=180)

from ocp_pbe_def import create_ocp_reaching_pbe

VIEW = True
PLOT = True

# Load model (hardcoded for now, eventually should be in example-robot-data)
urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/config/panda_inertias_nohand.urdf"
package_dirs = ["/home/mfourmy/catkin_ws/src/franka_ros/"]
robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs)

delta_trans = np.array([0.2, 0.0, -0.0])

# Number of shooting nodes
T = 100
# shooting nodes integration dt
dt_ddp = 1e-2  # seconds
# Solve every...
dt_ddp_solve = 1e-2  # seconds

# Simulation
N_sim = 3000
dt_sim = 1e-3

# franka_control/config/start_pose.yaml
q0 = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])

ee_frame_name = "panda_link8"
ee_fid = robot.model.getFrameId(ee_frame_name)
oMe_0 = robot.framePlacement(q0, ee_fid, True)
oMe_goal = oMe_0.copy()
oMe_goal.translation += delta_trans
oMe_goal.rotation = np.eye(3)
print(oMe_0)

ddp = create_ocp_reaching_pbe(robot.model, x0, ee_frame_name, oMe_goal, T, dt_ddp, goal_is_se3=False, verbose=False)


# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T + 1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
# Initial solution
x_traj, u_traj, success = ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

qk_sim, vk_sim = q0, v0

# Logs
t_solve = []
dt_solve = []
q_sim_arr = np.zeros((N_sim, 7))
v_sim_arr = np.zeros((N_sim, 7))
dv_sim_arr = np.zeros((N_sim, 7))
u_ref_arr = np.zeros((N_sim, 7))
t_sim_arr = dt_sim*np.arange(N_sim)
for k in range(N_sim):
    xk_sim = np.concatenate([qk_sim, vk_sim])

    tk = dt_sim*k 

    if (k % 1000) == 0:
        print(f'{k}/{N_sim}')

    #  Warm start using previous solution
    ddp.problem.x0 = xk_sim
    xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]  # shift solution
    xs_init[0] = xk_sim
    us_init = list(ddp.us[1:]) + [ddp.us[-1]]

    # Solve
    t1 = time.time()
    success = ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
    print(success)
    t_solve.append(tk)
    dt_solve.append(1e3*(time.time() - t1))

    # using current torque cmd, compute simulation acceleration 
    u_ref_mpc = ddp.us[0]
    dvk_sim = pin.aba(robot.model, robot.data, qk_sim, vk_sim, u_ref_mpc)

    # simulation step: integrate the current acceleration 
    # vk_sim += dvk_sim*dt_sim
    # qk_sim += vk_sim*dt_sim

    v_mean = vk_sim + 0.5*dvk_sim*dt_sim
    vk_sim += dvk_sim*dt_sim
    qk_sim = pin.integrate(robot.model, qk_sim, v_mean*dt_sim)


    # Logs
    t_sim_arr[k] = tk
    q_sim_arr[k,:] = qk_sim
    v_sim_arr[k,:] = vk_sim
    dv_sim_arr[k,:] = dvk_sim
    u_ref_arr[k,:] = u_ref_mpc



if VIEW:
    # setup visualizer (instead of simulator)
    viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(loadModel=True)

    viz.viewer.gui.addSphere("world/target", 0.05, [0, 1, 0, 0.5])
    viz.viewer.gui.applyConfiguration("world/target", oMe_goal.translation.tolist() + [0, 0, 0, 1])
    viz.viewer.gui.addSphere("world/final", 0.05, [0, 0, 1, 0.5])
    # solution joint trajectory
    xs = np.array(ddp.xs)
    q_final = xs[-1, : robot.model.nq]
    oMe_fin = robot.framePlacement(q_final, ee_fid, True)
    viz.viewer.gui.applyConfiguration("world/final", oMe_fin.translation.tolist() + [0, 0, 0, 1])

    # Viewer loop 
    k = 0
    while k < N_sim:
        t1 = time.time()
        viz.display(q_sim_arr[k,:])
        delay =  dt_sim - (time.time() - t1)
        if delay > 0: 
            time.sleep(delay)
        k += 1



if PLOT:
    import matplotlib.pyplot as plt

    # State
    fig, axes = plt.subplots(7,2)
    fig.canvas.manager.set_window_title('sim_state')
    fig.suptitle('State trajectories (q,v)', size=18)
    for i in range(7):
        axes[i,0].plot(t_sim_arr, q_sim_arr[:,i])
        axes[i,1].plot(t_sim_arr, v_sim_arr[:,i])
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)

    # End effector pose trajectory
    oMe_lst = [robot.framePlacement(q, ee_fid, True) 
               for q in q_sim_arr]

    t_oe_arr = np.array([M.translation for M in oMe_lst])
    o_oe_arr = np.rad2deg(np.array([pin.log3(M.rotation) for M in oMe_lst]))

    fig, axes = plt.subplots(3,2)
    fig.canvas.manager.set_window_title('end_effector_traj')
    fig.suptitle('End effector trajectories (position,orientation)', size=18)
    for i in range(3):
        axes[i,0].plot(t_sim_arr, t_oe_arr[:,i])
        axes[i,1].plot(t_sim_arr, o_oe_arr[:,i])
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)

    # o_nu_e_lst = [robot.frameVelocity(q, v, ee_fid, True, pin.LOCAL_WORLD_ALIGNED) 
    #             for q, v in zip(q_sim_arr, v_sim_arr)]


    # Controls
    fig, axes = plt.subplots(7,1)
    fig.canvas.manager.set_window_title('joint_torques')
    fig.suptitle('Joint torques', size=18)
    for i in range(7):
        axes[i].plot(t_sim_arr, u_ref_arr[:,i])
    axes[-1].set_xlabel('Time (s)', fontsize=16)

    # Solve time
    fig, ax = plt.subplots(1,1)
    fig.canvas.manager.set_window_title('solve_times')
    fig.suptitle('Solve times (ms)', size=18)
    ax.plot(t_solve, dt_solve)
    ax.set_xlabel('Time (s)', fontsize=16)


    plt.show()