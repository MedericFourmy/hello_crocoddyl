
import time
import numpy as np
import pinocchio as pin
import config_panda as conf 

np.set_printoptions(precision=4, linewidth=180)

from ocp_pbe_def import create_ocp_reaching_pbe

GVIEWER = True
PLOT = True
USE_PYBULLET = True
USE_PYBULLET_GUI = True

# Load model (hardcoded for now, eventually should be in example-robot-data)
robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)

delta_trans = np.array([0.2, 0.0, -0.0])
# delta_trans = np.array([0.5, 0.4, -0.0])
# -0.5, 0.6


# Simulation
N_sim = 5000
# dt_sim = 1/240  # pybullet
dt_sim = 1e-3

# Number of shooting nodes
T = 100
# shooting nodes integration dt
dt_ddp = 1e-3  # seconds
# Solve every...
dt_ddp_solve = 1e-3  # seconds
PRINT_EVERY = 500
SOLVE_EVERY = int(dt_ddp_solve/dt_sim)


# franka_control/config/start_pose.yaml
q0 = conf.q0
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])

ee_fid = robot.model.getFrameId(conf.ee_name)
oMe_0 = robot.framePlacement(q0, ee_fid, True)
oMe_goal = oMe_0.copy()
oMe_goal.translation += delta_trans
oMe_goal.rotation = np.eye(3)
print(oMe_0)

ddp = create_ocp_reaching_pbe(robot.model, x0, conf.ee_name, oMe_goal, T, dt_ddp, goal_is_se3=False, verbose=False)


# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T + 1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
# Initial solution
success = ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

qk_sim, vk_sim = q0, v0

# Simulation
if USE_PYBULLET:
    from pybullet_sim import PybulletSim

    sim = PybulletSim(dt_sim, conf.urdf_path, conf.package_dirs, conf.joint_names)
    sim.reset_state(conf.q0, conf.v0)



# Logs
t_solve = []
dt_solve = []
nb_iter_solve = []
q_sim_arr = np.zeros((N_sim, 7))
v_sim_arr = np.zeros((N_sim, 7))
dv_sim_arr = np.zeros((N_sim, 7))
u_ref_arr = np.zeros((N_sim, 7))
t_sim_arr = dt_sim*np.arange(N_sim)
for k in range(N_sim):
    xk_sim = np.concatenate([qk_sim, vk_sim])

    tk = dt_sim*k 

    if (k % PRINT_EVERY) == 0:
        print(f'{k}/{N_sim}')

    #  Warm start using previous solution
    if (k % SOLVE_EVERY) == 0:
        ddp.problem.x0 = xk_sim
        xs_init = list(ddp.xs[1:]) + [ddp.xs[-1]]  # shift solution
        xs_init[0] = xk_sim
        us_init = list(ddp.us[1:]) + [ddp.us[-1]]

        # Solve
        t1 = time.time()
        success = ddp.solve(xs_init, us_init, maxiter=10, isFeasible=False)
        t_solve.append(tk)
        dt_solve.append(1e3*(time.time() - t1))  # store milliseconds
        nb_iter_solve.append(ddp.iter)
    
    # control to apply
    u_ref_mpc = ddp.us[0]
    # print(u_ref_mpc)

    if USE_PYBULLET:
        sim.send_joint_command(u_ref_mpc)
        sim.step_simulation()
        qk_sim, vk_sim = sim.get_state()
        dvk_sim = np.zeros(7)  #? pb.getJointaccl??

    else:
        # using current torque cmd, compute simulation acceleration 
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




if GVIEWER:
    # setup visualizer 
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
        delay = time.time() - t1
        if delay < dt_sim: 
            time.sleep(dt_sim - delay)
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
    fig, axes = plt.subplots(2,1)
    fig.canvas.manager.set_window_title('solve_times')
    axes[0].set_title('Solve times (ms)', size=18)
    axes[0].plot(t_solve, dt_solve)
    axes[0].set_title('# iterations', size=18)
    axes[1].plot(t_solve, nb_iter_solve)
    axes[1].set_xlabel('Time (s)', fontsize=16)


    plt.show()