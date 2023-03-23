
import time
import numpy as np
import pinocchio as pin
from example_robot_data import load

# from unified_simulators.pinocchio_sim import PinocchioSim as Simulator
from unified_simulators.pybullet_sim import PybulletSim as Simulator
from gviewer_mpc import GviewerMpc
import utils

np.set_printoptions(precision=4, linewidth=180)

from ocp_pbe_def import create_ocp_reaching_pbe

GVIEWER = True
PLOT = True

robot_name = 'panda'
robot = load(robot_name)
ee_name = 'panda_link8'
fixed_joints = ['panda_finger_joint1', 'panda_finger_joint2']
# fixed_joints = None
robot = utils.freezed_robot(robot, fixed_joints)

# delta_trans = np.array([0.0, 0.0, 0.0])
# delta_trans = np.array([0.0, 0.0, 0.4])
delta_trans = np.array([0.2, 0.2, 0.1])


# Simulation
N_sim = 30000
# dt_sim = 1/240  # pybullet default
dt_sim = 1e-3

# Number of shooting nodes
T = 100
# shooting nodes integration dt
dt_ddp = 1e-2  # seconds
# Solve every...
dt_ddp_solve = 1e-2  # seconds
PRINT_EVERY = 500
SOLVE_EVERY = int(dt_ddp_solve/dt_sim)
GOAL_IS_SE3 = False


# franka_control/config/start_pose.yaml
v0 = np.zeros(robot.nv)
x0 = np.concatenate([robot.q0, v0])

ee_fid = robot.model.getFrameId(ee_name)
oMe0 = robot.framePlacement(robot.q0, ee_fid, True)
oMe_goal = oMe0.copy()
oMe_goal.translation += delta_trans

ddp = create_ocp_reaching_pbe(robot.model, x0, ee_name, oMe_goal, T, dt_ddp, goal_is_se3=GOAL_IS_SE3, verbose=False)


# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T + 1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
# Initial solution
success = ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)

qk_sim, vk_sim = robot.q0, v0

# Simulation
sim = Simulator()
sim.init(dt_sim, robot_name, fixed_joints, visual=True)
sim.setState(x0)

# Visualization of mpc preview
gmpc = GviewerMpc(robot_name, 2, fixed_joints)

# Force disturbance
t1_fext, t2_fext = 1.0, 4.0
fext = np.array([0,30,0, 0,0,0])
# fext = np.array([0,10,0, 0,0,0])
# fext = np.array([0,0,0, 0,0,0])
frame_fext = "panda_link8"


# Logs
t_solve = []
dt_solve = []
nb_iter_solve = []
q_sim_arr = np.zeros((N_sim, robot.nq))
v_sim_arr = np.zeros((N_sim, robot.nv))
u_ref_arr = np.zeros((N_sim, robot.nv))
t_sim_arr = dt_sim*np.arange(N_sim)
print('\n==========================')
print('Begin simulation + control')
print('Apply force between ', t1_fext, t2_fext, ' seconds')
print('   -> ', t1_fext/dt_sim, t2_fext/dt_sim, ' iterations')
print('fext: ', fext)
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
    xs_arr = np.array(ddp.xs)[:,:robot.nq]
    gmpc.display_keyframes(xs_arr)

    if t1_fext < tk < t2_fext:
        sim.applyExternalForce(fext, frame_fext, rf_frame=pin.LOCAL_WORLD_ALIGNED)
    
    sim.step(u_ref_mpc)
    xk = sim.getState()
    qk_sim, vk_sim = xk[:robot.nq], xk[robot.nq:]

    # Logs
    t_sim_arr[k] = tk
    q_sim_arr[k,:] = qk_sim
    v_sim_arr[k,:] = vk_sim
    u_ref_arr[k,:] = u_ref_mpc




# if GVIEWER:
#     print('\n=================')
#     print('Realtime playback')
#     # setup visualizer 
#     robot.initViewer(loadModel=True)

#     gui = robot.viewer.gui
#     gui.addSphere("world/target", 0.05, [0, 1, 0, 0.5])
#     gui.applyConfiguration("world/target", oMe_goal.translation.tolist() + [0, 0, 0, 1])
#     gui.addSphere("world/final", 0.05, [0, 0, 1, 0.5])
#     # solution joint trajectory
#     xs = np.array(ddp.xs)
#     q_final = xs[-1, : robot.model.nq]
#     oMe_fin = robot.framePlacement(q_final, ee_fid, True)
#     gui.applyConfiguration("world/final", oMe_fin.translation.tolist() + [0, 0, 0, 1])

#     # Viewer loop 
#     k = 0
#     while k < N_sim:
#         t1 = time.time()
#         robot.display(q_sim_arr[k,:])
#         delay = time.time() - t1
#         if delay < dt_sim: 
#             time.sleep(dt_sim - delay)
#         k += 1



if PLOT:
    print('\n=========')
    print('Plot traj')
    import matplotlib.pyplot as plt

    ##############################
    # State
    fig, axes = plt.subplots(robot.nq,2)
    fig.canvas.manager.set_window_title('sim_state')
    fig.suptitle('State trajectories (q,v)', size=18)
    for i in range(robot.nq):
        axes[i,0].plot(t_sim_arr, q_sim_arr[:,i])
        axes[i,1].plot(t_sim_arr, v_sim_arr[:,i])
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)


    ##############################
    # Controls
    fig, axes = plt.subplots(robot.nq,1)
    fig.canvas.manager.set_window_title('joint_torques')
    fig.suptitle('Joint torques', size=12)
    for i in range(robot.nq):
        axes[i].plot(t_sim_arr, u_ref_arr[:,i])
    axes[-1].set_xlabel('Time (s)', fontsize=16)

    ##############################
    # Solve time
    fig, axes = plt.subplots(2,1)
    fig.canvas.manager.set_window_title('solve_times')
    axes[0].set_title('Solve times (ms)', size=12)
    axes[0].plot(t_solve, dt_solve, '.')
    axes[1].set_title('# iterations', size=12)
    axes[1].plot(t_solve, nb_iter_solve, '.')
    axes[1].set_xlabel('Time (s)', fontsize=16)
    plt.grid()

    ##############################
    # End effector pose trajectory
    oMe_lst = [robot.framePlacement(q, ee_fid, True) 
               for q in q_sim_arr]

    t_oe_arr = np.array([M.translation for M in oMe_lst])
    o_oe_arr = np.rad2deg(np.array([pin.log3(M.rotation) for M in oMe_lst]))

    fig, axes = plt.subplots(3,2)
    fig.canvas.manager.set_window_title('end_effector_traj')
    fig.suptitle('End effector trajectories (position,orientation)', size=18)
    for i in range(3):
        l = 'xyz'[i]
        c = 'rgb'[i]
        axes[i,0].plot(t_sim_arr, t_oe_arr[:,i], f'.{c}', label=f't{l}')
        axes[i,0].plot([t_sim_arr[0], t_sim_arr[-1]], 2*[oMe_goal.translation[i]], ':k', label=f'ref_t_{l}')
        axes[i,1].plot(t_sim_arr, o_oe_arr[:,i])

    plt.grid()
    plt.legend()
    axes[-1,0].set_xlabel('Time (s)', fontsize=16)
    axes[-1,1].set_xlabel('Time (s)', fontsize=16)

    # o_nu_e_lst = [robot.frameVelocity(q, v, ee_fid, True, pin.LOCAL_WORLD_ALIGNED) 
    #             for q, v in zip(q_sim_arr, v_sim_arr)]


    plt.show()