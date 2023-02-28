
import numpy as np
import pinocchio as pin
# np.set_printoptions(precision=4, linewidth=180)
import ocp_utils
import time
import matplotlib.pyplot as plt

from ocp_pbe_def import create_ocp_reaching_pbe
from bench_croco import MPCBenchmark



# Load model (hardcoded for now, eventually should be in example-robot-data)
urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/config/panda_inertias_nohand.urdf"
package_dirs = ["/home/mfourmy/catkin_ws/src/franka_ros/"]
robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs)

delta_trans = np.array([0.2, 0.0, -0.0])

# Number of shooting nodes
T = 100
dt = 1e-2  # seconds

# franka_control/config/start_pose.yaml
q0 = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])

ee_frame_name = 'panda_link8'
oMe_0 = robot.framePlacement(q0, robot.model.getFrameId(ee_frame_name), update_kinematics=True)

N_SOLVE = 5

NX, NY = 10, 10
# NX, NY = 10, 5
dx_vals = np.linspace(-0.5, 0.6, NX)
dy_vals = np.linspace(-0.6, 0.6, NY)
dx_vals = np.around(dx_vals, 2)
dy_vals = np.around(dy_vals, 2)

t_solve_avg = np.zeros((NX, NY))
t_solve_avg_bench = np.zeros((NX, NY))
t_solve_std = np.zeros((NX, NY))
t_solve_ws_avg = np.zeros((NX, NY))
t_solve_ws_avg_bench = np.zeros((NX, NY))
t_solve_ws_std = np.zeros((NX, NY))


bench = MPCBenchmark()


# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T+1)]

for i, dx in enumerate(dx_vals):
    for j, dy in enumerate(dy_vals):
        print(i, j)
        delta_trans = np.array([dx, dy, -0.0])
        oMe_goal = oMe_0.copy()
        oMe_goal.translation += delta_trans
        oMe_goal.rotation = np.eye(3)
        # TODO: change ref instead of creating new one
        ddp = create_ocp_reaching_pbe(robot.model, x0, ee_frame_name, oMe_goal, T, dt, goal_is_se3=False, verbose=False)

        # Solve and measure timings
        bench.reset_profiles()
        bench.start_croco_profiler()
        
        solve_times = np.zeros(N_SOLVE)
        solve_times_ws = np.zeros(N_SOLVE)

        xs_init_lst = []
        us_init_lst = []
        for k in range(N_SOLVE):
            t1 = time.time()
            # Quasi-static warm start
            us_init = ddp.problem.quasiStatic(xs_init[:-1])
            ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
            solve_times[k] = 1e3*(time.time() - t1)

            xs_init_lst.append(ddp.xs)
            us_init_lst.append(ddp.us) 
        
        bench.stop_croco_profiler()
        bench.record_profiles()
        t_solve_avg[i,j] = solve_times.mean()
        # bench.print_croco_profiler()
        # print("bench.avg['SolverFDDP::solve']")
        # print(bench.avg['SolverFDDP::solve'])
        t_solve_avg_bench[i,j] = bench.avg['SolverFDDP::solve'][0]
        t_solve_std[i,j] = np.sqrt(np.std(solve_times - solve_times.mean()))
        
        bench.reset_profiles()
        bench.start_croco_profiler()
        for k in range(N_SOLVE):
            t1 = time.time()
            # Previous solution warm start 
            ddp.problem.x0 = x0
            xs_init = xs_init_lst[k]
            us_init = us_init_lst[k]
            ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
            ddp_data_warm = ocp_utils.extract_ocp_data(ddp, ee_frame_name=ee_frame_name)
            solve_times_ws[k] = 1e3*(time.time() - t1)
        
        bench.stop_croco_profiler()
        bench.record_profiles()
        t_solve_ws_avg[i,j] = solve_times_ws.mean()
        t_solve_ws_avg_bench[i,j] = bench.avg['SolverFDDP::solve'][0]
        t_solve_ws_std[i,j] = np.sqrt(np.std(solve_times_ws - solve_times_ws.mean()))



fig = plt.figure('solve time avg')
im = plt.imshow(t_solve_avg)
plt.xticks(np.arange(NX), labels=dx_vals)
plt.yticks(np.arange(NY), labels=dy_vals)
cbar = fig.colorbar(im, ax=fig.axes[0])

# fig = plt.figure('solve time avg BENCH')
# im = plt.imshow(t_solve_avg)
# plt.xticks(np.arange(NX), labels=dx_vals)
# plt.yticks(np.arange(NY), labels=dy_vals)
# cbar = fig.colorbar(im, ax=fig.axes[0])

#############################
# NO DIFFERENCE
#############################
# fig = plt.figure('solve time avg BENCH vs TIME')
# im = plt.imshow(t_solve_avg_bench - t_solve_avg_bench)
# plt.xticks(np.arange(NX), labels=dx_vals)
# plt.yticks(np.arange(NY), labels=dy_vals)
# cbar = fig.colorbar(im, ax=fig.axes[0])


fig = plt.figure('solve time std')
plt.imshow(t_solve_std)
plt.xticks(np.arange(NX), labels=dx_vals)
plt.yticks(np.arange(NY), labels=dy_vals)
cbar = fig.colorbar(im, ax=fig.axes[0])

fig = plt.figure('solve time WS avg')
im = plt.imshow(t_solve_ws_avg)
plt.xticks(np.arange(NX), labels=dx_vals)
plt.yticks(np.arange(NY), labels=dy_vals)
cbar = fig.colorbar(im, ax=fig.axes[0])

# fig = plt.figure('solve time WS avg BENCH')
# im = plt.imshow(t_solve_ws_avg_bench)
# plt.xticks(np.arange(NX), labels=dx_vals)
# plt.yticks(np.arange(NY), labels=dy_vals)
# cbar = fig.colorbar(im, ax=fig.axes[0])

fig = plt.figure('solve time WS std')
plt.imshow(t_solve_ws_std)
plt.xticks(np.arange(NX), labels=dx_vals)
plt.yticks(np.arange(NY), labels=dy_vals)
cbar = fig.colorbar(im, ax=fig.axes[0])




plt.show()