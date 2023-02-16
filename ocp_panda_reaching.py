"""
Example script : Crocoddyl OCP with Panda arm 
static target reaching task

Directly inspired by https://github.com/skleff1994/minimal_examples_crocoddyl/blob/master/ocp_kuka_reaching.py

# Questions:
- How to update reference translation without rebuilding another ddp problem?
-> ddp.problem.terminalModel.differential.costs.costs['placement'].cost.residual.reference.translation = new_ref?
"""

import crocoddyl
import numpy as np
import pinocchio as pin

np.set_printoptions(precision=4, linewidth=180)
import ocp_utils

from ocp_pbe_def import create_ocp_reaching_pbe

# Load model (hardcoded for now, eventually should be in example-robot-data)
urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/config/panda_inertias_nohand.urdf"
mesh_path = "/home/mfourmy/catkin_ws/src/franka_ros/franka_description/meshes/"
robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
delta_trans = np.array([0.2, 0.0, -0.0])

# Number of shooting nodes
T = 200
dt = 1e-2  # seconds

# franka_control/config/start_pose.yaml
q0 = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])

ee_frame_name = "panda_link8"
oMe_0 = robot.framePlacement(q0, robot.model.getFrameId(ee_frame_name), update_kinematics=True)
oMe_goal = oMe_0.copy()
oMe_goal.translation += delta_trans
oMe_goal.rotation = np.eye(3)
print(oMe_0)

ddp = create_ocp_reaching_pbe(robot.model, x0, ee_frame_name, oMe_goal, T, dt, goal_is_se3=True, verbose=False)

# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T + 1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
# Extract DDP data and plot
ddp_data = ocp_utils.extract_ocp_data(ddp, ee_frame_name=ee_frame_name)

#  Warm start using exactly the previous solution
ddp.problem.x0 = x0
xs_init = ddp.xs
us_init = ddp.us 
ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
ddp_data_warm = ocp_utils.extract_ocp_data(ddp, ee_frame_name=ee_frame_name)


# setup visualizer (instead of simulator)
viz = pin.visualize.GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)

viz.viewer.gui.addSphere("world/target", 0.05, [0, 1, 0, 0.5])
viz.viewer.gui.applyConfiguration("world/target", oMe_goal.translation.tolist() + [0, 0, 0, 1])
viz.viewer.gui.addSphere("world/final", 0.05, [0, 0, 1, 0.5])
# solution joint trajectory
xs = np.array(ddp.xs)
q_final = xs[-1, : robot.model.nq]
oMe_fin = robot.framePlacement(
    q_final, robot.model.getFrameId(ee_frame_name), update_kinematics=True
)
viz.viewer.gui.applyConfiguration("world/final", oMe_fin.translation.tolist() + [0, 0, 0, 1])

print("Final - goal placement")
print('translation (mm): ', 1e3*(oMe_fin.translation - oMe_goal.translation))
print('orientation (deg): ', np.rad2deg(pin.log(oMe_goal.rotation.T*oMe_fin.rotation)))

# Display trajectory solution in Gepetto Viewer
display = crocoddyl.GepettoDisplay(robot)
display.displayFromSolver(ddp, factor=1)

fig_d, axes_d = ocp_utils.plot_ocp_results(
    ddp_data,
    which_plots="all",
    labels=None,
    markers=["."],
    colors=["b"],
    sampling_plot=1,
    SHOW=False,
)

ocp_utils.plot_ocp_state(ddp_data_warm, fig_d['x'], axes_d['x'])


