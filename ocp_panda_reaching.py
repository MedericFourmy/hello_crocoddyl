'''
Example script : Crocoddyl OCP with Panda arm 
static target reaching task

Directly inspired by https://github.com/skleff1994/minimal_examples_crocoddyl/blob/master/ocp_kuka_reaching.py
'''

import crocoddyl
import numpy as np
import pinocchio as pin
np.set_printoptions(precision=4, linewidth=180)
import ocp_utils
import time

# Load model (hardcoded for now, eventually should be in example-robot-data)
urdf_path = '/home/mfourmy/catkin_ws/src/panda_torque_mpc/config/panda_inertias_nohand.urdf'
mesh_path = '/home/mfourmy/catkin_ws/src/franka_ros/franka_description/meshes/'
robot = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_path) 
model = robot.model
data = robot.data
# delta_trans = np.array([-0.8,0.2,-0.2])
delta_trans = np.array([-0.0,0.2,-0.2])

# Number of shooting nodes
T = 100
dt = 1e-2  # seconds

# Timing
N_SOLVE = 20

# Main parameters
ee_frame_name = 'panda_link7'
# franka_control/config/start_pose.yaml
q0 = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])

# setup visualizer (instead of simulator)
viz = pin.visualize.GepettoVisualizer(model, robot.collision_model, robot.visual_model)
# viz = pin.visualize.MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True)

# franka_control/config/start_pose.yaml

x0 = np.concatenate([q0, v0])
ee_id = model.getFrameId(ee_frame_name)
pin.framesForwardKinematics(model, data, q0)
oMe_0 = data.oMf[ee_id]

t_oe_goal =  oMe_0.translation + delta_trans 

viz.viewer.gui.addSphere('world/target', 0.05, [0,1,0,0.5])
viz.viewer.gui.applyConfiguration('world/target', t_oe_goal.tolist()+[0,0,0,1])
viz.display(q0)
print(oMe_0)

# # # # # # # # # # # # # # #
###  SETUP CROCODDYL OCP  ###
# # # # # # # # # # # # # # #

"""
  Objects we need to define 

  CostModelResidual defines a cost model as cost = a(r(x,u)) where
  r is a residual function, a is an activation model
  
  The class (as other Residual types) implements:
  - calc: computes the residual function
  - calcDiff: computes the residual derivatives
  Results are stored in a ResidualDataAbstract

  Default activation function is quadratic
"""


# State and actuation model
state = crocoddyl.StateMultibody(model)
actuation = crocoddyl.ActuationModelFull(state)

###################
# Create cost terms
# Control regularization cost: r(x_i, u_i) = tau_i - g(q_i) 
uResidual = crocoddyl.ResidualModelControlGrav(state)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# State regularization cost: r(x_i, u_i) = diff(x_i, x_ref)
xResidual = crocoddyl.ResidualModelState(state, x0)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# end translation cost: r(x_i, u_i) = translation(q_i) - t_ref
frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, ee_id, t_oe_goal)
frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)

# Initialize structures to hold the cost models 
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

#####################
# add the cost models with respective weights
runningCostModel.addCost('stateReg', xRegCost, 1e-1)
runningCostModel.addCost('ctrlRegGrav', uRegCost, 1e-3)
# runningCostModel.addCost('translation', frameTranslationCost, 10)
terminalCostModel.addCost('stateReg', xRegCost, 1e-1)
terminalCostModel.addCost('translation', frameTranslationCost, 10)

# Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)

# Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)

# Create the shooting problem
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
# Create solver + callbacks
ddp = crocoddyl.SolverFDDP(problem)
# ddp.setCallbacks([crocoddyl.CallbackLogger(),
#                   crocoddyl.CallbackVerbose()])


# Warm start : initial state + gravity compensation
xs_init = [x0 for i in range(T+1)]
us_init = ddp.problem.quasiStatic(xs_init[:-1])

# Solve
solve_times = np.zeros(N_SOLVE)
for i in range(N_SOLVE):
  t1 = time.time()
  ddp.solve(xs_init, us_init, maxiter=100, isFeasible=False)
  solve_times[i] = 1e3*(time.time() - t1)

print('Solve time (ms) avg + sig: ', solve_times.mean(), np.sqrt(np.std(solve_times - solve_times.mean())))

# Extract DDP data and plot
ddp_data = ocp_utils.extract_ocp_data(ddp, ee_frame_name=ee_frame_name)

# Display solution in Gepetto Viewer
display = crocoddyl.GepettoDisplay(robot)
display.displayFromSolver(ddp, factor=1)

ocp_utils.plot_ocp_results(ddp_data, which_plots='all', labels=None, markers=['.'], colors=['b'], sampling_plot=1, SHOW=True)


