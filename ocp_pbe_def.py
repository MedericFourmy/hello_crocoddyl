import crocoddyl



def create_ocp_reaching_pbe(model, x0, ee_frame_name, t_oe_goal, T, dt, verbose=False):

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
  frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, model.getFrameId(ee_frame_name), t_oe_goal)
  frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)

  # Initialize structures to hold the cost models 
  runningCostModel = crocoddyl.CostModelSum(state)
  terminalCostModel = crocoddyl.CostModelSum(state)

  #####################
  # add the cost models with respective weights
  runningCostModel.addCost('stateReg', xRegCost, 1e-1)
  runningCostModel.addCost('ctrlRegGrav', uRegCost, 1e-4)
  runningCostModel.addCost('translation', frameTranslationCost, 10)
  terminalCostModel.addCost('stateReg', xRegCost, 1e-3)
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
  if verbose:
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

  return ddp

