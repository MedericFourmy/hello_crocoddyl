import numpy as np
import pinocchio as pin
import crocoddyl


def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)

def tanh_interpolation(x, low, high, scale, shift=0):
    x_norm = linear_interpolation(x, 0, len(x), scale*(-1 - shift), scale*(1 - shift))
    return low + 0.5*high*(np.tanh(x_norm)+1)

def create_ocp_reaching_pbe(model, x0, ee_frame_name, oMe_goal, T, dt, goal_is_se3=True, verbose=False):
    # # # # # # # # # # # # # # #
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

    ee_frame_id = model.getFrameId(ee_frame_name)

    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)
    

    ###################
    # Create cost terms

    # end translation cost: r(x_i, u_i) = translation(q_i) - t_ref
    frameGoalResidual = None
    if goal_is_se3:
        frameGoalResidual = crocoddyl.ResidualModelFramePlacement(state, ee_frame_id, oMe_goal)
    else:
        frameGoalResidual = crocoddyl.ResidualModelFrameTranslation(state, ee_frame_id, oMe_goal.translation)
    frameGoalCost = crocoddyl.CostModelResidual(state, frameGoalResidual)


    ##############
    # Task weigths
    ##############
    # EE pose
    w_running_frame_low = 0.0
    w_running_frame_high = 0.0
    w_frame_terminal = 1000.0

    # EE vel
    # w_frame_vel_terminal = 0.0
    w_frame_vel_terminal = 10.0
    diag_vel_terminal = np.array(
        3*[1.0] + 3*[1.0]
    )
    
    # State regularization
    w_x_reg_running = 0.01
    # w_x_reg_running = 1.0
    diag_x_reg_running = np.array(
        7*[0.0] + 7*[1.0]
    )
    w_x_reg_terminal = 0.0
    diag_x_reg_terminal = np.array(
        7*[0.0] + 7*[1.0]
    )

    # Control regularization
    w_u_reg_running = 0.01
    diag_u_reg_arr = np.array([
        1, 1, 1, 1, 1, 1, 10
    ])


    # w_frame_schedule = linear_interpolation(np.arange(T), 0, T-1, w_running_frame_low, w_running_frame_high)
    w_frame_schedule = tanh_interpolation(np.arange(T), w_running_frame_low, w_running_frame_high, scale=5, shift=0.0)
    # w_frame_schedule = tanh_interpolation(np.arange(T), w_running_frame_low, w_running_frame_high, scale=8, shift=0.0)

    ###############
    # Running costs
    goal_cost_name = 'placement' if goal_is_se3 else 'translation'
    runningModel_lst = []
    for i in range(T):
        runningCostModel = crocoddyl.CostModelSum(state)

        # State regularization cost: r(x_i, u_i) = diff(x_i, x_ref)
        xRegCost = crocoddyl.CostModelResidual(state, 
                                               crocoddyl.ActivationModelWeightedQuad(diag_x_reg_running**2), 
                                               crocoddyl.ResidualModelState(state, x0, actuation.nu))

        # Control regularization cost: r(x_i, u_i) = tau_i - g(q_i)
        uRegCost = crocoddyl.CostModelResidual(state, 
                                               crocoddyl.ActivationModelWeightedQuad(diag_u_reg_arr**2), 
                                               crocoddyl.ResidualModelControlGrav(state, actuation.nu))


        runningCostModel.addCost('stateReg', xRegCost, w_x_reg_running)
        runningCostModel.addCost('ctrlRegGrav', uRegCost, w_u_reg_running)
        runningCostModel.addCost(goal_cost_name, frameGoalCost, w_frame_schedule[i])
        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        )
        # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
        runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        # Optionally add armature to take into account actuator's inertia
        runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        runningModel_lst.append(runningModel)
    
    ###############
    # Terminal cost
    # !! weights scale has a different meaning here since
    # weights in the running cost are multiplied by dt 
    ###############
    terminalCostModel = crocoddyl.CostModelSum(state)
    xRegCost = crocoddyl.CostModelResidual(state, 
                                        crocoddyl.ActivationModelWeightedQuad(diag_x_reg_terminal**2), 
                                        crocoddyl.ResidualModelState(state, x0, actuation.nu))
    # Control regularization cost: nu(x_i) = v_ee(x_i) - v_ee*
    frameVelCost = crocoddyl.CostModelResidual(state, 
                                            crocoddyl.ActivationModelWeightedQuad(diag_vel_terminal**2), 
                                            crocoddyl.ResidualModelFrameVelocity(state, ee_frame_id, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, actuation.nu))

    terminalCostModel.addCost('stateReg', xRegCost, w_x_reg_terminal)
    terminalCostModel.addCost(goal_cost_name, frameGoalCost, w_frame_terminal)
    terminalCostModel.addCost('terminal_vel', frameVelCost, w_frame_vel_terminal)


    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCostModel
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)
    # Optionally add armature to take into account actuator's inertia
    terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Create the shooting problem
    problem = crocoddyl.ShootingProblem(x0, runningModel_lst, terminalModel)

    # Create solver + callbacks
    ddp = crocoddyl.SolverFDDP(problem)
    if verbose:
        ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    return ddp
