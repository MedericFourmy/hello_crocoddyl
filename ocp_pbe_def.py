import crocoddyl
import numpy as np
import pinocchio as pin


def linear_interpolation(x, x1, x2, y1, y2):
    return y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)

def tanh_interpolation(x, low, high, scale, shift=0):
    x_norm = linear_interpolation(x, 0, len(x), scale*(-1 - shift), scale*(1 - shift))
    return low + high*(np.tanh(x_norm)+1) 

def create_ocp_reaching_pbe(model, x0, ee_frame_name, oMe_goal, N_nodes, dt, goal_is_se3=True, verbose=False):
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

    # State and actuation model
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFull(state)

    ###################
    # Create cost terms

    # end translation cost: r(x_i, u_i) = translation(q_i) - t_ref
    frameGoalResidual = None
    if goal_is_se3:
        frameGoalResidual = crocoddyl.ResidualModelFramePlacement(state, model.getFrameId(ee_frame_name), oMe_goal)
    else:
        frameGoalResidual = crocoddyl.ResidualModelFrameTranslation(state, model.getFrameId(ee_frame_name), oMe_goal.translation)
    frameGoalCost = crocoddyl.CostModelResidual(state, frameGoalResidual)


    # State regularization cost: r(x_i, u_i) = diff(x_i, x_ref)
    xRegCost = crocoddyl.CostModelResidual(state,  
                                           crocoddyl.ResidualModelState(state, x0))

    # Control regularization cost: r(x_i, u_i) = tau_i - g(q_i)
    uRegCost = crocoddyl.CostModelResidual(state, 
                                           crocoddyl.ResidualModelControlGrav(state, actuation.nu))



    # WEIGTHS
    w_x_reg = 1e-1
    w_x_reg_arr = w_x_reg*np.array([
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
    ])
    w_u_reg = 1e-3

    w_u_reg_arr = w_u_reg*np.array([
        10, 1, 1, 1, 1, 1, 1
    ])
    
    w_frame_low = 0.0001
    w_frame_high = 10



    w_frame_schedule = linear_interpolation(np.arange(N_nodes), 0, N_nodes-1, w_frame_low, w_frame_high)
    # w_frame_schedule = tanh_interpolation(np.arange(N_nodes), w_frame_low, w_frame_high, 5, 0)
    # w_frame_schedule = tanh_interpolation(np.arange(N_nodes), w_frame_low, w_frame_high, 8, 0.5)


    print('w_frame_schedule[:5]: ', w_frame_schedule[:5])
    print('w_frame_schedule[-5:]: ', w_frame_schedule[-5:])

    ###############
    # Running costs
    goal_cost_name = 'placement' if goal_is_se3 else 'translation'
    runningModel_lst = []
    for i in range(N_nodes):
        runningCostModel = crocoddyl.CostModelSum(state)

        # State regularization cost: r(x_i, u_i) = diff(x_i, x_ref)
        xRegCost = crocoddyl.CostModelResidual(state, 
                                               crocoddyl.ActivationModelWeightedQuad(w_x_reg_arr**2), 
                                               crocoddyl.ResidualModelState(state, x0, actuation.nu))

        # Control regularization cost: r(x_i, u_i) = tau_i - g(q_i)
        uRegCost = crocoddyl.CostModelResidual(state, 
                                               crocoddyl.ActivationModelWeightedQuad(w_u_reg_arr**2), 
                                               crocoddyl.ResidualModelControlGrav(state, actuation.nu))


        runningCostModel.addCost("stateReg", xRegCost, 1)
        runningCostModel.addCost("ctrlRegGrav", uRegCost, 1)
        runningCostModel.addCost(goal_cost_name, frameGoalCost, w_frame_schedule[i])
        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        )
        # Create Integrated Action Model (IAM), i.e. Euler integration of continuous dynamics and cost
        runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        # Optionally add armature to take into account actuator's inertia
        runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
        # runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        runningModel_lst.append(runningModel)
    
    ###############
    # Terminal cost
    terminalCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel.addCost("stateReg", xRegCost, w_x_reg)
    terminalCostModel.addCost("ctrlRegGrav", uRegCost, w_u_reg)
    terminalCostModel.addCost(goal_cost_name, frameGoalCost, w_frame_high)

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
