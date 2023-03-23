import pinocchio as pin

def freezed_robot(robot, fixed_joints):
    
    # Remove some joints from pinocchio model
    fixed_ids = [robot.model.getJointId(jname) for jname in fixed_joints] \
                if fixed_joints is not None else []
    # Ugly code to resize model and q0
    rmodel, [gmodel_col, gmodel_vis] = pin.buildReducedModel(
            robot.model, [robot.collision_model, robot.visual_model],
            fixed_ids, robot.q0,
        )
    robot = pin.RobotWrapper(rmodel, gmodel_col, gmodel_vis)
    robot.q0 = robot.model.referenceConfigurations['default']

    return robot
