import numpy as np
import pybullet as pb

"""
Simplified version of
https://github.com/machines-in-motion/bullet_utils/blob/main/src/bullet_utils/wrapper.py
assuming a fixed base.

According to doc, default simulation timestep is 1/240 and should be kept as is (see pb.setTimeStep method doc).
"""


class PybulletSim:

    def __init__(self, robot_id, pinocchio_robot, joint_names):
        """Initializes the wrapper.
        Args:
            robot_id (int): PyBullet id of the robot.
            pinocchio_robot (:obj:'Pinocchio.RobotWrapper'): Pinocchio RobotWrapper for the robot.
            joint_names (:obj:`list` of :obj:`str`): Names of the joints.
            useFixedBase (bool, optional): Determines if the robot base if fixed.. Defaults to False.
        """
        self.nq = pinocchio_robot.nq
        self.nv = pinocchio_robot.nv
        self.nj = len(joint_names)
        self.robot_id = robot_id
        self.pinocchio_robot = pinocchio_robot

        self.joint_names = joint_names

        bullet_joint_map = {}
        for ji in range(pb.getNumJoints(robot_id)):
            bullet_joint_map[
                pb.getJointInfo(robot_id, ji)[1].decode("UTF-8")
            ] = ji

        self.bullet_joint_ids = np.array(
            [bullet_joint_map[name] for name in joint_names]
        )
        self.pinocchio_joint_ids = np.array(
            [pinocchio_robot.model.getJointId(name) for name in joint_names]
        )

        print('bullet_joint_map')
        print(bullet_joint_map)
        print('self.bullet_joint_ids')
        print(self.bullet_joint_ids)
        print('self.pinocchio_joint_ids')
        print(self.pinocchio_joint_ids)

        self.pin2bullet_joint_only_array = []
        # skip the universe joint
        for i in range(1, self.nj + 1):
            self.pin2bullet_joint_only_array.append(
                np.where(self.pinocchio_joint_ids == i)[0][0]
            )

        # Disable the velocity control on the joints as we use torque control.
        pb.setJointMotorControlArray(
            robot_id,
            self.bullet_joint_ids,
            pb.VELOCITY_CONTROL,
            forces=np.zeros(self.nj),
        )

    def get_state(self):
        """Returns a pinocchio-like representation of the q, dq matrices. Note that the base velocities are expressed in the base frame.
        Returns:
            ndarray: Generalized positions.
            ndarray: Generalized velocities.
        """

        q = np.zeros(self.nq)
        dq = np.zeros(self.nv)

        # Query the joint readings.
        joint_states = pb.getJointStates(self.robot_id, self.bullet_joint_ids)

        for i in range(self.nj):
            q[self.pinocchio_joint_ids[i] - 1] = joint_states[i][0]
            dq[self.pinocchio_joint_ids[i] - 1] = joint_states[i][1]

        return q, dq

    def update_pinocchio(self, q, dq):
        """Updates the pinocchio robot.
        This includes updating:
        - kinematics
        - joint and frame jacobian
        - centroidal momentum
        Args:
          q: Pinocchio generalized position vector.
          dq: Pinocchio generalize velocity vector.
        """
        self.pinocchio_robot.computeJointJacobians(q)
        self.pinocchio_robot.framesForwardKinematics(q)
        self.pinocchio_robot.centroidalMomentum(q, dq)

    def get_state_update_pinocchio(self):
        """Get state from pybullet and update pinocchio robot internals.
        This gets the state from the pybullet simulator and forwards
        the kinematics, jacobians, centroidal moments on the pinocchio robot
        (see forward_pinocchio for details on computed quantities)."""
        q, dq = self.get_state()
        self.update_pinocchio(q, dq)
        return q, dq

    def reset_state(self, q, dq):
        """Reset the robot to the desired states.
        Args:
            q (ndarray): Desired generalized positions.
            dq (ndarray): Desired generalized velocities.
        """

        for i, bullet_joint_id in enumerate(self.bullet_joint_ids):
            pb.resetJointState(
                self.robot_id,
                bullet_joint_id,
                q[self.pinocchio_joint_ids[i] - 1],
                dq[self.pinocchio_joint_ids[i] - 1],
            )

    def send_joint_command(self, tau):
        """Apply the desired torques to the joints.
        Args:
            tau (ndarray): Torque to be applied.
        """
        # TODO: Apply the torques on the base towards the simulator as well.
        assert tau.shape[0] == self.nv

        zeroGains = tau.shape[0] * (0.0,)

        pb.setJointMotorControlArray(
            self.robot_id,
            self.bullet_joint_ids,
            pb.TORQUE_CONTROL,
            forces=tau[self.pin2bullet_joint_only_array],
            positionGains=zeroGains,
            velocityGains=zeroGains,
        )

    def step_simulation(self):
        """Step the simulation forward."""
        pb.stepSimulation()



if __name__ == '__main__':

    import time
    import pybullet_data
    import pinocchio as pin
    import config_panda as conf
    physicsClient = pb.connect(pb.GUI)
    # physicsClient = pb.connect(pb.DIRECT)  #or pb.DIRECT for non-graphical version
    print(pybullet_data.getDataPath())
    pb.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    # pb.setGravity(0,0,-10)
    planeId = pb.loadURDF("plane.urdf")
    # startPos = [0,0,1]
    # startPos = [0,0,1]
    # startOrientation = pb.getQuaternionFromEuler([0,0,0])
    print(conf.urdf_path)
    robot_id = pb.loadURDF(conf.urdf_path, [0,0,1], [0,0,0,1])
    # robot_id = pb.loadURDF(conf.urdf_path, [0,0,2], [1,0,0,0])

    # Load model (hardcoded for now, eventually should be in example-robot-data)
    robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)
    pb.setGravity(*robot.model.gravity.linear)

    joint_names = [
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
    ]
    pbs = PybulletSim(robot_id, robot, joint_names)

    pbs.reset_state(conf.q0, conf.v0)
    q, v = conf.q0, conf.v0

    # Gains are tuned at max before instability for each dt
    # dt_sim = 1./240
    # Kp = 200
    # Kd = 2
    
    dt_sim = 1./1000
    Kp = 1000
    Kd = 9
    print('conf.q0')
    print(conf.q0)
    pb.setTimeStep(dt_sim)
    for i in range (50000):
        t1 = time.time()
        q, v = pbs.get_state()
        
        # Gravity compensation feedforward
        # tau_ff = robot.gravity(q)
        # Gravity compensation + coriolis
        tau_ff = pin.rnea(robot.model, robot.data, q, v, np.zeros(7))

        # Pure feedforward
        # tau = tau_ff
        # PD
        tau = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        # PD+
        # tau = tau_ff - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        
        pbs.send_joint_command(tau)
        pb.stepSimulation()
        delay = time.time() - t1
        if delay < dt_sim:
            print(delay)
            time.sleep(dt_sim - delay)
    pb.disconnect()
