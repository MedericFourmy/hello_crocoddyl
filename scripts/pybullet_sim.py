import numpy as np
import pinocchio as pin
import pybullet as pb
import pybullet_data


class PybulletSim:

    def __init__(self, dt_sim, urdf_path, package_dirs, joint_names, base_pose=[0, 0, 0, 0, 0, 0, 1], visual=True):
        """Initializes the wrapper.

        Simplified version of
        https://github.com/machines-in-motion/bullet_utils/blob/main/src/bullet_utils/wrapper.py
        assuming a fixed base.

        According to doc, default simulation timestep is 1/240 and should be kept as is (see pb.setTimeStep method doc).
        """

        # Pinocchio model
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs)
        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.nj = len(joint_names)
        self.pinocchio_robot = self.robot
        self.tau_fext = np.zeros(self.robot.nv)

        # Pybullet setup
        self.physicsClient = pb.connect(pb.GUI if visual else pb.DIRECT)
        pb.setTimeStep(dt_sim)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        planeId = pb.loadURDF("plane.urdf")
        robot_id = pb.loadURDF(urdf_path, base_pose[:3], base_pose[3:])
        self.robot_id = robot_id
        pb.setGravity(*self.robot.model.gravity.linear)

        # Mapping between both models
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
            [self.robot.model.getJointId(name) for name in joint_names]
        )

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

        # ddq??
        return q, dq, np.zeros(self.robot.nv)

    def set_state(self, q, dq):
        """Set the robot to the desired states.
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

    def apply_external_force(self, f, ee_frame, rf_frame=pin.LOCAL_WORLD_ALIGNED):
        # Store the torque due to exterior forces for simulation step

        q = self.get_state()[0]
        self.robot.framesForwardKinematics(q)
        self.robot.computeJointJacobians(q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        Jf = pin.getFrameJacobian(
            self.robot.model, self.robot.data, self.robot.model.getFrameId(ee_frame), rf_frame)
        self.tau_fext = Jf.T @ f

    def send_joint_command(self, tau):
        """Apply the desired torques to the joints.
        Args:
            tau (ndarray): Torque to be applied.
        """
        # TODO: Apply the torques on the base towards the simulator as well.
        assert tau.shape[0] == self.nv

        # Add the torque due to the external force here
        tau += self.tau_fext

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

        # reset external force automatically after each simulation step
        self.tau_fext = np.zeros(self.nv)


if __name__ == '__main__':
    import time
    import pinocchio as pin
    import config_panda as conf

    dur_sim = 10.0

    # Gains are tuned at max before instability for each dt
    dt_sim = 1./240
    Kp = 200
    Kd = 2

    # dt_sim = 1./1000
    # Kp = 200
    # # Kd = 5
    # Kd = 2*np.sqrt(Kp)

    N_sim = int(dur_sim/dt_sim)

    robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)

    sim = PybulletSim(dt_sim, conf.urdf_path, conf.package_dirs,
                      conf.joint_names, visual=True)
    sim.set_state(conf.q0, conf.v0)

    print('conf.q0')
    print(conf.q0)
    for i in range(N_sim):
        ts = i*dt_sim
        t1 = time.time()
        q, v = sim.get_state()

        # Pure feedforward
        # tau = tau_ffs
        # PD
        # tau = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        # PD+
        # tau = tau_ff - Kp*(q - conf.q0) - Kd*(v - conf.v0)

        # Joint Space Inverse Dynamics
        qd = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        tau = pin.rnea(robot.model, robot.data, q, v, qd)
        if 2.0 < ts < 3.0:
            fext = np.array([0, 100, 0, 0, 0, 0])
            # fext = np.array([0,0,0, 0,-6,0])
            sim.apply_external_force(
                fext, "panda_link4", rf_frame=pin.LOCAL_WORLD_ALIGNED)

        sim.send_joint_command(tau)
        sim.step_simulation()

        delay = time.time() - t1
        if delay < dt_sim:
            # print(delay)
            time.sleep(dt_sim - delay)

    pb.disconnect()
