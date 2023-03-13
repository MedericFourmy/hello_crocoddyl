import numpy as np
import pinocchio as pin


class PinocchioSim:

    def __init__(self, dt_sim, urdf_path, package_dirs, joint_names, base_pose=[0, 0, 0, 0, 0, 0, 1], visual=True):

        self.dt_sim = dt_sim
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs)
        self.visual = visual
        if visual:
            self.robot.initViewer(loadModel=True)

        self.nq = self.robot.nq
        self.nv = self.robot.nv
        self.q = np.zeros(self.robot.nq)
        self.dq = np.zeros(self.robot.nv)
        self.ddq = np.zeros(self.robot.nv)
        self.tau_cmd = np.zeros(self.robot.nv)
        self.tau_fext = np.zeros(self.robot.nv)

    def get_state(self):
        return self.q, self.dq, self.ddq

    def set_state(self, q, dq, ddq=None):
        self.q = q
        self.dq = dq
        if ddq is None:
            self.ddq = np.zeros(self.nv)

    def apply_external_force(self, f, ee_frame, rf_frame=pin.LOCAL_WORLD_ALIGNED):
        # Store the torque due to exterior forces for simulation step

        self.robot.framesForwardKinematics(self.q)
        self.robot.computeJointJacobians(self.q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        Jf = pin.getFrameJacobian(
            self.robot.model, self.robot.data, self.robot.model.getFrameId(ee_frame), rf_frame)
        self.tau_fext = Jf.T @ f

    def send_joint_command(self, tau):
        """Apply the desired torques to the joints.
        Args:
            tau (ndarray): Torque to be applied.
        """
        assert tau.shape[0] == self.nv
        self.tau_cmd = tau

    def step_simulation(self):
        """Step the simulation forward."""

        # Free foward dynamics algorithm
        tau_ext = self.tau_cmd + self.tau_fext
        self.ddq = pin.aba(self.robot.model, self.robot.data,
                           self.q, self.dq, tau_ext)

        # Integration step
        v_mean = self.dq + 0.5*self.ddq*self.dt_sim
        self.dq += self.ddq*self.dt_sim
        self.q = pin.integrate(self.robot.model, self.q, v_mean*self.dt_sim)

        # update visual
        if self.visual:
            self.robot.display(self.q)

        # reset external force automatically after each simulation step
        self.tau_fext = np.zeros(self.nv)


if __name__ == '__main__':
    import time
    import config_panda as conf

    dur_sim = 10.0

    # Gains are tuned at max before instability for each dt
    # dt_sim = 1./240
    # Kp = 200
    # # Kd = 2
    # Kd = 2*np.sqrt(Kp)

    dt_sim = 1./1000
    Kp = 200
    # Kd = 5
    Kd = 2*np.sqrt(Kp)

    N_sim = int(dur_sim/dt_sim)

    robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)

    sim = PinocchioSim(dt_sim, conf.urdf_path,
                       conf.package_dirs, conf.joint_names, visual=True)
    sim.set_state(conf.q0, conf.v0)

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
            fext = np.array([0, 20, 0, 0, 0, 0])
            sim.apply_external_force(
                fext, "panda_link4", rf_frame=pin.LOCAL_WORLD_ALIGNED)

        sim.send_joint_command(tau)
        sim.step_simulation()

        delay = time.time() - t1
        if delay < dt_sim:
            # print(delay)
            time.sleep(dt_sim - delay)
