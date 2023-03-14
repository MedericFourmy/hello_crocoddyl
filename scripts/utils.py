import time
import numpy as np
import pinocchio as pin
import config_panda as conf


def test_run_simulator(Simulator):

    dur_sim = 5.0

    # force disturbance
    t1_fext, t2_fext = 2.0, 3.0
    fext = np.array([0, 100, 0, 0, 0, 0])

    # Gains are tuned at max before instability for each dt
    # dt_sim = 1./240
    # Kp = 200
    # Kd = 2

    dt_sim = 1./1000
    Kp = 200
    # Kd = 5
    Kd = 2*np.sqrt(Kp)


    N_sim = int(dur_sim/dt_sim)

    robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)

    sim = Simulator(dt_sim, conf.urdf_path, conf.package_dirs,
                    conf.joint_names, visual=True)
    sim.set_state(conf.q0+0.1, conf.v0)

    for i in range(N_sim):
        ts = i*dt_sim
        t1 = time.time()
        q, v, dv = sim.get_state()

        # Pure feedforward
        # tau = tau_ffs
        # PD
        # tau = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        # PD+
        # tau = tau_ff - Kp*(q - conf.q0) - Kd*(v - conf.v0)

        # Joint Space Inverse Dynamics
        qd = - Kp*(q - conf.q0) - Kd*(v - conf.v0)
        tau = pin.rnea(robot.model, robot.data, q, v, qd)
        if t1_fext < ts < t2_fext:
            sim.apply_external_force(
                fext, "panda_link4", rf_frame=pin.LOCAL_WORLD_ALIGNED)

        sim.send_joint_command(tau)
        sim.step_simulation()

        delay = time.time() - t1
        if delay < dt_sim:
            # print(delay)
            time.sleep(dt_sim - delay)

