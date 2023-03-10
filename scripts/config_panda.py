import numpy as np

# urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/res/panda_inertias_nohand_copy.urdf"
urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/res/panda_inertias.urdf"
package_dirs = ["/home/mfourmy/catkin_ws/src/franka_ros/"]


ee_name = "panda_link8"

joint_names = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]
q0 = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])
