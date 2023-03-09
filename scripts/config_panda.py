import numpy as np

# urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/res/panda_inertias_nohand_copy.urdf"
urdf_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/res/panda_inertias.urdf"
package_dirs = ["/home/mfourmy/catkin_ws/src/franka_ros/"]


ee_name = "panda_link8"
q0 = np.array([0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397])
v0 = np.zeros(7)
x0 = np.concatenate([q0, v0])
