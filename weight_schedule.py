import numpy as np
import matplotlib.pyplot as plt
from ocp_pbe_def import linear_interpolation, tanh_interpolation

N_nodes = 100
w_frame_low = 0.0001
w_frame_high = 10

x = np.arange(N_nodes)
w_linear_schedule = linear_interpolation(x, 0, N_nodes-1, w_frame_low, w_frame_high)
w_tanh_schedule = tanh_interpolation(x, w_frame_low, w_frame_high, xb=5)

plt.figure('linear')
plt.plot(np.arange(N_nodes), w_linear_schedule)

plt.figure('tanh')
plt.plot(x, w_tanh_schedule)

plt.show()