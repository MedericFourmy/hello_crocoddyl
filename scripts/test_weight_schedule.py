import numpy as np
import matplotlib.pyplot as plt
from ocp_pbe_def import linear_interpolation, tanh_interpolation

T = 100
w_low = 1e-4
w_high = 0.1

x = np.arange(T)
w_linear_schedule = linear_interpolation(x, 0, T-1, w_low, w_high)
# w_tanh_schedule = tanh_interpolation(x, w_low, w_high, 5, 0.5)
w_tanh_schedule = tanh_interpolation(x, w_low, w_high, scale=6, shift=0.0)

plt.figure('linear')
plt.plot(np.arange(T), w_linear_schedule)

plt.figure('tanh')
plt.plot(x, w_tanh_schedule)
plt.hlines([w_low, w_high], 0, T, 'r')
plt.grid()

plt.show()
