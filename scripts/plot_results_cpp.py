import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BUILD_DIR = 'build_multi'
df_T = pd.read_csv(f'../{BUILD_DIR}/T.csv')
df_q = pd.read_csv(f'../{BUILD_DIR}/q.csv')
df_tau = pd.read_csv(f'../{BUILD_DIR}/tau.csv')
df_v = pd.read_csv(f'../{BUILD_DIR}/v.csv')

t_arr = np.arange(len(df_q))*1e-3

df_q.plot()
plt.title('configurations (rad)')
plt.xlabel('iterations')
plt.legend()
plt.grid()

df_v.plot()
plt.title('velocities (rad/s)')
plt.xlabel('iterations')
plt.legend()
plt.grid()

fig, axes = plt.subplots(2,1)
axes[0].set_title('translation (m)')
axes[1].set_title('orientation (deg)')
for i in range(3):
    k = 'xyz'[i]
    c = 'rgb'[i]
    axes[0].plot(t_arr, df_T[f't{k}'], f'{c}', label=f't{k}')
    axes[0].plot(t_arr, df_T[f't{k}_r'], f'{c}--', label=f't{k}_r')
    axes[1].plot(t_arr, np.rad2deg(df_T[f'o{k}']), f'{c}', label=f'o{k}')
    axes[1].plot(t_arr, np.rad2deg(df_T[f'o{k}_r']), f'{c}--', label=f'o{k}_r')

plt.xlabel('iterations')
plt.legend()
plt.grid()


df_tau.plot()
plt.title('torques (N.m)')
plt.xlabel('iterations')
plt.legend()
plt.grid()




plt.show()