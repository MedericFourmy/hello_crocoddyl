import pinocchio as pin
import config_panda as conf 

robot = pin.RobotWrapper.BuildFromURDF(conf.urdf_path, conf.package_dirs)
from example_robot_data import load
robot_erd = load('panda')

# Compute something to update the data.mass vector
robot.com(robot.q0)
robot_erd.com(robot_erd.q0)
print('robot tot mass: ', robot.data.mass[0])
print('robot_erd tot mass: ', robot_erd.data.mass[0])


print('Robot inertias.mass vs model.mass')
for i, iner in enumerate(robot.model.inertias):
    print(f'iner {i}', iner.mass)
for i, m in enumerate(robot.data.mass):
    print(f'm {i}', m)
print()
print('Robot erd inertias.mass vs model.mass')
for i, iner in enumerate(robot_erd.model.inertias):
    print(f'iner {i}', iner.mass)
for i, m in enumerate(robot_erd.data.mass):
    print(f'm {i}', m)


for i, (m, m_er) in enumerate(zip(robot.data.mass, robot_erd.data.mass)):
    print('\nDiff mass id ', i)
    print('Mass')
    print(m - m_er)

print("\n\n\n robot_erd")
for j in range(i+1, len(robot_erd.data.mass)):
    print('\nIner id ', j)
    iner = robot_erd.model.inertias[j]
    print('Mass: ', robot_erd.data.mass[j])

print(len(robot.model.inertias))
print(len(robot_erd.model.inertias))


for i, (iner, iner_erd) in enumerate(zip(robot.model.inertias, robot_erd.model.inertias)):
    print('\nDiff Iner id ', i)
    print('Mass: ', iner.mass - iner_erd.mass)
    print('lever: ', iner.lever - iner_erd.lever)
    print('Rot Inertia')
    print(iner.inertia - iner_erd.inertia)

print("\n\n\n robot_erd")
for j in range(i+1, len(robot_erd.model.inertias)):
    print('\nIner id ', j)
    iner = robot_erd.model.inertias[j]
    print('Mass: ', iner_erd.mass)
    print('lever: ', iner_erd.lever)
    print('Rot Inertia')
    print(iner_erd.inertia)
