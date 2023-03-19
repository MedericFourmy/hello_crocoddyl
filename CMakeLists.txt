cmake_minimum_required(VERSION 3.10)
project(hello_crocoddyl)



find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(crocoddyl REQUIRED)
find_package(yaml-cpp REQUIRED)


add_executable(main_reaching src/main_reaching.cpp)

target_link_libraries(main_reaching 
    Eigen3::Eigen
    pinocchio::pinocchio pinocchio::pinocchio_headers
    crocoddyl::crocoddyl
    yaml-cpp
)
