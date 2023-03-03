#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <iostream>
#include <fstream>

#include "crocoddyl_reaching.h"

namespace pin = pinocchio;


int main()
{

    std::string ee_frame_pin = "panda_link8";

    pin::Model model_pin;
    std::string model_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/config/panda_inertias_nohand.urdf";
    pin::urdf::buildModel(model_path, model_pin);
    pin::Data data_pin(model_pin);


    CrocoddylConfig config;
    Eigen::Vector3d delta_trans; delta_trans << -0.3, -0.3, -0.0;  
    // Number of shooting nodes (minus terminal one)
    config.T = 200;
    config.dt_ocp = 1e-2;

    // franka_control/config/start_pose.yaml
    Eigen::Matrix<double,7,1> q0; q0 << 0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397;
    Eigen::Matrix<double,7,1> v0 = Eigen::Matrix<double,7,1>::Zero();
    Eigen::Matrix<double,14,1> x0; x0 << q0, v0;
    config.x0 = x0;

    config.ee_frame_name = "panda_link8";
    pin::forwardKinematics(model_pin, data_pin, q0);
    pin::updateFramePlacements(model_pin, data_pin);
    pin::SE3 T_ee0 = data_pin.oMf[model_pin.getFrameId(config.ee_frame_name)];
    config.goal_trans = T_ee0.translation() + delta_trans;


    config.armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;


    CrocoddylReaching croco_pbe(model_pin, config);

    // Warm start : initial state + gravity compensation
    std::vector<Eigen::Matrix<double,-1,1>> xs_init; 
    std::vector<Eigen::Matrix<double,-1,1>> us_init; 
    for (int i=0; i < config.T; i++)
    {
        xs_init.push_back(x0);
        us_init.push_back(pin::computeGeneralizedGravity(model_pin, data_pin, q0));
    }

    // Terminal node
    xs_init.push_back(x0);

    std::cout << "ddp problem initialized " << std::endl;
    croco_pbe.ddp_->solve(xs_init, us_init, 10, false);
    std::cout << "ddp problem solved " << std::endl;

    // croco_pbe.sol


    

}
