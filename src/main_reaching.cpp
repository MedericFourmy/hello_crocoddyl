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
    std::string model_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/res/panda_inertias.urdf";
    // std::string model_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/res/panda_inertias_nohand.urdf";
    pin::urdf::buildModel(model_path, model_pin);
    pin::Data data_pin(model_pin);

    CrocoddylConfig config;
    Eigen::Vector3d delta_trans;
    delta_trans << -0.3, -0.3, -0.0;
    // Number of shooting nodes (minus terminal one)
    config.T = 200;
    config.dt_ocp = 1e-2;

    // franka_control/config/start_pose.yaml
    Eigen::Matrix<double, 7, 1> q0;
    q0 << 0, -0.785398163397, 0, -2.35619449019, 0, 1.57079632679, 0.785398163397;
    Eigen::Matrix<double, 7, 1> v0 = Eigen::Matrix<double, 7, 1>::Zero();
    Eigen::Matrix<double, 14, 1> x0;
    x0 << q0, v0;
    config.x0 = x0;

    config.ee_frame_name = "panda_link8";
    pin::forwardKinematics(model_pin, data_pin, q0);
    pin::updateFramePlacements(model_pin, data_pin);
    pin::SE3 T_ee0 = data_pin.oMf[model_pin.getFrameId(config.ee_frame_name)];
    Eigen::Vector3d goal_trans = T_ee0.translation() + delta_trans;

    config.armature << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;

    CrocoddylReaching croco_pbe(model_pin, config);

    croco_pbe.set_goal_translation(goal_trans);

    // Warm start : initial state + gravity compensation
    std::vector<Eigen::Matrix<double, -1, 1>> xs_init;
    std::vector<Eigen::Matrix<double, -1, 1>> us_init;
    for (int i = 0; i < config.T; i++)
    {
        xs_init.push_back(x0);
        us_init.push_back(pin::computeGeneralizedGravity(model_pin, data_pin, q0));
    }

    // Terminal node
    xs_init.push_back(x0);

    std::cout << "ddp problem initialized " << std::endl;
    croco_pbe.ddp_->solve(xs_init, us_init, 100, false);
    std::cout << "ddp problem solved, nb iterations: " << croco_pbe.ddp_->get_iter() << std::endl;

    std::vector<Eigen::Matrix<double, -1, 1>> xs = croco_pbe.ddp_->get_xs();
    std::vector<Eigen::Matrix<double, -1, 1>> us = croco_pbe.ddp_->get_us();

    croco_pbe.ddp_->solve(xs, us, 100, false);
    std::cout << "ddp problem solved with WS, nb iterations: " << croco_pbe.ddp_->get_iter() << std::endl;

    // Record results

    std::ofstream file_q;
    std::ofstream file_v;
    std::ofstream file_tau;
    std::ofstream file_T;
    file_q.open("tsid_out_q.csv");
    file_v.open("tsid_out_v.csv");
    file_tau.open("tsid_out_tau.csv");
    file_T.open("tsid_out_T.csv");

    file_q << "q0,q1,q2,q3,q4,q5,q6"
           << "\n";
    file_v << "v0,v1,v2,v3,v4,v5,v6"
           << "\n";
    file_tau << "tau0,tau1,tau2,tau3,tau4,tau5,tau6"
             << "\n";
    file_T << "tx,ty,tz,ox,oy,oz,tx_r,ty_r,tz_r,ox_r,oy_r,oz_r"
           << "\n";

    for (int i = 0; i < config.T; i++)
    {

        pin::forwardKinematics(model_pin, data_pin, xs[i].head<7>());
        pin::updateFramePlacements(model_pin, data_pin);
        pin::SE3 T_ee = data_pin.oMf[model_pin.getFrameId(ee_frame_pin)];

        file_q << xs[i](0) << ","
               << xs[i](1) << ","
               << xs[i](2) << ","
               << xs[i](3) << ","
               << xs[i](4) << ","
               << xs[i](5) << ","
               << xs[i](6) << "\n";

        file_v << xs[i](7 + 0) << ","
               << xs[i](7 + 1) << ","
               << xs[i](7 + 2) << ","
               << xs[i](7 + 3) << ","
               << xs[i](7 + 4) << ","
               << xs[i](7 + 5) << ","
               << xs[i](7 + 6) << "\n";

        file_tau << us[i](0) << ","
                 << us[i](1) << ","
                 << us[i](2) << ","
                 << us[i](3) << ","
                 << us[i](4) << ","
                 << us[i](5) << ","
                 << us[i](6) << "\n";

        // Eigen::Vector3d aa_ee = pin::log3(T_ee.rotation());
        // Eigen::Vector3d aa_r = pin::log3(x_r.rotation());

        file_T << T_ee.translation()(0) << ","
               << T_ee.translation()(1) << ","
               << T_ee.translation()(2) << ","
               << 0.0 << ","
               << 0.0 << ","
               << 0.0 << ","
               << goal_trans(0) << ","
               << goal_trans(1) << ","
               << goal_trans(2) << ","
               << 0.0 << ","
               << 0.0 << ","
               << 0.0 << "\n";
    }

    std::cout << "Files recorded" << std::endl;
}

void record_results()
{
}