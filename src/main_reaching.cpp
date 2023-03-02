#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <iostream>
#include <fstream>

namespace pin = pinocchio;


int main()
{

    std::string ee_frame_pin = "panda_link8";

    pin::Model model_pin;
    std::string model_path = "/home/mfourmy/catkin_ws/src/panda_torque_mpc/config/panda_inertias_nohand.urdf";
    pin::urdf::buildModel(model_path, model_pin);
    pin::Data data_pin(model_pin);


}
