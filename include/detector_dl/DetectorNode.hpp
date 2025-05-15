#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "rclcpp/rclcpp.hpp"

#include <opencv2/opencv.hpp>
#include "cv_bridge/cv_bridge.h"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

#include "tf2/convert.h"
#include "geometry_msgs/msg/quaternion.hpp"

#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"

#include "detector_dl/Detector.hpp"
#include "detector_dl/Monocular.hpp"
#include "detector_dl/Quaternion.hpp"

class DetectorDlNode : public rclcpp::Node
{
private:
    // 识别器
    std::unique_ptr<Detector> detector_;

    // 单目解算（PNP）
    std::unique_ptr<Monocular> monocular_;

    // 相机信息订阅
    sensor_msgs::msg::CameraInfo::SharedPtr cam_info_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;

    // 源图像订阅
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

    // Detected armor publisher
    rclcpp::Publisher<autoaim_interfaces::msg::Armors>::SharedPtr armors_pub_;

public:
    explicit DetectorDlNode(const rclcpp::NodeOptions &options)
        : Node("detector_dl_node", options),
          detector_(initDetector()),
          armors_pub_(create_publisher<autoaim_interfaces::msg::Armors>(
              "/detector/armors", rclcpp::SensorDataQoS())),
          cam_info_sub_(create_subscription<sensor_msgs::msg::CameraInfo>(
              "/camera_info", rclcpp::SensorDataQoS(),
              [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
              {
                  cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
                  monocular_ = std::make_unique<Monocular>(camera_info->k, camera_info->d);
                  cam_info_sub_.reset(); // 停止接收
              })),
          img_sub_(create_subscription<sensor_msgs::msg::Image>(
              "/image_raw", rclcpp::SensorDataQoS(),
              std::bind(&DetectorDlNode::imageCallback, this, std::placeholders::_1)))

    {
        RCLCPP_INFO(get_logger(), "The detector_dl_node has been started!");
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
    {
        auto armors = detectArmors(img_msg);
        autoaim_interfaces::msg::Armor armor_msg;
        // 当无装甲板，则不进行下一步
        if (armors.empty())
            return;
        // 当单目PnP生成成功
        if (monocular_)
        {
            // 新增：对所有识别到的装甲板进行pnp解算，发送到robot_tracker
            autoaim_interfaces::msg::Armors armors_msg;
            armors_msg.header = img_msg->header;
            armors_msg.header.frame_id = "camera";
            armors_msg.armors.reserve(armors.size());

            // 对所有装甲板进行pnp解算
            for (const auto &armor : armors)
            {
                auto [rVec, tVec, distance_to_image_center] = monocular_->PnP_solver(armor);
                armor_msg.type = static_cast<int>(armor.type);
                // armor_msg.number = "0"; // 需要实装数字检测
                armor_msg.number = armor.number;

                // 对获得的rvec tvec对应的坐标系做旋转变换
                cv::Mat R_temp, T_temp;
                cv::Rodrigues(rVec, R_temp);
                
                // 让x轴朝前
                static const cv::Matx33d R_x(0, 0, 1, -1, 0, 0, 0, -1, 0);
                R_temp = R_x * R_temp;
                T_temp = R_x * tVec;

                armor_msg.pose.position.x = T_temp.at<double>(0);
                armor_msg.pose.position.y = T_temp.at<double>(1);
                armor_msg.pose.position.z = T_temp.at<double>(2);

                // 旋转矩阵转四元数
                Quaternion<double> q(R_temp);
                armor_msg.pose.orientation = q.toMsg();
                armors_msg.armors.emplace_back(armor_msg);
            }
            armors_pub_->publish(armors_msg);
        }
        else
        {
            RCLCPP_ERROR(get_logger(), "PnP init failed!");
        }
    }

    std::unique_ptr<Detector> initDetector()
    {
        // 初始化参数
        int NUM_CLASSES = declare_parameter("4P_NUM_CLASSES", 36);
        std::string TARGET_COLOUR = declare_parameter("4P_TARGET_COLOUR", "RED");
        float NMS_THRESH = declare_parameter("4P_NMS_THRESH", 0.45);
        float BBOX_CONF_THRESH = declare_parameter("4P_BBOX_CONF_THRESH", 0.5);
        int INPUT_W = declare_parameter("4P_INPUT_W", 448);
        int INPUT_H = declare_parameter("4P_INPUT_H", 448);
        std::string engine_file_path = declare_parameter(
            "4P_engine_file_path",
            "/home/nvidia/mosas_autoaim_dl/src/detector_dl/model/"
            "RM4points_v3_5_sim_448x448.trt");

        // 装甲板限定阈值
        float min_large_center_distance = declare_parameter("min_large_center_distance", 3.2);

        // 初始化识别器
        return std::make_unique<Detector>(
            NUM_CLASSES, TARGET_COLOUR, NMS_THRESH,
            BBOX_CONF_THRESH, INPUT_W, INPUT_H,
            engine_file_path, min_large_center_distance);
    }

    std::vector<Armor> detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
    {
        // 转换为Mat
        auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
        auto start_time = this->now();
        // 更新参数（如果改变）
        detector_->NUM_CLASSES = get_parameter("4P_NUM_CLASSES").as_int();
        detector_->TARGET_COLOUR = get_parameter("4P_TARGET_COLOUR").as_string();
        detector_->NMS_THRESH = get_parameter("4P_NMS_THRESH").as_float();
        detector_->BBOX_CONF_THRESH = get_parameter("4P_BBOX_CONF_THRESH").as_float();
        detector_->INPUT_W = get_parameter("4P_INPUT_W").as_int();
        detector_->INPUT_H = get_parameter("4P_INPUT_H").as_int();
        detector_->engine_file_path = get_parameter("4P_engine_file_path").as_string();
        // 开始识别
        // show image在detect实现里
        auto armors = detector_->detect(img);
        // 计算每张延时与FPS
        auto final_time = this->now();
        auto latency = (final_time - start_time).seconds() * 1000;
        RCLCPP_INFO(get_logger(), "Latency: %lfms", latency);
        RCLCPP_INFO(get_logger(), "FPS: %lf", 1000 / latency);

        // 绘出图像并显示
        // cv::resize(img, img, {640, 480});
        // cv::imshow("result", img);
        // cv::waitKey(1);

        return armors;
    }
};
