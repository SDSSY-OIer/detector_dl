#include "bits/stdc++.h"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
#include "detector_dl/Armor.hpp"
#include "detector_dl/Detector.hpp"
#include "detector_dl/Monocular.hpp"

class DetectorDlNode : public rclcpp::Node {
private:
  // 图像订阅者
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;

  // 识别器
  std::unique_ptr<Detector> detector_;

  // 单目解算（PNP）
  std::unique_ptr<Monocular> monocular_;

  // 相机信息订阅
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;

  // 源图像订阅
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  // Detected armor publisher
  rclcpp::Publisher<autoaim_interfaces::msg::Armors>::SharedPtr armors_pub_;

public:
  explicit DetectorDlNode(const rclcpp::NodeOptions &options)
      : Node("detector_dl_node", options) {
    RCLCPP_INFO(this->get_logger(), "Starting DetectorDlNode!");

    // 初始化Detector
    this->detector_ = initDetector();

    // Armor发布者
    this->armors_pub_ = this->create_publisher<autoaim_interfaces::msg::Armors>(
        "/detector/armors", rclcpp::SensorDataQoS());
    // 相机信息订阅
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
          cam_info_ =
              std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
          monocular_ =
              std::make_unique<Monocular>(camera_info->k, camera_info->d);
          cam_info_sub_.reset(); // 停止接收
        });
    // 相机图片订阅
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&DetectorDlNode::imageCallback, this, std::placeholders::_1));
  }
  ~DetectorDlNode() = default;

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg) {
    auto armors = detectArmors(img_msg);
    autoaim_interfaces::msg::Armor armor_msg;

    // 当无装甲板，则不进行下一步
    if (armors.empty()) {
      return;
    }
    // 过滤掉id == "negative"的装甲板
    armors.erase(std::remove_if(armors.begin(), armors.end(),
                                [](const Armor &armor) {
                                  return armor.number == "negative";
                                }),
                 armors.end());
    // 当单目PnP生成成功
    if (monocular_ != nullptr) {
      // 新增：对所有识别到的装甲板进行pnp解算，发送到robot_tracker
      autoaim_interfaces::msg::Armors armors_msg;
      armors_msg.header = img_msg->header;
      armors_msg.header.frame_id = "camera";
      armors_msg.armors.reserve(armors.size());

      // 对所有装甲板进行pnp解算
      for (const auto &armor : armors) {
        auto [rVec, tVec, distance_to_image_center] =
            monocular_->PnP_solver(armor);
        armor_msg.type = static_cast<int>(armor.type);
        // armor_msg.number = "0"; // 需要实装数字检测
        armor_msg.number = armor.number;

        // 对获得的rvec tvec对应的坐标系做旋转变换
        // 让x轴朝前
        cv::Mat R_x = (cv::Mat_<double>(3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
        cv::Mat R_temp, T_temp;
        cv::Rodrigues(rVec, R_temp);

        R_temp = R_x * R_temp;
        T_temp = R_x * tVec;

        armor_msg.pose.position.x = T_temp.at<double>(0);
        armor_msg.pose.position.y = T_temp.at<double>(1);
        armor_msg.pose.position.z = T_temp.at<double>(2);

        // 旋转矩阵转四元数
        tf2::Matrix3x3 tf2_rotation_matrix(
            R_temp.at<double>(0, 0), R_temp.at<double>(0, 1),
            R_temp.at<double>(0, 2), R_temp.at<double>(1, 0),
            R_temp.at<double>(1, 1), R_temp.at<double>(1, 2),
            R_temp.at<double>(2, 0), R_temp.at<double>(2, 1),
            R_temp.at<double>(2, 2));
        tf2::Quaternion tf2_q;
        tf2_rotation_matrix.getRotation(tf2_q);
        armor_msg.pose.orientation = tf2::toMsg(tf2_q);
        armors_msg.armors.push_back(armor_msg);
      }
      armors_pub_->publish(armors_msg);
    } else {
      RCLCPP_ERROR_STREAM(this->get_logger(), "PnP init failed!");
    }
  }

  std::unique_ptr<Detector> initDetector() {
    // 初始化参数
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    int NUM_CLASSES = declare_parameter("4P_NUM_CLASSES", 36, param_desc);
    std::string TARGET_COLOUR =
        declare_parameter("4P_TARGET_COLOUR", "RED", param_desc);
    float NMS_THRESH = declare_parameter("4P_NMS_THRESH", 0.45, param_desc);
    float BBOX_CONF_THRESH =
        declare_parameter("4P_BBOX_CONF_THRESH", 0.5, param_desc);
    int INPUT_W = declare_parameter("4P_INPUT_W", 448, param_desc);
    int INPUT_H = declare_parameter("4P_INPUT_H", 448, param_desc);
    std::string engine_file_path =
        declare_parameter("4P_engine_file_path",
                          "/home/nvidia/mosas_autoaim_dl/src/detector_dl/model/"
                          "RM4points_v3_5_sim_448x448.trt",
                          param_desc);

    // 装甲板限定阈值
    Detector::ArmorParams a_params;
    // a_params.min_light_ratio = declare_parameter("armor.min_light_ratio",
    // 0.7); a_params.min_small_center_distance =
    // declare_parameter("armor.min_small_center_distance", 0.8);
    // a_params.max_small_center_distance =
    // declare_parameter("armor.max_small_center_distance", 3.2);
    a_params.min_large_center_distance =
        declare_parameter("armor.min_large_center_distance", 3.2);
    // a_params.max_large_center_distance =
    // declare_parameter("armor.max_large_center_distance", 5.5);
    // a_params.max_angle = declare_parameter("armor.max_angle", 35.0);

    // 初始化识别器
    return std::make_unique<Detector>(NUM_CLASSES, TARGET_COLOUR, NMS_THRESH,
                                      BBOX_CONF_THRESH, INPUT_W, INPUT_H,
                                      engine_file_path, a_params);
  }

  std::vector<Armor>
  detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr &img_msg) {
    // 转换为Mat

    auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
    auto start_time = this->now();
    // 更新参数（如果改变）
    detector_->NUM_CLASSES = get_parameter("4P_NUM_CLASSES").as_int();
    detector_->TARGET_COLOUR = get_parameter("4P_TARGET_COLOUR").as_string();
    detector_->NMS_THRESH = get_parameter("4P_NMS_THRESH").as_double();
    detector_->BBOX_CONF_THRESH = get_parameter("4P_BBOX_CONF_THRESH").as_double();
    detector_->INPUT_W = get_parameter("4P_INPUT_W").as_int();
    detector_->INPUT_H = get_parameter("4P_INPUT_H").as_int();
    detector_->engine_file_path = get_parameter("4P_engine_file_path").as_string();
    // 开始识别
    bool show_img = true;
    auto armors = detector_->detect(img, show_img);
    // 计算每张延时与FPS
    auto final_time = this->now();
    auto latency = (final_time - start_time).seconds() * 1000;
    RCLCPP_INFO_STREAM(this->get_logger(), "Latency: " << latency << "ms");
    RCLCPP_INFO_STREAM(this->get_logger(), "FPS: " << 1000 / latency);

    // 绘出图像并显示
    // cv::resize(img, img, {640, 480});
    // cv::imshow("result", img);
    // cv::waitKey(1);

    return armors;
  }
};
