#pragma once
#include "detector_dl/Armor.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

class Monocular {
public:
  // 相机参数
  cv::Mat CAMERA_MATRIX;    // 相机内参 3*3
  cv::Mat DISTORTION_COEFF; // 畸变系数 1*5

  // 物体在三维坐标系下的坐标
  std::vector<cv::Point3f> SMALL_ARMOR_POINTS_3D; // 小装甲板
  std::vector<cv::Point3f> LARGE_ARMOR_POINTS_3D; // 大装甲板

  // WIDTH->灯条长度 HEIGHT->装甲板长度 单位:mm
  // PnP重要依据,非必要无需修改
  static constexpr float SMALL_ARMOR_WIDTH = 135;
  static constexpr float SMALL_ARMOR_HEIGHT = 55;
  static constexpr float LARGE_ARMOR_WIDTH = 225;
  static constexpr float LARGE_ARMOR_HEIGHT = 55;

  static constexpr float to_angle = 180.0 / CV_PI;

  // results
  // double x_pos, y_pos, z_pos; // 装甲板坐标
  // double distance;            // 相机距装甲板的距离

public:
  Monocular(const std::array<double, 9> &camera_matrix,
            const std::vector<double> &dist_coeffs)
      : CAMERA_MATRIX(
            cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data()))
                .clone()),
        DISTORTION_COEFF(
            cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data()))
                .clone()) {
    // 单位:米 注意注意注意是米不是毫米
    constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
    constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
    constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
    constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

    // 从左下开始,顺时针顺序
    // 坐标系: X:前 Y:左 Z:上 右手系
    SMALL_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, small_half_y, -small_half_z));
    SMALL_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, small_half_y, small_half_z));
    SMALL_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, -small_half_y, small_half_z));
    SMALL_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, -small_half_y, -small_half_z));

    LARGE_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, large_half_y, -large_half_z));
    LARGE_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, large_half_y, large_half_z));
    LARGE_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, -large_half_y, large_half_z));
    LARGE_ARMOR_POINTS_3D.emplace_back(
        cv::Point3f(0, -large_half_y, -large_half_z));
  }

  /**
   * @brief 单目PnP测距
   *
   * @param armor 目标装甲板
   * @return std::tuple<cv::Mat, cv::Mat, double> rVec,tVec,distance 旋转矩阵
   * 平移矩阵 距离
   */
  std::tuple<cv::Mat, cv::Mat, double> PnP_solver(const Armor &armor) {
    // 旋转矩阵 平移矩阵
    // s[R|t]=s'  s->world coordinate;s`->camera coordinate
    cv::Mat rVec;
    cv::Mat tVec; // 装甲板中心的世界坐标系坐标

    // 进行PnP解算
    auto object_points = armor.type == ArmorType::SMALL ? SMALL_ARMOR_POINTS_3D
                                                        : LARGE_ARMOR_POINTS_3D;

    
    cv::solvePnP(object_points, armor.armorVertices_vector, CAMERA_MATRIX,
                 DISTORTION_COEFF, rVec, tVec, false, cv::SOLVEPNP_IPPE);

    // 获取装甲板中心世界系坐标
    double x_pos = tVec.at<double>(0, 0);
    double y_pos = tVec.at<double>(1, 0);
    double z_pos = tVec.at<double>(2, 0);
    std::cout << "x_pos: " << x_pos << "y_pos: " << y_pos << "z_pos: " << z_pos
              << std::endl;

    // 光心距装甲板中心距离
    float cx = CAMERA_MATRIX.at<double>(0, 2);
    float cy = CAMERA_MATRIX.at<double>(1, 2);
    double distance = cv::norm(armor.center - cv::Point2f(cx, cy));

    return {rVec, tVec, distance};
  }
};
