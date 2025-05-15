#pragma once

#include "opencv2/opencv.hpp"
#include "bits/stdc++.h"

enum ArmorType // 装甲板类型（confirm in detector.hpp，pnp解算的时候用））
{
    SMALL,
    LARGE
};

class Armor
{
public:
    cv::Point2f center;
    ArmorType type;
    double area;
    std::vector<cv::Point2f> armorVertices_vector; // 装甲板的四个顶点 bl->tl->tr->br(向量形式) 左下 左上 右上 右下
    std::string number;

public:
    Armor() = default;

    Armor(std::vector<cv::Point2f> points) : Armor()
    {
        this->armorVertices_vector = points;
        CalCenter(this->armorVertices_vector[0], this->armorVertices_vector[1], 
                  this->armorVertices_vector[2], this->armorVertices_vector[3]);        
    }

    ~Armor() = default;

public:
    void CalCenter(const cv::Point2f &down_left, const cv::Point2f &upper_left, 
                   const cv::Point2f &upper_right, const cv::Point2f &down_right)
    {
        // 灯条中心点相加再平分即为装甲板中心点
        this->center.x = (upper_left.x + down_right.x) / 2;
        this->center.y = (upper_left.y + down_right.y) / 2;

        // 装甲板面积
        this->area = (down_right.x-upper_left.x) * (upper_left.y - down_right.y);//后面pnp解算要用（优先攻击面积大的）
    }
};