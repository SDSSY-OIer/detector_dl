#pragma once

#include <algorithm>
#include <opencv2/opencv.hpp>

class AffineMatrix
{
public:
    float i2d[6]; // 仿射变换正变换
    float d2i[6]; // 仿射变换逆变换
    AffineMatrix(cv::Size to, cv::Size from)
    {
        // 设置仿射变换矩阵，按原比例缩小，使to恰好框住缩小后图片，且平移至to的中心
        float scale = std::min(1.0f * to.width / from.width, 1.0f * to.height / from.height);
        i2d[0] = scale;
        i2d[1] = 0;
        i2d[2] = 0.5f * (to.width - scale * from.width);
        i2d[3] = 0;
        i2d[4] = scale;
        i2d[5] = 0.5f * (to.height - scale * from.height);

        // 设置仿射变换逆矩阵
        scale = std::max(1.0f * from.width / to.width, 1.0f * from.height / to.height);
        d2i[0] = scale;
        d2i[1] = 0;
        d2i[2] = 0.5f * (from.width - scale * to.width);
        d2i[3] = 0;
        d2i[4] = scale;
        d2i[5] = 0.5f * (from.height - scale * to.height);
    }
};