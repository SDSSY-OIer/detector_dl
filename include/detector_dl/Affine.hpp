#pragma once

#include <opencv2/opencv.hpp>

/**
 * @brief letter_box 仿射变换矩阵
 * 
 */
struct AffineMatrix
{
    float i2d[6]; // 仿射变换正变换
    float d2i[6]; // 仿射变换逆变换
};

/**
 * @brief 计算仿射变换的矩阵和逆矩阵
 * 
 * @param afmt 
 * @param to 
 * @param from 
 * @todo 尝试迁移至GPU上计算
 */
void getd2i(AffineMatrix &afmt, cv::Size to,cv::Size from)
{
    float scale = std::min(1.0*to.width/from.width, 1.0*to.height/from.height);
    afmt.i2d[0]=scale;
    afmt.i2d[1]=0;
    afmt.i2d[2]=-scale*from.width*0.5+to.width*0.5;
    afmt.i2d[3]=0;
    afmt.i2d[4]=scale;
    afmt.i2d[5]=-scale*from.height*0.5+to.height*0.5;
    cv::Mat i2d_mat(2,3,CV_32F,afmt.i2d);
    cv::Mat d2i_mat(2,3,CV_32F,afmt.d2i);
    cv::invertAffineTransform(i2d_mat,d2i_mat);
    memcpy(afmt.d2i, d2i_mat.ptr<float>(0), sizeof(afmt.d2i));
}


// /**
//  * @brief 通过仿射变换逆矩阵，恢复成原图的坐标
//  * 
//  * @param d2i 
//  * @param x 
//  * @param y 
//  * @param ox 
//  * @param oy 
//  */
// void affine_project(float *d2i,float x,float y,float *ox,float *oy) 
// {
//     *ox = d2i[0]*x+d2i[1]*y+d2i[2];
//     *oy = d2i[3]*x+d2i[4]*y+d2i[5];
// }