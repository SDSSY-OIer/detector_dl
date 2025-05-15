#pragma once

#include <cmath>
#include <vector>
#include <cstring>
#include <fstream>

#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <opencv2/opencv.hpp>

#include "detector_dl/Armor.hpp"
#include "detector_dl/Affine.hpp"
#include "detector_dl/CudaUtils.cuh"

using namespace nvinfer1;

// 用于画图
cv::Scalar color_list[]{
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(255, 255, 255)};

// 模型参数
constexpr static int DEVICE = 0;
constexpr static int NUM_CLASSES = 36; // 类别数量
constexpr static int CKPT_NUM = 4;     // 关键点数量
constexpr static int NUM_BOX_ELEMENT = 7 + CKPT_NUM * 2;
constexpr static char *INPUT_BLOB_NAME = "input";               // 模型导出ONNX文件时设置的输入名字
constexpr static char *OUTPUT_BLOB_NAME = "output";             // 模型导出ONNX文件时设置的输出名字
constexpr static int MAX_IMAGE_INPUT_SIZE_THRESH = 5000 * 5000; // 图像输入尺寸上限
constexpr static int MAX_OBJECTS = 32;

class Detector
{
public:
    int NUM_CLASSES;
    std::string TARGET_COLOUR;
    float NMS_THRESH;
    float BBOX_CONF_THRESH;
    // 目标尺寸
    int INPUT_W;
    int INPUT_H;
    std::string engine_file_path;
    // 装甲板限定属性
    float min_large_center_distance;

private:
    // 创建引擎
    IRuntime *runtime_det;
    ICudaEngine *engine_det;
    IExecutionContext *context_det;
    // CUDA与TRT相关
    Logger gLogger;
    cudaStream_t stream;
    float *buffers[2];
    int inputIndex;
    int outputIndex;
    uint8_t *img_host = nullptr;
    uint8_t *img_device = nullptr;
    float *affine_matrix_d2i_host = nullptr;
    float *affine_matrix_d2i_device = nullptr;
    float *decode_ptr_device = nullptr;
    float *decode_ptr_host = new float[1 + MAX_OBJECTS * NUM_BOX_ELEMENT];
    int OUTPUT_CANDIDATES;

public:
    Detector(int NUM_CLASSES, const std::string &TARGET_COLOUR, float NMS_THRESH, float BBOX_CONF_THRESH, int INPUT_W, int INPUT_H, const std::string &engine_file_path, float min_large_center_distance);
    void InitModelEngine();
    void AllocMem();
    // show image在detect实现里
    std::vector<Armor> detect(cv::Mat &frame);
    void Release();
    ~Detector();
};