#pragma once

#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#include "detector_dl/Armor.hpp"
#include "detector_dl/Affine.hpp"
#include "detector_dl/CudaUtils.cuh"

using namespace nvinfer1;

// 模型参数
constexpr static int DEVICE = 0;
constexpr static int NUM_CLASSES = 36;                              // 类别数量
constexpr static int CKPT_NUM = 4;                                  // 关键点数量
constexpr static int NUM_BOX_ELEMENT = 7 + CKPT_NUM * 2;
constexpr static char* INPUT_BLOB_NAME = "input";                  // 模型导出ONNX文件时设置的输入名字
constexpr static char* OUTPUT_BLOB_NAME = "output";                 // 模型导出ONNX文件时设置的输出名字
constexpr static int MAX_IMAGE_INPUT_SIZE_THRESH = 5000 * 5000;     // 图像输入尺寸上限
constexpr static int MAX_OBJECTS = 32;

class Detector
{
public:
    /**
     * @brief 关于装甲板的限定属性
     *
     */
    struct ArmorParams
    {
        // 两个灯条的最小长度比
        // double min_light_ratio;
        // light pairs distance
        // double min_small_center_distance;
        // double max_small_center_distance;
        double min_large_center_distance;
        // double max_large_center_distance;
        // horizontal angle
        // double max_angle;
    };

public:
    int NUM_CLASSES;
    std::string TARGET_COLOUR;
    float NMS_THRESH;
    float BBOX_CONF_THRESH;
    int INPUT_W;                                 // 目标尺寸
    int INPUT_H;
    std::string engine_file_path;
    ArmorParams a;

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
    bool is_detected = false;

public:
    Detector() = delete;
    Detector(const int &NUM_CLASSES, const std::string TARGET_COLOUR, const float &NMS_THRESH, const float &BBOX_CONF_THRESH, const int &INPUT_W, const int &INPUT_H, const std::string engine_file_path, const ArmorParams &a);

    void InitModelEngine();
    void AllocMem();
    std::vector<Armor> detect(cv::Mat &frame, bool show_img);
    void Release();
    ~Detector();
};
