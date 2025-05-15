#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * @brief 使用宏定义不需要显式传入FILE与LINE参数，让代码简洁的同时能显示CUDA运行时可能出现的报错
 * 
 */
#define CHECK(call)                                                             \
do                                                                              \
{                                                                               \
    const cudaError_t error_code = call;                                        \
    if (error_code != cudaSuccess)                                              \
    {                                                                           \
        printf("File: %s\n", __FILE__);                                         \
        printf("Line: %d\n", __LINE__);                                         \
        printf("CUDA Runtime Error: %s\n",cudaGetErrorString(error_code));      \
        exit(1);                                                                \
    }                                                                           \
} while(0)

/**
 * @brief 如果不使用宏定义可以使用此函数，但因未显式传入参数，会导致无法定位BUG位置
 * 
 * 
 */
// inline cudaError_t CHECK(cudaError_t result)
// {
//   if (result != cudaSuccess) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//     assert(result == cudaSuccess);
//   }
//   exit(1);
// }

// 用于画图
const float color_list[4][3] = {
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {0, 255, 255},
};

/**
 * 模型推理的结果
*/
struct bbox {
    float x1, x2, y1, y2;
    float landmarks[8]; // 4个关键点
    float score;
    int class_id;
  };  

void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                           float* dst, int dst_width, int dst_height,
                           float*d2i,cudaStream_t stream);

void decode_kernel_invoker(
    float* predict,
    int NUM_BOX_ELEMENT, 
    int num_bboxes, 
    int num_classes, 
    int ckpt,
    float confidence_threshold, 
    float* invert_affine_matrix, 
    float* parray,
    int max_objects, 
    cudaStream_t stream
);

void nms_kernel_invoker(
    float* parray, 
    float nms_threshold, 
    int max_objects, 
    cudaStream_t stream,
    int NUM_BOX_ELEMENT
);
