#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

// 使用宏定义不需要显式传入FILE与LINE参数，让代码简洁的同时能显示CUDA运行时可能出现的报错
#define CHECK(call)                                                             \
    do                                                                          \
    {                                                                           \
        const cudaError_t error_code = call;                                    \
        if (error_code != cudaSuccess)                                          \
        {                                                                       \
            printf("File: %s\n", __FILE__);                                     \
            printf("Line: %d\n", __LINE__);                                     \
            printf("CUDA Runtime Error: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// 模型推理的结果
struct bbox
{
    // 装甲板四顶点，opencv亚像素坐标下的左上，左下，右下，右上
    float landmarks[8];
    int class_id;
};

void preprocess(
    uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    float *d2i, cudaStream_t stream);

void decode_kernel_invoker(
    float *predict,
    int NUM_BOX_ELEMENT,
    int num_bboxes,
    int num_classes,
    int ckpt,
    float confidence_threshold,
    float *invert_affine_matrix,
    float *parray,
    int max_objects,
    cudaStream_t stream);

void nms_kernel_invoker(
    float *parray,
    float nms_threshold,
    int max_objects,
    cudaStream_t stream,
    int NUM_BOX_ELEMENT);
