#include <cstdint>
#include <cuda_runtime.h>

__global__ void warp_affine(
    std::uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    std::uint8_t const_value, float *d2i)
{
    // 计算索引对应目标图像坐标
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_width || dst_y >= dst_height)
        return;

    // 应用仿射变换逆矩阵找到对应源图像亚像素坐标
    float src_x = d2i[0] * dst_x + d2i[1] * dst_y + d2i[2];
    float src_y = d2i[3] * dst_x + d2i[4] * dst_y + d2i[5];

    // 双线性插值，超界用const_value填充
    float b = const_value, g = const_value, r = const_value;
    if (src_x >= 0 && src_x < src_width && src_y >= 0 && src_y < src_height)
    {
        // 计算最近邻四像素
        int x_low = __float2int_rd(src_x);
        int y_low = __float2int_rd(src_y);
        int x_high = min(x_low + 1, src_width - 1);
        int y_high = min(y_low + 1, src_height - 1);

        // 计算插值权重
        float lx = src_x - x_low, ly = src_y - y_low;
        float hx = x_high - src_x, hy = y_high - src_y;
        float w1 = hx * hy, w2 = lx * hy, w3 = hx * ly, w4 = lx * ly;
        
        // 获取四近邻像素指针
        std::uint8_t *v1 = src + 3 * (y_low * src_width + x_low);
        std::uint8_t *v2 = src + 3 * (y_low * src_width + x_high);
        std::uint8_t *v3 = src + 3 * (y_high * src_width + x_low);
        std::uint8_t *v4 = src + 3 * (y_high * src_width + x_high);

        // 执行双线性插值
        b = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        g = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        r = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // 重新排列通道: bgrbgr...->rr...rgg...gbb...b
    int area = dst_width * dst_height;
    float *pdst_c0 = dst + dst_y * dst_width + dst_x;
    float *pdst_c1 = pdst_c0 + area;
    float *pdst_c2 = pdst_c1 + area;

    // 归一化
    *pdst_c0 = r / 255.0f;
    *pdst_c1 = g / 255.0f;
    *pdst_c2 = b / 255.0f;
}

void preprocess(
    std::uint8_t *src, int src_width, int src_height,
    float *dst, int dst_width, int dst_height,
    float *d2i, cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((dst_width + 15) / 16, (dst_height + 15) / 16);
    warp_affine<<<grid_size, block_size, 0, stream>>>(
        src, src_width, src_height,
        dst, dst_width, dst_height,
        128, d2i);
}