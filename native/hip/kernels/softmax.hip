// softmax.hip - Numerically stable softmax kernel for DotLLM.Hip (H2.2).
// FP16 in/out and FP32 in/out, FP32 accumulator. One block per row.
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
softmax_f16(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const int rows,
    const int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const __half* x = input + (size_t)row * cols;
    __half* y = output + (size_t)row * cols;

    __shared__ float s_mem[32];
    __shared__ float s_scalar;

    // Pass 1: find max across the row
    float max_val = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = to_float(x[i]);
        if (v > max_val) max_val = v;
    }

    float b_max = block_reduce_max(max_val, s_mem);
    if (threadIdx.x == 0)
    {
        s_scalar = b_max;
    }
    __syncthreads();
    max_val = s_scalar;

    // Pass 2: compute exp(x - max), store intermediate to output, accumulate sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float e = expf(to_float(x[i]) - max_val);
        sum_exp += e;
        y[i] = to_half(e);
    }

    float b_sum = block_reduce_sum(sum_exp, s_mem);
    if (threadIdx.x == 0)
    {
        s_scalar = 1.0f / b_sum;
    }
    __syncthreads();
    float inv_sum = s_scalar;

    // Pass 3: normalize from stored exp values
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        y[i] = to_half(to_float(y[i]) * inv_sum);
    }
}

extern "C" __global__ void __launch_bounds__(256)
softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int rows,
    const int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    __shared__ float s_mem[32];
    __shared__ float s_scalar;

    float max_val = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = x[i];
        if (v > max_val) max_val = v;
    }

    float b_max = block_reduce_max(max_val, s_mem);
    if (threadIdx.x == 0)
    {
        s_scalar = b_max;
    }
    __syncthreads();
    max_val = s_scalar;

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float e = expf(x[i] - max_val);
        sum_exp += e;
        y[i] = e;
    }

    float b_sum = block_reduce_sum(sum_exp, s_mem);
    if (threadIdx.x == 0)
    {
        s_scalar = 1.0f / b_sum;
    }
    __syncthreads();
    float inv_sum = s_scalar;

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        y[i] = y[i] * inv_sum;
    }
}
