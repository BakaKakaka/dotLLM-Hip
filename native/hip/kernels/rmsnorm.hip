// rmsnorm.hip - RMS normalization kernels for DotLLM.Hip (H2.2).
// Supports FP32 input for residual stream with FP16 output for downstream GEMM,
// as well as pure FP16 and FP32 variants.
#include "rdna_common.h"

// RMSNorm: FP32 input, FP32 weight, FP16 output.
extern "C" __global__ void __launch_bounds__(256)
rmsnorm_f32in_f16out(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    __half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    __half* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    __shared__ float s_mem[32];
    __shared__ float s_rms_inv;

    float b_sum = block_reduce_sum(sum_sq, s_mem);
    if (threadIdx.x == 0)
    {
        s_rms_inv = rsqrtf(b_sum / (float)n + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        y[i] = to_half(x[i] * rms_inv * weight[i]);
    }
}

// RMSNorm: FP32 input, FP16 weight, FP16 output.
extern "C" __global__ void __launch_bounds__(256)
rmsnorm_f32in_f16out_w16(
    const float* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    __half* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    __shared__ float s_mem[32];
    __shared__ float s_rms_inv;

    float b_sum = block_reduce_sum(sum_sq, s_mem);
    if (threadIdx.x == 0)
    {
        s_rms_inv = rsqrtf(b_sum / (float)n + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        y[i] = to_half(x[i] * rms_inv * to_float(weight[i]));
    }
}

// RMSNorm: pure FP16 in/out with FP32 accumulator.
extern "C" __global__ void __launch_bounds__(256)
rmsnorm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const __half* x = input + (size_t)row * n;
    __half* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = to_float(x[i]);
        sum_sq += v * v;
    }

    __shared__ float s_mem[32];
    __shared__ float s_rms_inv;

    float b_sum = block_reduce_sum(sum_sq, s_mem);
    if (threadIdx.x == 0)
    {
        s_rms_inv = rsqrtf(b_sum / (float)n + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        y[i] = to_half(to_float(x[i]) * rms_inv * to_float(weight[i]));
    }
}

// RMSNorm: pure FP32 in/out.
extern "C" __global__ void __launch_bounds__(256)
rmsnorm_f32(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    const float* x = input + (size_t)row * n;
    float* y = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    __shared__ float s_mem[32];
    __shared__ float s_rms_inv;

    float b_sum = block_reduce_sum(sum_sq, s_mem);
    if (threadIdx.x == 0)
    {
        s_rms_inv = rsqrtf(b_sum / (float)n + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        y[i] = x[i] * rms_inv * weight[i];
    }
}
