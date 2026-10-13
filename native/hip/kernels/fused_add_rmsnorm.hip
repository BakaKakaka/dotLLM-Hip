// fused_add_rmsnorm.hip - Fused residual-add + RMS normalization for DotLLM.Hip (H2.2).
// Implements FP32 residual stream (D5):
//   1. sum = residual[i] (FP32) + x[i] (FP16 promoted to FP32)
//   2. residual[i] = sum  (FP32 retained in residual buffer)
//   3. output[i] = FP16(sum * rsqrt(rms) * weight[i])
#include "rdna_common.h"

// Fused residual add + RMSNorm: FP32 residual, FP16 layer output x, FP32 weight, FP16 normalized output.
extern "C" __global__ void __launch_bounds__(256)
fused_add_rmsnorm_f32res(
    float* __restrict__ residual,       // [rows, n] in/out: updated in FP32
    const __half* __restrict__ x,       // [rows, n] layer output to add
    const float* __restrict__ weight,   // [n] norm weights
    __half* __restrict__ output,        // [rows, n] normalized output
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    float* res_row = residual + (size_t)row * n;
    const __half* x_row = x + (size_t)row * n;
    __half* out_row = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float sum = res_row[i] + to_float(x_row[i]);
        res_row[i] = sum;
        sum_sq += sum * sum;
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
        out_row[i] = to_half(res_row[i] * rms_inv * weight[i]);
    }
}

// Fused residual add + RMSNorm: FP32 residual, FP16 layer output x, FP16 weight, FP16 normalized output.
extern "C" __global__ void __launch_bounds__(256)
fused_add_rmsnorm_f32res_w16(
    float* __restrict__ residual,       // [rows, n] in/out: updated in FP32
    const __half* __restrict__ x,       // [rows, n] layer output to add
    const __half* __restrict__ weight,  // [n] norm weights
    __half* __restrict__ output,        // [rows, n] normalized output
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    float* res_row = residual + (size_t)row * n;
    const __half* x_row = x + (size_t)row * n;
    __half* out_row = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float sum = res_row[i] + to_float(x_row[i]);
        res_row[i] = sum;
        sum_sq += sum * sum;
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
        out_row[i] = to_half(res_row[i] * rms_inv * to_float(weight[i]));
    }
}

// Legacy/FP16 residual variant for comparison.
extern "C" __global__ void __launch_bounds__(256)
fused_add_rmsnorm_f16(
    __half* __restrict__ residual,
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int n,
    const float eps)
{
    const int row = blockIdx.x;
    __half* res_row = residual + (size_t)row * n;
    const __half* x_row = x + (size_t)row * n;
    __half* out_row = output + (size_t)row * n;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        float sum = to_float(res_row[i]) + to_float(x_row[i]);
        res_row[i] = to_half(sum);
        sum_sq += sum * sum;
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
        out_row[i] = to_half(to_float(res_row[i]) * rms_inv * to_float(weight[i]));
    }
}
