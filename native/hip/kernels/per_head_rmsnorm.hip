// per_head_rmsnorm.hip - Per-head RMS normalization kernel (QK-norm) for DotLLM.Hip (H2.2).
// Normalizes each head vector independently: qk[t, h, :headDim].
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
per_head_rmsnorm_f16(
    __half* __restrict__ qk,
    const __half* __restrict__ weight,
    const float eps,
    const int num_heads,
    const int head_dim,
    const int seq_len)
{
    int block_id = blockIdx.x;
    int t = block_id / num_heads;
    int h = block_id % num_heads;

    if (t >= seq_len) return;

    int stride = num_heads * head_dim;
    __half* vec = qk + (size_t)t * stride + h * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        float v = to_float(vec[i]);
        sum_sq += v * v;
    }

    __shared__ float s_mem[32];
    __shared__ float s_rms_inv;

    float b_sum = block_reduce_sum(sum_sq, s_mem);
    if (threadIdx.x == 0)
    {
        s_rms_inv = rsqrtf(b_sum / (float)head_dim + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        float v = to_float(vec[i]);
        float w = to_float(weight[i]);
        vec[i] = to_half(v * rms_inv * w);
    }
}

extern "C" __global__ void __launch_bounds__(256)
per_head_rmsnorm_f32(
    float* __restrict__ qk,
    const float* __restrict__ weight,
    const float eps,
    const int num_heads,
    const int head_dim,
    const int seq_len)
{
    int block_id = blockIdx.x;
    int t = block_id / num_heads;
    int h = block_id % num_heads;

    if (t >= seq_len) return;

    int stride = num_heads * head_dim;
    float* vec = qk + (size_t)t * stride + h * head_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        float v = vec[i];
        sum_sq += v * v;
    }

    __shared__ float s_mem[32];
    __shared__ float s_rms_inv;

    float b_sum = block_reduce_sum(sum_sq, s_mem);
    if (threadIdx.x == 0)
    {
        s_rms_inv = rsqrtf(b_sum / (float)head_dim + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
    {
        vec[i] = vec[i] * rms_inv * weight[i];
    }
}
