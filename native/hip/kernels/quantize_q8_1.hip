// quantize_q8_1.hip - On-the-fly Q8_1 activation quantization for DotLLM.Hip (H3.2).
//
// Quantizes an input activation vector x (FP16 or FP32) into block_q8_1 format on device.
// block_q8_1 layout (36 bytes per 32 values):
//   bytes 0..1:  half d        (max_abs / 127)
//   bytes 2..3:  half s        (d * sum(qs))
//   bytes 4..35: int8_t qs[32] (round(x[i] / d) clamped to [-127, 127])
//
// Execution model:
//   - Exactly ONE wave (32 lanes) per block of 32 values.
//   - BlockSize = 256 threads (8 waves per workgroup).
//   - Wave-wide max reduction finds d, lanes quantize in parallel, wave-wide sum computes s.
#include "rdna_common.h"
#include <stdint.h>

extern "C" __global__ void __launch_bounds__(256)
quantize_q8_1_f16(
    const __half* __restrict__ x,
    uint8_t* __restrict__ y,
    const int k)
{
    const int num_blocks = k / 32;
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int b = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (b >= num_blocks) return;

    float val = to_float(x[b * 32 + lane]);
    float abs_val = fabsf(val);

    float max_abs = wave_max(abs_val);
    max_abs = __shfl_sync(FULL_MASK, max_abs, 0);

    float d = max_abs / 127.0f;
    float id = d > 0.0f ? 127.0f / max_abs : 0.0f;

    int q = (int)roundf(val * id);
    q = device_clamp(q, -127, 127);

    int sum_q = wave_sum(q);

    uint8_t* block_dst = y + (size_t)b * 36;
    if (lane == 0)
    {
        *reinterpret_cast<__half*>(block_dst) = to_half(d);
        *reinterpret_cast<__half*>(block_dst + 2) = to_half(d * (float)sum_q);
    }
    block_dst[4 + lane] = (uint8_t)(int8_t)q;
}

extern "C" __global__ void __launch_bounds__(256)
quantize_q8_1_f32(
    const float* __restrict__ x,
    uint8_t* __restrict__ y,
    const int k)
{
    const int num_blocks = k / 32;
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int b = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (b >= num_blocks) return;

    float val = x[b * 32 + lane];
    float abs_val = fabsf(val);

    float max_abs = wave_max(abs_val);
    max_abs = __shfl_sync(FULL_MASK, max_abs, 0);

    float d = max_abs / 127.0f;
    float id = d > 0.0f ? 127.0f / max_abs : 0.0f;

    int q = (int)roundf(val * id);
    q = device_clamp(q, -127, 127);

    int sum_q = wave_sum(q);

    uint8_t* block_dst = y + (size_t)b * 36;
    if (lane == 0)
    {
        *reinterpret_cast<__half*>(block_dst) = to_half(d);
        *reinterpret_cast<__half*>(block_dst + 2) = to_half(d * (float)sum_q);
    }
    block_dst[4 + lane] = (uint8_t)(int8_t)q;
}
