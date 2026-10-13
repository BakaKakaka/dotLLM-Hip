// rdna_common.h - Shared RDNA (gfx1100 / gfx1201) primitives for DotLLM.Hip (H2.1).
//
// Key architecture mandates:
//  1. RDNA uses 32-thread wavefronts (wave32). warpSize is not constexpr in ROCm 7,
//     so WAVE = 32 is defined as constexpr for compile-time sizing.
//  2. __shfl_*_sync intrinsics require 64-bit masks (FULL_MASK = 0xFFFFFFFFFFFFFFFFull).
//     Never pass a 32-bit mask.
//  3. Avoid std::min / std::max due to MSVC STL constexpr conflicts under hipcc (W15).
//     Use device_min, device_max, device_clamp or HIP intrinsics.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// RDNA hardware wave size is 32 (docs/HIP_PLAN.md H2.1).
constexpr int WAVE = 32;

// Full 64-bit lane mask for ROCm 7 shuffle intrinsics (docs/HIP_PITFALLS.md W15 / §4).
constexpr unsigned long long FULL_MASK = 0xFFFFFFFFFFFFFFFFull;

// ---------------------------------------------------------------------------
// Min / Max / Clamp helpers (avoid std::min / std::max due to W15 constexpr issue)
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T device_min(T a, T b)
{
    return a < b ? a : b;
}

template <typename T>
__device__ __forceinline__ T device_max(T a, T b)
{
    return a > b ? a : b;
}

template <typename T>
__device__ __forceinline__ T device_clamp(T val, T low, T high)
{
    return val < low ? low : (val > high ? high : val);
}

// ---------------------------------------------------------------------------
// FP16 helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float to_float(__half h)
{
    return __half2float(h);
}

__device__ __forceinline__ __half to_half(float f)
{
    return __float2half(f);
}

// ---------------------------------------------------------------------------
// Wave reductions (32 lanes per wave)
// ---------------------------------------------------------------------------

// Sum reduction within a wave (32 lanes). Result in lane 0.
__device__ __forceinline__ float wave_sum(float val)
{
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ int wave_sum(int val)
{
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Max reduction within a wave (32 lanes). Result in lane 0.
__device__ __forceinline__ float wave_max(float val)
{
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(FULL_MASK, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

// All-reduce sum within a wave using xor butterfly. All 32 lanes get the total sum.
__device__ __forceinline__ float wave_all_sum(float val)
{
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;
}

// All-reduce max within a wave using xor butterfly. All 32 lanes get the max.
__device__ __forceinline__ float wave_all_max(float val)
{
#pragma unroll
    for (int offset = WAVE / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(FULL_MASK, val, offset);
        val = val > other ? val : other;
    }
    return val;
}

// ---------------------------------------------------------------------------
// Block reductions (up to 1024 threads / 32 waves)
// ---------------------------------------------------------------------------

// Sum reduction across a block.
// s_mem must have at least (blockDim.x / WAVE) elements (max 32 floats).
// Returns the block sum on threadIdx.x == 0.
__device__ __forceinline__ float block_reduce_sum(float val, float* s_mem)
{
    val = wave_sum(val);
    int lane = threadIdx.x % WAVE;
    int wid = threadIdx.x / WAVE;

    if (lane == 0)
    {
        s_mem[wid] = val;
    }
    __syncthreads();

    int num_waves = (blockDim.x + WAVE - 1) / WAVE;
    float sum = (lane < num_waves) ? s_mem[lane] : 0.0f;
    if (wid == 0)
    {
        sum = wave_sum(sum);
    }
    return sum; // Valid on threadIdx.x == 0
}

// Broadcast block sum to all threads in the block.
__device__ __forceinline__ float block_all_sum(float val, float* s_mem)
{
    float sum = block_reduce_sum(val, s_mem);
    if (threadIdx.x == 0)
    {
        s_mem[0] = sum;
    }
    __syncthreads();
    return s_mem[0];
}

// Max reduction across a block.
// s_mem must have at least (blockDim.x / WAVE) elements (max 32 floats).
// Returns the block max on threadIdx.x == 0.
__device__ __forceinline__ float block_reduce_max(float val, float* s_mem)
{
    val = wave_max(val);
    int lane = threadIdx.x % WAVE;
    int wid = threadIdx.x / WAVE;

    if (lane == 0)
    {
        s_mem[wid] = val;
    }
    __syncthreads();

    int num_waves = (blockDim.x + WAVE - 1) / WAVE;
    float max_val = (lane < num_waves) ? s_mem[lane] : -1e38f;
    if (wid == 0)
    {
        max_val = wave_max(max_val);
    }
    return max_val; // Valid on threadIdx.x == 0
}

// Broadcast block max to all threads in the block.
__device__ __forceinline__ float block_all_max(float val, float* s_mem)
{
    float max_val = block_reduce_max(val, s_mem);
    if (threadIdx.x == 0)
    {
        s_mem[0] = max_val;
    }
    __syncthreads();
    return s_mem[0];
}
