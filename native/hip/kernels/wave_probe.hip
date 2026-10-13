// wave_probe.hip - Probe kernel testing RDNA wave size and rdna_common.h helpers (H2.1).
#include "rdna_common.h"

// Verifies warpSize == 32 and WAVE == 32 on the device.
// out[0] = warpSize as reported by hardware/runtime.
// out[1] = WAVE constant (32).
extern "C" __global__ void __launch_bounds__(WAVE)
probe_warp_size(int* __restrict__ out)
{
    if (threadIdx.x == 0)
    {
        out[0] = warpSize;
        out[1] = WAVE;
    }
}

// Tests wave_sum, wave_max, wave_all_sum, wave_all_max.
// Launched with 1 wave (32 threads).
extern "C" __global__ void __launch_bounds__(WAVE)
probe_wave_reductions(
    const float* __restrict__ in,
    float* __restrict__ out_sum,
    float* __restrict__ out_max,
    float* __restrict__ out_all_sum,
    float* __restrict__ out_all_max)
{
    int lane = threadIdx.x;
    float val = in[lane];

    float s = wave_sum(val);
    float m = wave_max(val);
    float as = wave_all_sum(val);
    float am = wave_all_max(val);

    out_sum[lane] = s;
    out_max[lane] = m;
    out_all_sum[lane] = as;
    out_all_max[lane] = am;
}

// Tests block_reduce_sum and block_all_sum across multiple waves (e.g. 128 or 256 threads).
extern "C" __global__ void __launch_bounds__(256)
probe_block_reductions(
    const float* __restrict__ in,
    float* __restrict__ out_reduce_sum,
    float* __restrict__ out_reduce_max,
    float* __restrict__ out_all_sum,
    float* __restrict__ out_all_max,
    const int n)
{
    __shared__ float s_mem[32]; // up to 32 waves = 1024 threads
    int idx = threadIdx.x;
    float val = idx < n ? in[idx] : 0.0f;

    float r_sum = block_reduce_sum(val, s_mem);
    if (idx == 0) out_reduce_sum[0] = r_sum;
    __syncthreads();

    float r_max = block_reduce_max(val, s_mem);
    if (idx == 0) out_reduce_max[0] = r_max;
    __syncthreads();

    float a_sum = block_all_sum(val, s_mem);
    out_all_sum[idx] = a_sum;
    __syncthreads();

    float a_max = block_all_max(val, s_mem);
    out_all_max[idx] = a_max;
}

// Tests FP16 helpers and min/max/clamp.
extern "C" __global__ void __launch_bounds__(WAVE)
probe_helpers(
    const float* __restrict__ in,
    float* __restrict__ out,
    const int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        float x = in[i];
        __half h = to_half(x);
        float x2 = to_float(h);
        float c = device_clamp(x2, 1.0f, 10.0f);
        float m = device_min(c, 5.0f);
        float mx = device_max(m, 2.0f);
        out[i] = mx;
    }
}
