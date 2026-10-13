// add.hip - Element-wise addition kernels for DotLLM.Hip (H2.2).
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
add_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2)
    {
        const __half2* a2 = reinterpret_cast<const __half2*>(a);
        const __half2* b2 = reinterpret_cast<const __half2*>(b);
        __half2* out2 = reinterpret_cast<__half2*>(output);
        out2[idx] = __hadd2(a2[idx], b2[idx]);
    }

    if ((n & 1) && idx == 0)
    {
        int last = n - 1;
        output[last] = to_half(to_float(a[last]) + to_float(b[last]));
    }
}

extern "C" __global__ void __launch_bounds__(256)
add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = a[idx] + b[idx];
}

// output_f32 = a_f32 + b_f16 (adds FP16 layer output into FP32 residual stream)
extern "C" __global__ void __launch_bounds__(256)
add_f32_f16(
    const float* __restrict__ a,
    const __half* __restrict__ b,
    float* __restrict__ output,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        output[idx] = a[idx] + to_float(b[idx]);
}
