// convert.hip - Type conversion kernels for DotLLM.Hip (H2.2).
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
convert_f16_to_f32(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2)
    {
        __half2 in = reinterpret_cast<const __half2*>(src)[idx];
        float2 out;
        out.x = to_float(__low2half(in));
        out.y = to_float(__high2half(in));
        reinterpret_cast<float2*>(dst)[idx] = out;
    }

    if ((n & 1) && idx == 0)
        dst[n - 1] = to_float(src[n - 1]);
}

extern "C" __global__ void __launch_bounds__(256)
convert_f32_to_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    const int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n / 2;

    if (idx < n2)
    {
        float2 in = reinterpret_cast<const float2*>(src)[idx];
        __half2 out;
        out.x = to_half(in.x);
        out.y = to_half(in.y);
        reinterpret_cast<__half2*>(dst)[idx] = out;
    }

    if ((n & 1) && idx == 0)
        dst[n - 1] = to_half(src[n - 1]);
}
