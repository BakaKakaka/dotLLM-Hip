// swiglu.hip - SwiGLU activation kernel for DotLLM.Hip (H2.2).
// out[i] = SiLU(gate[i]) * up[i] = (gate[i] / (1 + exp(-gate[i]))) * up[i]
// Vectorized: half2 operations, FP32 computation for sigmoid precision.
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
swiglu_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ output,
    const int n,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * seq_len;
    int total2 = total / 2;

    if (idx < total2)
    {
        __half2 g2 = reinterpret_cast<const __half2*>(gate)[idx];
        __half2 u2 = reinterpret_cast<const __half2*>(up)[idx];

        float g0 = to_float(__low2half(g2));
        float g1 = to_float(__high2half(g2));
        float u0 = to_float(__low2half(u2));
        float u1 = to_float(__high2half(u2));

        float s0 = (g0 / (1.0f + expf(-g0))) * u0;
        float s1 = (g1 / (1.0f + expf(-g1))) * u1;

        __half2 res;
        res.x = to_half(s0);
        res.y = to_half(s1);
        reinterpret_cast<__half2*>(output)[idx] = res;
    }

    if ((total & 1) && idx == 0)
    {
        int last = total - 1;
        float g = to_float(gate[last]);
        float u = to_float(up[last]);
        output[last] = to_half((g / (1.0f + expf(-g))) * u);
    }
}

extern "C" __global__ void __launch_bounds__(256)
swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ output,
    const int n,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * seq_len;
    if (idx < total)
    {
        float g = gate[idx];
        float u = up[idx];
        output[idx] = (g / (1.0f + expf(-g))) * u;
    }
}
