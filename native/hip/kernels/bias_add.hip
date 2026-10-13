// bias_add.hip - Bias addition kernel for DotLLM.Hip (H2.2).
// output[t, i] += bias[i]
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
bias_add_f16(
    __half* __restrict__ output,
    const __half* __restrict__ bias,
    const int dim,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * seq_len;
    int dim2 = dim / 2;
    int total2 = total / 2;

    if (idx < total2)
    {
        __half2* out2 = reinterpret_cast<__half2*>(output);
        const __half2* bias2 = reinterpret_cast<const __half2*>(bias);
        int col2 = idx % dim2;
        out2[idx] = __hadd2(out2[idx], bias2[col2]);
    }

    if ((total & 1) && idx == 0)
    {
        int last = total - 1;
        int col = last % dim;
        output[last] = to_half(to_float(output[last]) + to_float(bias[col]));
    }
}

extern "C" __global__ void __launch_bounds__(256)
bias_add_f32(
    float* __restrict__ output,
    const __half* __restrict__ bias,
    const int dim,
    const int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * seq_len;
    if (idx < total)
    {
        output[idx] += to_float(bias[idx % dim]);
    }
}
