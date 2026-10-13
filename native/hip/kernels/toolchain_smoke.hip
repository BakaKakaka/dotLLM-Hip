// toolchain_smoke.hip - HIP toolchain smoke kernels (H0.1 / H0.4).
//
// Each kernel is launched with exactly one wave (32 lanes) and exercises one feature
// that the real kernels depend on. scripts/check-hip-env.ps1 compiles this file for
// gfx1100 and gfx1201, verifies the ISA in the disassembly and runs it on both cards.
//
//   smoke_dot4  -> v_dot4_i32_iu8          (__builtin_amdgcn_sudot4, needed by mmvq, H3.2)
//   smoke_wave  -> 64-bit shuffle mask + warpSize == 32
//   smoke_wmma  -> v_wmma_f32_16x16x16_f16 (f16 in, f32 accumulator, D5/D8)
#include <hip/hip_runtime.h>
#include "toolchain_smoke.h"

// out[i] = dot(int8x4 a[i], int8x4 b[i]), both operands signed, no clamp.
extern "C" __global__ void __launch_bounds__(SMOKE_WAVE)
smoke_dot4(const int* __restrict__ a, const int* __restrict__ b, int* __restrict__ out)
{
    const int i = threadIdx.x;
    out[i] = __builtin_amdgcn_sudot4(true, a[i], true, b[i], 0, false);
}

// out[lane]          = sum of all lane ids (0 + 1 + ... + 31 = 496) via xor-butterfly.
// out[WAVE + lane]   = warpSize as seen by the device (must be 32).
extern "C" __global__ void __launch_bounds__(SMOKE_WAVE)
smoke_wave(int* __restrict__ out)
{
    const int lane = threadIdx.x;
    int v = lane;
#pragma unroll
    for (int offset = SMOKE_WAVE / 2; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(SMOKE_FULL_MASK, v, offset);
    out[lane] = v;
    out[SMOKE_WAVE + lane] = warpSize;
}

// One 16x16x16 WMMA: C = A * B with A, B row-major f16 [16x16], C f32 [32 lanes x 8].
// The host fills A and B with constants, so the result is layout-independent; the
// operand layout itself is validated by the H4.1 layout tests, not here.
extern "C" __global__ void __launch_bounds__(SMOKE_WAVE)
smoke_wmma(const _Float16* __restrict__ a, const _Float16* __restrict__ b, float* __restrict__ c)
{
    const int lane = threadIdx.x;
    const int row = lane % SMOKE_WMMA_DIM;
    smoke_float8 acc = {};

#if defined(__gfx1100__)
    // RDNA3: A/B = 16 halves per lane, replicated across the two half-waves.
    smoke_half16 va;
    smoke_half16 vb;
#pragma unroll
    for (int k = 0; k < 16; ++k)
    {
        va[k] = a[row * SMOKE_WMMA_DIM + k];
        vb[k] = b[k * SMOKE_WMMA_DIM + row];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(va, vb, acc);
#elif defined(__gfx1201__)
    // RDNA4: A/B = 8 halves per lane, the half-waves hold different K halves.
    const int kBase = (lane / SMOKE_WMMA_DIM) * 8;
    smoke_half8 va;
    smoke_half8 vb;
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        va[k] = a[row * SMOKE_WMMA_DIM + kBase + k];
        vb[k] = b[(kBase + k) * SMOKE_WMMA_DIM + row];
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, vb, acc);
#elif defined(__HIP_DEVICE_COMPILE__)
#error "toolchain_smoke.hip: unsupported offload arch (expected gfx1100 or gfx1201)"
#endif

#pragma unroll
    for (int i = 0; i < 8; ++i)
        c[lane * 8 + i] = acc[i];
}
