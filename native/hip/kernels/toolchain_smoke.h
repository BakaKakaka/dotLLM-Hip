// toolchain_smoke.h - shared definitions for the HIP toolchain smoke kernels.
//
// Used by scripts/check-hip-env.ps1 (H0.1) and embedded into DotLLM.Hip by the
// CompileHipKernels MSBuild target (H0.4). Not product code: it exists to prove that
// hipcc + MSVC STL + Windows SDK + the HIP runtime work end-to-end on gfx1100/gfx1201.
// See docs/HIP_PITFALLS.md (W15) and docs/HIP_PLAN.md (H0.1).
#pragma once

// warpSize is not constexpr in ROCm 7 - hard-code the wave size (RDNA = wave32).
#define SMOKE_WAVE 32

// ROCm 7 __shfl_*_sync requires a 64-bit lane mask. Never pass a raw 0xFFFFFFFF.
#define SMOKE_FULL_MASK 0xFFFFFFFFFFFFFFFFull

// WMMA tile is 16x16x16. A is filled with SMOKE_WMMA_A, B with SMOKE_WMMA_B, so every
// element of C must equal 16 * A * B regardless of the per-arch operand layout.
#define SMOKE_WMMA_DIM 16
#define SMOKE_WMMA_A 1.0f
#define SMOKE_WMMA_B 2.0f

typedef _Float16 smoke_half8  __attribute__((ext_vector_type(8)));
typedef _Float16 smoke_half16 __attribute__((ext_vector_type(16)));
typedef float    smoke_float8 __attribute__((ext_vector_type(8)));
