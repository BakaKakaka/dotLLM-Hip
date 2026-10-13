// spike_kernels.hip - kernels for the H0.2 (launch overhead) and H0.3 (bandwidth / MALL) spikes.
// Benchmark code only; not embedded into DotLLM.Hip. See docs/HIP_PLAN.md H0.2/H0.3, docs/HIP.md.
#include <hip/hip_runtime.h>

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// H0.3: fills n 16-byte elements with a per-element hash so that bandwidth numbers cannot be
// inflated by constant data (compression / zero detection anywhere in the memory hierarchy).
extern "C" __global__ void __launch_bounds__(256)
spike_fill(u32x4* __restrict__ dst, unsigned long long n)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
        unsigned int h = (unsigned int)i * 0x9E3779B1u ^ (unsigned int)(i >> 32) * 0x85EBCA77u;
        h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
        dst[i] = u32x4{h, h * 0x01000193u, h ^ 0xA5A5A5A5u, h + 0x7F4A7C15u};
    }
}

// H0.2: measures pure dispatch cost (CPU enqueue + GPU front-end), no work.
extern "C" __global__ void spike_empty()
{
}

// H0.2: keeps the GPU busy while the CPU pre-enqueues work, so that event timings of the work
// queued behind it measure back-to-back GPU execution instead of the CPU enqueue rate.
// Each iteration sleeps ~127 * 64 clocks; the host calibrates iterations -> ms. Keep < 100 ms (W1).
extern "C" __global__ void __launch_bounds__(32) spike_spin(unsigned int iterations, unsigned int* __restrict__ sink)
{
    unsigned int acc = 0;
    for (unsigned int i = 0; i < iterations; ++i)
    {
        __builtin_amdgcn_s_sleep(127);
        acc += i;
    }
    if (acc == 0x9E3779B9u)
        sink[0] = acc;
}

template <bool NonTemporal>
__device__ __forceinline__ u32x4 load16(const u32x4* p)
{
    if constexpr (NonTemporal)
        return __builtin_nontemporal_load(p);
    else
        return *p;
}

// H0.3: grid-stride read of n 16-byte elements, repeated `passes` times (so that small working
// sets that fit in L2 / Infinity Cache are not dominated by launch overhead). Four independent
// 16 B loads in flight per lane per iteration. The result is consumed by an (in practice never
// taken) store so the loads cannot be eliminated.
template <bool NonTemporal>
__device__ __forceinline__ void stream_read(const u32x4* __restrict__ src, unsigned long long n,
                                            unsigned int passes, unsigned int* __restrict__ sink)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const unsigned long long start = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    u32x4 acc = {0u, 0u, 0u, 0u};
    for (unsigned int p = 0; p < passes; ++p)
    {
        unsigned long long i = start;
        for (; i + 3 * stride < n; i += 4 * stride)
        {
            const u32x4 a = load16<NonTemporal>(src + i);
            const u32x4 b = load16<NonTemporal>(src + i + stride);
            const u32x4 c = load16<NonTemporal>(src + i + 2 * stride);
            const u32x4 d = load16<NonTemporal>(src + i + 3 * stride);
            acc += (a ^ b) + (c ^ d);
        }
        for (; i < n; i += stride)
            acc += load16<NonTemporal>(src + i);
        acc += p;
    }
    const unsigned int r = acc.x ^ acc.y ^ acc.z ^ acc.w;
    if (r == 0x9E3779B9u)
        sink[0] = r;
}

extern "C" __global__ void __launch_bounds__(256)
spike_stream_read(const u32x4* __restrict__ src, unsigned long long n, unsigned int passes, unsigned int* __restrict__ sink)
{
    stream_read<false>(src, n, passes, sink);
}

extern "C" __global__ void __launch_bounds__(256)
spike_stream_read_nt(const u32x4* __restrict__ src, unsigned long long n, unsigned int passes, unsigned int* __restrict__ sink)
{
    stream_read<true>(src, n, passes, sink);
}
