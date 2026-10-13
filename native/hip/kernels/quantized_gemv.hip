// quantized_gemv.hip - Quantized GEMV kernels for DotLLM.Hip decode path (H2.3).
// y[n] = W[n, k] @ x[k]
// Mandates from docs/HIP_PLAN.md:
//  - Exactly ONE wave (32 lanes) per output row.
//  - FP16 activations (x), FP32 accumulation, FP16 output (y).
//  - All 32 lanes active throughout, single wave_sum reduction per row.
#include "rdna_common.h"
#include <stdint.h>

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
quantized_gemv_q8_0(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int blocks_per_row = k / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 34;

    float lane_acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++)
    {
        const uint8_t* block = w_row + (size_t)b * 34;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
        lane_acc += (d * (float)q) * to_float(x[b * 32 + lane]);
    }

    float row_sum = wave_sum(lane_acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q4_0: 18 bytes per 32 values ────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
quantized_gemv_q4_0(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int blocks_per_row = k / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 18;

    int byte_idx = lane / 2;
    int shift = (lane & 1) ? 4 : 0;

    float lane_acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++)
    {
        const uint8_t* block = w_row + (size_t)b * 18;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        uint8_t packed = block[2 + byte_idx];
        int val = (int)((packed >> shift) & 0x0F) - 8;
        lane_acc += (d * (float)val) * to_float(x[b * 32 + lane]);
    }

    float row_sum = wave_sum(lane_acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q5_0: 22 bytes per 32 values ────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
quantized_gemv_q5_0(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int blocks_per_row = k / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 22;

    int j = lane < 16 ? lane : lane - 16;
    int shift = (lane < 16) ? 0 : 4;

    float lane_acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++)
    {
        const uint8_t* block = w_row + (size_t)b * 22;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) |
                          ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
        uint8_t packed = block[6 + j];
        int nibble = (packed >> shift) & 0x0F;
        int high_bit = (qh >> lane) & 1;
        int val = (nibble | (high_bit << 4)) - 16;
        lane_acc += (d * (float)val) * to_float(x[b * 32 + lane]);
    }

    float row_sum = wave_sum(lane_acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q4_K: 144 bytes per 256 values ──────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
quantized_gemv_q4_k(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;

    float lane_acc = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; sb++)
    {
        const uint8_t* block = w_row + (size_t)sb * 144;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        float dmin = to_float(*reinterpret_cast<const __half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd = pair * 2 + 1;

            int sc0, m0, sc1, m1;
            if (sb_even < 4)
            {
                sc0 = scales_raw[sb_even] & 0x3F;
                m0  = scales_raw[sb_even + 4] & 0x3F;
                sc1 = scales_raw[sb_odd] & 0x3F;
                m1  = scales_raw[sb_odd + 4] & 0x3F;
            }
            else
            {
                sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
                m0  = (scales_raw[sb_even + 4] >> 4)   | ((scales_raw[sb_even] >> 6) << 4);
                sc1 = (scales_raw[sb_odd + 4] & 0x0F)  | ((scales_raw[sb_odd - 4] >> 6) << 4);
                m1  = (scales_raw[sb_odd + 4] >> 4)    | ((scales_raw[sb_odd] >> 6) << 4);
            }

            uint8_t byte_val = qs[pair * 32 + lane];
            int nibble0 = byte_val & 0x0F;
            int nibble1 = byte_val >> 4;

            float w0 = d * (float)sc0 * (float)nibble0 - dmin * (float)m0;
            float w1 = d * (float)sc1 * (float)nibble1 - dmin * (float)m1;

            lane_acc += w0 * to_float(x[sb * 256 + sb_even * 32 + lane]);
            lane_acc += w1 * to_float(x[sb * 256 + sb_odd * 32 + lane]);
        }
    }

    float row_sum = wave_sum(lane_acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q5_K: 176 bytes per 256 values ──────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
quantized_gemv_q5_k(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 176;

    float lane_acc = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; sb++)
    {
        const uint8_t* block = w_row + (size_t)sb * 176;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        float dmin = to_float(*reinterpret_cast<const __half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        for (int sub = 0; sub < 8; sub++)
        {
            int sc, m;
            if (sub < 4)
            {
                sc = scales_raw[sub] & 0x3F;
                m  = scales_raw[sub + 4] & 0x3F;
            }
            else
            {
                sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
                m  = (scales_raw[sub + 4] >> 4)   | ((scales_raw[sub] >> 6) << 4);
            }

            float scale = d * (float)sc;
            float min_val = dmin * (float)m;

            int pair = sub / 2;
            int is_odd = sub % 2;
            uint8_t byte_val = qs[pair * 32 + lane];
            int nibble = is_odd ? (byte_val >> 4) : (byte_val & 0x0F);
            int bit5 = (qh[lane] >> sub) & 1;
            int val = nibble | (bit5 << 4);

            float w = scale * (float)val - min_val;
            lane_acc += w * to_float(x[sb * 256 + sub * 32 + lane]);
        }
    }

    float row_sum = wave_sum(lane_acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q6_K: 210 bytes per 256 values ──────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
quantized_gemv_q6_k(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 210;

    int l = lane;
    int isc = l / 16;

    float lane_acc = 0.0f;
    for (int sb = 0; sb < superblocks_per_row; sb++)
    {
        const uint8_t* block = w_row + (size_t)sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh_base = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = to_float(*reinterpret_cast<const __half*>(block + 208));

        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            const uint8_t* ql_half = ql + half_idx * 64;
            const uint8_t* qh_half = qh_base + half_idx * 32;
            const int8_t* sc_half = scales + half_idx * 8;

            int q0 = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
            int q1 = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
            int q2 = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32;
            int q3 = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32;

            float sc0 = d * (float)sc_half[isc + 0 * 2];
            float sc1 = d * (float)sc_half[isc + 1 * 2];
            float sc2 = d * (float)sc_half[isc + 2 * 2];
            float sc3 = d * (float)sc_half[isc + 3 * 2];

            int base_k = sb * 256 + half_idx * 128 + l;
            lane_acc += (sc0 * (float)q0) * to_float(x[base_k + 0 * 32]);
            lane_acc += (sc1 * (float)q1) * to_float(x[base_k + 1 * 32]);
            lane_acc += (sc2 * (float)q2) * to_float(x[base_k + 2 * 32]);
            lane_acc += (sc3 * (float)q3) * to_float(x[base_k + 3 * 32]);
        }
    }

    float row_sum = wave_sum(lane_acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}
