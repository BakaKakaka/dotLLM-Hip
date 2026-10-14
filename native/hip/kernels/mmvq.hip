// mmvq.hip - Matrix-Matrix/Vector Multiplication Quantized (H3.2) for DotLLM.Hip.
// Reference: llama.cpp mmvq.cu / vecdotq.cuh (MIT).
//
// Computes y[n] = W[n, k] @ x[k] during decode (seqLen == 1):
//   - Activations x are quantized on-the-fly to block_q8_1 once per vector.
//   - Exactly ONE wave (32 lanes) per output row.
//   - 16 bytes per lane loads for maximum memory bandwidth.
//   - Low-precision hardware dot products via __builtin_amdgcn_sudot4 (v_dot4_i32_iu8).
//   - Dual variant with __builtin_amdgcn_fdot2 for benchmarking comparison.
#include "rdna_common.h"
#include <stdint.h>
#include <string.h>

// ── Q8_0 mmvq with sudot4 (H3.2 target) ──────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
mmvq_gemv_q8_0(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int blocks_per_row = k / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 34;

    const int block_half = lane & 1;       // 0: first 16 bytes, 1: second 16 bytes
    const int block_offset = lane >> 1;     // block within tile of 16 blocks (0..15)

    float acc = 0.0f;

    // Each iteration processes 16 blocks = 512 elements across all 32 lanes.
    for (int base_b = 0; base_b < blocks_per_row; base_b += 16)
    {
        int b = base_b + block_offset;
        if (b < blocks_per_row)
        {
            const uint8_t* w_blk = w_row + (size_t)b * 34;
            const uint8_t* x_blk = x_q8_1 + (size_t)b * 36;

            int4 w_v;
            memcpy(&w_v, w_blk + 2 + block_half * 16, sizeof(int4));
            int4 x_v;
            memcpy(&x_v, x_blk + 4 + block_half * 16, sizeof(int4));

            int dot = 0;
            dot = __builtin_amdgcn_sudot4(true, w_v.x, true, x_v.x, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w_v.y, true, x_v.y, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w_v.z, true, x_v.z, dot, false);
            dot = __builtin_amdgcn_sudot4(true, w_v.w, true, x_v.w, dot, false);

            float d_w = to_float(*reinterpret_cast<const __half*>(w_blk));
            float d_x = to_float(*reinterpret_cast<const __half*>(x_blk));

            acc += (d_w * d_x) * (float)dot;
        }
    }

    float row_sum = wave_sum(acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q8_0 mmvq variant with fdot2 (for H3.2 comparison benchmark) ─────
extern "C" __global__ void __launch_bounds__(256)
mmvq_gemv_q8_0_fdot2(
    const uint8_t* __restrict__ weight,
    const __half* __restrict__ x,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int blocks_per_row = k / 32;
    const uint8_t* w_row = weight + (size_t)row * blocks_per_row * 34;

    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; b++)
    {
        const uint8_t* w_blk = w_row + (size_t)b * 34;
        float d_w = to_float(*reinterpret_cast<const __half*>(w_blk));

        int8_t q = reinterpret_cast<const int8_t*>(w_blk + 2)[lane];
        __half w_h = to_half(d_w * (float)q);
        __half x_h = x[b * 32 + lane];

        acc += to_float(w_h) * to_float(x_h);
    }

    float row_sum = wave_sum(acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q4_K mmvq with sudot4 (H3.2 target) ──────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
mmvq_gemv_q4_k(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int num_sb = k / 256;
    const uint8_t* w_row = weight + (size_t)row * num_sb * 144;

    // Wave mapping across 2 superblocks (512 elements):
    // 32 lanes = 2 superblocks * 16 half-sub-blocks.
    const int s = lane / 16;              // superblock 0 or 1
    const int t = lane % 16;              // 0..15 within superblock
    const int p = t / 4;                  // sub-block pair 0..3
    const int h = (t / 2) % 2;            // 0: first 16 elements, 1: second 16 elements
    const int nib_sel = t % 2;            // 0: low nibble (sub-block 2p), 1: high nibble (sub-block 2p + 1)
    const int j = 2 * p + nib_sel;        // sub-block index (0..7)

    float acc = 0.0f;

    for (int base_sb = 0; base_sb < num_sb; base_sb += 2)
    {
        int sb = base_sb + s;
        if (sb < num_sb)
        {
            const uint8_t* sb_ptr = w_row + (size_t)sb * 144;
            const uint8_t* qs_ptr = sb_ptr + 16 + p * 32 + h * 16;

            int4 raw_qs;
            memcpy(&raw_qs, qs_ptr, sizeof(int4));

            int4 nibbles;
            if (nib_sel == 0)
            {
                nibbles.x = raw_qs.x & 0x0F0F0F0F;
                nibbles.y = raw_qs.y & 0x0F0F0F0F;
                nibbles.z = raw_qs.z & 0x0F0F0F0F;
                nibbles.w = raw_qs.w & 0x0F0F0F0F;
            }
            else
            {
                nibbles.x = (raw_qs.x >> 4) & 0x0F0F0F0F;
                nibbles.y = (raw_qs.y >> 4) & 0x0F0F0F0F;
                nibbles.z = (raw_qs.z >> 4) & 0x0F0F0F0F;
                nibbles.w = (raw_qs.w >> 4) & 0x0F0F0F0F;
            }

            int act_b = sb * 8 + j;
            const uint8_t* act_ptr = x_q8_1 + (size_t)act_b * 36 + 4 + h * 16;
            int4 act;
            memcpy(&act, act_ptr, sizeof(int4));

            int dot = 0;
            dot = __builtin_amdgcn_sudot4(false, nibbles.x, true, act.x, dot, false);
            dot = __builtin_amdgcn_sudot4(false, nibbles.y, true, act.y, dot, false);
            dot = __builtin_amdgcn_sudot4(false, nibbles.z, true, act.z, dot, false);
            dot = __builtin_amdgcn_sudot4(false, nibbles.w, true, act.w, dot, false);

            const uint8_t* scales = sb_ptr + 4;
            uint8_t sc_raw, mn_raw;
            if (j < 4)
            {
                sc_raw = scales[j] & 63;
                mn_raw = scales[j + 4] & 63;
            }
            else
            {
                sc_raw = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                mn_raw = (scales[j + 4] >> 4)   | ((scales[j] >> 6) << 4);
            }

            float d    = to_float(*reinterpret_cast<const __half*>(sb_ptr));
            float dmin = to_float(*reinterpret_cast<const __half*>(sb_ptr + 2));
            float sc = d * (float)sc_raw;
            float mn = dmin * (float)mn_raw;

            const uint8_t* act_blk = x_q8_1 + (size_t)act_b * 36;
            float d_x = to_float(*reinterpret_cast<const __half*>(act_blk));
            float s_x = to_float(*reinterpret_cast<const __half*>(act_blk + 2));

            float lane_val = (sc * d_x) * (float)dot;
            if (h == 0)
            {
                lane_val -= mn * s_x;
            }
            acc += lane_val;
        }
    }

    float row_sum = wave_sum(acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

// ── Q6_K mmvq with vectorized integer products (H3.2) ───────────────
extern "C" __global__ void __launch_bounds__(256)
mmvq_gemv_q6_k(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ y,
    const int n,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= n) return;

    const int num_sb = k / 256;
    const uint8_t* w_row = weight + (size_t)row * num_sb * 210;

    const int l = lane;
    const int isc = l / 16;

    float acc = 0.0f;

    for (int sb = 0; sb < num_sb; sb++)
    {
        const uint8_t* block = w_row + (size_t)sb * 210;
        const uint8_t* ql = block;
        const uint8_t* qh_base = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = to_float(*reinterpret_cast<const __half*>(block + 208));

        #pragma unroll
        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            const uint8_t* ql_half = ql + half_idx * 64;
            const uint8_t* qh_half = qh_base + half_idx * 32;
            const int8_t* sc_half = scales + half_idx * 8;

            uint8_t ql_0 = ql_half[l];
            uint8_t ql_1 = ql_half[l + 32];
            uint8_t qh   = qh_half[l];

            int q0 = ((ql_0 & 0x0F) | (((qh >> 0) & 3) << 4)) - 32;
            int q1 = ((ql_1 & 0x0F) | (((qh >> 2) & 3) << 4)) - 32;
            int q2 = ((ql_0 >> 4)   | (((qh >> 4) & 3) << 4)) - 32;
            int q3 = ((ql_1 >> 4)   | (((qh >> 6) & 3) << 4)) - 32;

            float sc0 = d * (float)sc_half[isc + 0 * 2];
            float sc1 = d * (float)sc_half[isc + 1 * 2];
            float sc2 = d * (float)sc_half[isc + 2 * 2];
            float sc3 = d * (float)sc_half[isc + 3 * 2];

            int b_base = sb * 8 + half_idx * 4;

            const uint8_t* act_blk0 = x_q8_1 + (size_t)(b_base + 0) * 36;
            const uint8_t* act_blk1 = x_q8_1 + (size_t)(b_base + 1) * 36;
            const uint8_t* act_blk2 = x_q8_1 + (size_t)(b_base + 2) * 36;
            const uint8_t* act_blk3 = x_q8_1 + (size_t)(b_base + 3) * 36;

            float d_x0 = to_float(*reinterpret_cast<const __half*>(act_blk0));
            float d_x1 = to_float(*reinterpret_cast<const __half*>(act_blk1));
            float d_x2 = to_float(*reinterpret_cast<const __half*>(act_blk2));
            float d_x3 = to_float(*reinterpret_cast<const __half*>(act_blk3));

            int8_t act0 = reinterpret_cast<const int8_t*>(act_blk0 + 4)[l];
            int8_t act1 = reinterpret_cast<const int8_t*>(act_blk1 + 4)[l];
            int8_t act2 = reinterpret_cast<const int8_t*>(act_blk2 + 4)[l];
            int8_t act3 = reinterpret_cast<const int8_t*>(act_blk3 + 4)[l];

            acc += (sc0 * d_x0) * (float)(q0 * (int)act0);
            acc += (sc1 * d_x1) * (float)(q1 * (int)act1);
            acc += (sc2 * d_x2) * (float)(q2 * (int)act2);
            acc += (sc3 * d_x3) * (float)(q3 * (int)act3);
        }
    }

    float row_sum = wave_sum(acc);
    if (lane == 0)
        y[row] = to_half(row_sum);
}

