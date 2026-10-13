// fused_qkv_gemv.hip - Fused Q, K, V GEMV projection kernel (H3.1).
// Computes Q, K, and V linear projections in a single grid launch:
//   Q[r] = W_q[r, :] @ x + q_bias[r]
//   K[r] = W_k[r, :] @ x + k_bias[r]
//   V[r] = W_v[r, :] @ x + v_bias[r]
//
// Key performance characteristics:
//   - Exactly ONE wave (32 lanes) per output row across the combined (dim_q + dim_k + dim_v) rows.
//   - Wave divergence is zero: the wave index is identical across all lanes of a wave.
//   - Replaces 3 separate GEMV kernel launches with a single grid launch.
//   - Replaces optional separate QBias, KBias, VBias kernel launches.
//   - Input vector x is loaded once into L2/MALL and shared across all 3 projections.
//   - During decode, v_out can point directly to the GPU KV-cache row, eliminating copy overhead.
#include "rdna_common.h"
#include <stdint.h>
#include <string.h>

// ── Q8_0 Helpers ────────────────────────────────────────────────────
__device__ __forceinline__ float dot_q8_0(const uint8_t* w_row, const __half* x, int k, int lane)
{
    const int blocks = k / 32;
    float acc = 0.0f;
    for (int b = 0; b < blocks; b++)
    {
        const uint8_t* block = w_row + (size_t)b * 34;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];
        acc += (d * (float)q) * to_float(x[b * 32 + lane]);
    }
    return wave_sum(acc);
}

extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_q8_0(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int blocks_per_row = k / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float sum = dot_q8_0(w_row, x, k, lane);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── Q4_0 Helpers ────────────────────────────────────────────────────
__device__ __forceinline__ float dot_q4_0(const uint8_t* w_row, const __half* x, int k, int lane)
{
    const int blocks = k / 32;
    int byte_idx = lane / 2;
    int shift = (lane & 1) ? 4 : 0;
    float acc = 0.0f;
    for (int b = 0; b < blocks; b++)
    {
        const uint8_t* block = w_row + (size_t)b * 18;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        uint8_t packed = block[2 + byte_idx];
        int val = (int)((packed >> shift) & 0x0F) - 8;
        acc += (d * (float)val) * to_float(x[b * 32 + lane]);
    }
    return wave_sum(acc);
}

extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_q4_0(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int blocks_per_row = k / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 18;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float sum = dot_q4_0(w_row, x, k, lane);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── Q5_0 Helpers ────────────────────────────────────────────────────
__device__ __forceinline__ float dot_q5_0(const uint8_t* w_row, const __half* x, int k, int lane)
{
    const int blocks = k / 32;
    int j = lane < 16 ? lane : lane - 16;
    int shift = (lane < 16) ? 0 : 4;
    float acc = 0.0f;
    for (int b = 0; b < blocks; b++)
    {
        const uint8_t* block = w_row + (size_t)b * 22;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) |
                          ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
        uint8_t packed = block[6 + j];
        int nibble = (packed >> shift) & 0x0F;
        int high_bit = (qh >> lane) & 1;
        int val = (nibble | (high_bit << 4)) - 16;
        acc += (d * (float)val) * to_float(x[b * 32 + lane]);
    }
    return wave_sum(acc);
}

extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_q5_0(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int blocks_per_row = k / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 22;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float sum = dot_q5_0(w_row, x, k, lane);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── Q4_K Helpers ────────────────────────────────────────────────────
__device__ __forceinline__ float dot_q4_k(const uint8_t* w_row, const __half* x, int k, int lane)
{
    const int superblocks = k / 256;
    float acc = 0.0f;
    for (int sb = 0; sb < superblocks; sb++)
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

            acc += w0 * to_float(x[sb * 256 + sb_even * 32 + lane]);
            acc += w1 * to_float(x[sb * 256 + sb_odd * 32 + lane]);
        }
    }
    return wave_sum(acc);
}

extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_q4_k(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int superblocks_per_row = k / 256;
    const size_t row_bytes = (size_t)superblocks_per_row * 144;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float sum = dot_q4_k(w_row, x, k, lane);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── Q5_K Helpers ────────────────────────────────────────────────────
__device__ __forceinline__ float dot_q5_k(const uint8_t* w_row, const __half* x, int k, int lane)
{
    const int superblocks = k / 256;
    float acc = 0.0f;
    for (int sb = 0; sb < superblocks; sb++)
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
            acc += w * to_float(x[sb * 256 + sub * 32 + lane]);
        }
    }
    return wave_sum(acc);
}

extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_q5_k(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int superblocks_per_row = k / 256;
    const size_t row_bytes = (size_t)superblocks_per_row * 176;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float sum = dot_q5_k(w_row, x, k, lane);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── Q6_K Helpers ────────────────────────────────────────────────────
__device__ __forceinline__ float dot_q6_k(const uint8_t* w_row, const __half* x, int k, int lane)
{
    const int superblocks = k / 256;
    int l = lane;
    int isc = l / 16;
    float acc = 0.0f;
    for (int sb = 0; sb < superblocks; sb++)
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
            acc += (sc0 * (float)q0) * to_float(x[base_k + 0 * 32]);
            acc += (sc1 * (float)q1) * to_float(x[base_k + 1 * 32]);
            acc += (sc2 * (float)q2) * to_float(x[base_k + 2 * 32]);
            acc += (sc3 * (float)q3) * to_float(x[base_k + 3 * 32]);
        }
    }
    return wave_sum(acc);
}

extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_q6_k(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int superblocks_per_row = k / 256;
    const size_t row_bytes = (size_t)superblocks_per_row * 210;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float sum = dot_q6_k(w_row, x, k, lane);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── FP16 Helpers ────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_f16(
    const __half* __restrict__ q_weight,
    const __half* __restrict__ k_weight,
    const __half* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const __half* __restrict__ x,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const __half* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * k;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * k;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * k;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    float acc = 0.0f;
    for (int col = lane; col < k; col += WAVE)
    {
        acc += to_float(w_row[col]) * to_float(x[col]);
    }

    float sum = wave_sum(acc);
    if (lane == 0)
    {
        if (bias_ptr != nullptr) sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(sum);
    }
}

// ── MMVQ Fused QKV (H3.2): Q8_0 weights with Q8_1 activation ────────
extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_mmvq_q8_0(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    const int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int blocks_per_row = k / 32;
    const size_t row_bytes = (size_t)blocks_per_row * 34;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    const int block_half = lane & 1;
    const int block_offset = lane >> 1;

    float acc = 0.0f;

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
    {
        if (bias_ptr != nullptr) row_sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(row_sum);
    }
}

// ── MMVQ Fused QKV (H3.2): Q4_K weights with Q8_1 activation ────────
extern "C" __global__ void __launch_bounds__(256)
fused_qkv_gemv_mmvq_q4_k(
    const uint8_t* __restrict__ q_weight,
    const uint8_t* __restrict__ k_weight,
    const uint8_t* __restrict__ v_weight,
    const __half* __restrict__ q_bias,
    const __half* __restrict__ k_bias,
    const __half* __restrict__ v_bias,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    __half* __restrict__ v_out,
    const int dim_q,
    const int dim_k,
    const int dim_v,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    const int total_rows = dim_q + dim_k + dim_v;
    if (row >= total_rows) return;

    const int num_sb = k / 256;
    const size_t row_bytes = (size_t)num_sb * 144;

    const uint8_t* w_row;
    __half* out_ptr;
    const __half* bias_ptr;
    int out_idx;

    if (row < dim_q)
    {
        w_row = q_weight + (size_t)row * row_bytes;
        out_ptr = q_out;
        bias_ptr = q_bias;
        out_idx = row;
    }
    else if (row < dim_q + dim_k)
    {
        int r = row - dim_q;
        w_row = k_weight + (size_t)r * row_bytes;
        out_ptr = k_out;
        bias_ptr = k_bias;
        out_idx = r;
    }
    else
    {
        int r = row - dim_q - dim_k;
        w_row = v_weight + (size_t)r * row_bytes;
        out_ptr = v_out;
        bias_ptr = v_bias;
        out_idx = r;
    }

    const int s = lane / 16;
    const int t = lane % 16;
    const int p = t / 4;
    const int h = (t / 2) % 2;
    const int nib_sel = t % 2;
    const int j = 2 * p + nib_sel;

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
    {
        if (bias_ptr != nullptr) row_sum += to_float(bias_ptr[out_idx]);
        out_ptr[out_idx] = to_half(row_sum);
    }
}

