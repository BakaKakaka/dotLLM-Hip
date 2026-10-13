// fused_gate_up_swiglu.hip - Fused Gate + Up GEMV with SwiGLU activation (H3.1).
// Computes:
//   gate[row] = W_gate[row, :] @ x + gate_bias[row]
//   up[row]   = W_up[row, :]   @ x + up_bias[row]
//   output[row] = SiLU(gate[row]) * up[row] = (gate / (1 + exp(-gate))) * up
//
// Key performance characteristics:
//   - Exactly ONE wave (32 lanes) per output row.
//   - Input vector x is read once from L1/L2 and shared between Gate and Up dot products.
//   - Eliminates two separate global memory write-then-read roundtrips for FfnGate and FfnUp.
//   - Eliminates the standalone SwiGLU kernel launch entirely.
//   - Result written directly to intermediate FP16 SiluOutput.
#include "rdna_common.h"
#include <stdint.h>
#include <string.h>

// ── Q8_0: 34 bytes per 32 values ────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_q8_0(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int blocks_per_row = k / 32;
    const uint8_t* g_row = gate_weight + (size_t)row * blocks_per_row * 34;
    const uint8_t* u_row = up_weight   + (size_t)row * blocks_per_row * 34;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = 0; b < blocks_per_row; b++)
    {
        float x_val = to_float(x[b * 32 + lane]);

        // Gate
        const uint8_t* g_block = g_row + (size_t)b * 34;
        float g_d = to_float(*reinterpret_cast<const __half*>(g_block));
        int8_t g_q = reinterpret_cast<const int8_t*>(g_block + 2)[lane];
        gate_acc += (g_d * (float)g_q) * x_val;

        // Up
        const uint8_t* u_block = u_row + (size_t)b * 34;
        float u_d = to_float(*reinterpret_cast<const __half*>(u_block));
        int8_t u_q = reinterpret_cast<const int8_t*>(u_block + 2)[lane];
        up_acc += (u_d * (float)u_q) * x_val;
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── Q4_0: 18 bytes per 32 values ────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_q4_0(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int blocks_per_row = k / 32;
    const uint8_t* g_row = gate_weight + (size_t)row * blocks_per_row * 18;
    const uint8_t* u_row = up_weight   + (size_t)row * blocks_per_row * 18;

    int byte_idx = lane / 2;
    int shift = (lane & 1) ? 4 : 0;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = 0; b < blocks_per_row; b++)
    {
        float x_val = to_float(x[b * 32 + lane]);

        // Gate
        const uint8_t* g_block = g_row + (size_t)b * 18;
        float g_d = to_float(*reinterpret_cast<const __half*>(g_block));
        uint8_t g_packed = g_block[2 + byte_idx];
        int g_val = (int)((g_packed >> shift) & 0x0F) - 8;
        gate_acc += (g_d * (float)g_val) * x_val;

        // Up
        const uint8_t* u_block = u_row + (size_t)b * 18;
        float u_d = to_float(*reinterpret_cast<const __half*>(u_block));
        uint8_t u_packed = u_block[2 + byte_idx];
        int u_val = (int)((u_packed >> shift) & 0x0F) - 8;
        up_acc += (u_d * (float)u_val) * x_val;
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── Q5_0: 22 bytes per 32 values ────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_q5_0(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int blocks_per_row = k / 32;
    const uint8_t* g_row = gate_weight + (size_t)row * blocks_per_row * 22;
    const uint8_t* u_row = up_weight   + (size_t)row * blocks_per_row * 22;

    int j = lane < 16 ? lane : lane - 16;
    int shift = (lane < 16) ? 0 : 4;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int b = 0; b < blocks_per_row; b++)
    {
        float x_val = to_float(x[b * 32 + lane]);

        // Gate
        const uint8_t* g_block = g_row + (size_t)b * 22;
        float g_d = to_float(*reinterpret_cast<const __half*>(g_block));
        unsigned int g_qh = (unsigned int)g_block[2] | ((unsigned int)g_block[3] << 8) |
                            ((unsigned int)g_block[4] << 16) | ((unsigned int)g_block[5] << 24);
        uint8_t g_packed = g_block[6 + j];
        int g_nibble = (g_packed >> shift) & 0x0F;
        int g_high_bit = (g_qh >> lane) & 1;
        int g_val = (g_nibble | (g_high_bit << 4)) - 16;
        gate_acc += (g_d * (float)g_val) * x_val;

        // Up
        const uint8_t* u_block = u_row + (size_t)b * 22;
        float u_d = to_float(*reinterpret_cast<const __half*>(u_block));
        unsigned int u_qh = (unsigned int)u_block[2] | ((unsigned int)u_block[3] << 8) |
                            ((unsigned int)u_block[4] << 16) | ((unsigned int)u_block[5] << 24);
        uint8_t u_packed = u_block[6 + j];
        int u_nibble = (u_packed >> shift) & 0x0F;
        int u_high_bit = (u_qh >> lane) & 1;
        int u_val = (u_nibble | (u_high_bit << 4)) - 16;
        up_acc += (u_d * (float)u_val) * x_val;
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── Q4_K: 144 bytes per 256 values ──────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_q4_k(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* g_row = gate_weight + (size_t)row * superblocks_per_row * 144;
    const uint8_t* u_row = up_weight   + (size_t)row * superblocks_per_row * 144;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; sb++)
    {
        // Gate superblock
        const uint8_t* g_block = g_row + (size_t)sb * 144;
        float g_d = to_float(*reinterpret_cast<const __half*>(g_block));
        float g_dmin = to_float(*reinterpret_cast<const __half*>(g_block + 2));
        const uint8_t* g_scales_raw = g_block + 4;
        const uint8_t* g_qs = g_block + 16;

        // Up superblock
        const uint8_t* u_block = u_row + (size_t)sb * 144;
        float u_d = to_float(*reinterpret_cast<const __half*>(u_block));
        float u_dmin = to_float(*reinterpret_cast<const __half*>(u_block + 2));
        const uint8_t* u_scales_raw = u_block + 4;
        const uint8_t* u_qs = u_block + 16;

        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd = pair * 2 + 1;

            float x0 = to_float(x[sb * 256 + sb_even * 32 + lane]);
            float x1 = to_float(x[sb * 256 + sb_odd * 32 + lane]);

            // Gate scales & mins
            int g_sc0, g_m0, g_sc1, g_m1;
            if (sb_even < 4)
            {
                g_sc0 = g_scales_raw[sb_even] & 0x3F;
                g_m0  = g_scales_raw[sb_even + 4] & 0x3F;
                g_sc1 = g_scales_raw[sb_odd] & 0x3F;
                g_m1  = g_scales_raw[sb_odd + 4] & 0x3F;
            }
            else
            {
                g_sc0 = (g_scales_raw[sb_even + 4] & 0x0F) | ((g_scales_raw[sb_even - 4] >> 6) << 4);
                g_m0  = (g_scales_raw[sb_even + 4] >> 4)   | ((g_scales_raw[sb_even] >> 6) << 4);
                g_sc1 = (g_scales_raw[sb_odd + 4] & 0x0F)  | ((g_scales_raw[sb_odd - 4] >> 6) << 4);
                g_m1  = (g_scales_raw[sb_odd + 4] >> 4)    | ((g_scales_raw[sb_odd] >> 6) << 4);
            }

            uint8_t g_byte_val = g_qs[pair * 32 + lane];
            int g_nibble0 = g_byte_val & 0x0F;
            int g_nibble1 = g_byte_val >> 4;

            float w0_g = g_d * (float)g_sc0 * (float)g_nibble0 - g_dmin * (float)g_m0;
            float w1_g = g_d * (float)g_sc1 * (float)g_nibble1 - g_dmin * (float)g_m1;
            gate_acc += w0_g * x0 + w1_g * x1;

            // Up scales & mins
            int u_sc0, u_m0, u_sc1, u_m1;
            if (sb_even < 4)
            {
                u_sc0 = u_scales_raw[sb_even] & 0x3F;
                u_m0  = u_scales_raw[sb_even + 4] & 0x3F;
                u_sc1 = u_scales_raw[sb_odd] & 0x3F;
                u_m1  = u_scales_raw[sb_odd + 4] & 0x3F;
            }
            else
            {
                u_sc0 = (u_scales_raw[sb_even + 4] & 0x0F) | ((u_scales_raw[sb_even - 4] >> 6) << 4);
                u_m0  = (u_scales_raw[sb_even + 4] >> 4)   | ((u_scales_raw[sb_even] >> 6) << 4);
                u_sc1 = (u_scales_raw[sb_odd + 4] & 0x0F)  | ((u_scales_raw[sb_odd - 4] >> 6) << 4);
                u_m1  = (u_scales_raw[sb_odd + 4] >> 4)    | ((u_scales_raw[sb_odd] >> 6) << 4);
            }

            uint8_t u_byte_val = u_qs[pair * 32 + lane];
            int u_nibble0 = u_byte_val & 0x0F;
            int u_nibble1 = u_byte_val >> 4;

            float w0_u = u_d * (float)u_sc0 * (float)u_nibble0 - u_dmin * (float)u_m0;
            float w1_u = u_d * (float)u_sc1 * (float)u_nibble1 - u_dmin * (float)u_m1;
            up_acc += w0_u * x0 + w1_u * x1;
        }
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── Q5_K: 176 bytes per 256 values ──────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_q5_k(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* g_row = gate_weight + (size_t)row * superblocks_per_row * 176;
    const uint8_t* u_row = up_weight   + (size_t)row * superblocks_per_row * 176;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; sb++)
    {
        // Gate
        const uint8_t* g_block = g_row + (size_t)sb * 176;
        float g_d = to_float(*reinterpret_cast<const __half*>(g_block));
        float g_dmin = to_float(*reinterpret_cast<const __half*>(g_block + 2));
        const uint8_t* g_scales_raw = g_block + 4;
        const uint8_t* g_qh = g_block + 16;
        const uint8_t* g_qs = g_block + 48;

        // Up
        const uint8_t* u_block = u_row + (size_t)sb * 176;
        float u_d = to_float(*reinterpret_cast<const __half*>(u_block));
        float u_dmin = to_float(*reinterpret_cast<const __half*>(u_block + 2));
        const uint8_t* u_scales_raw = u_block + 4;
        const uint8_t* u_qh = u_block + 16;
        const uint8_t* u_qs = u_block + 48;

        for (int sub = 0; sub < 8; sub++)
        {
            float x_val = to_float(x[sb * 256 + sub * 32 + lane]);

            // Gate sub
            int g_sc, g_m;
            if (sub < 4)
            {
                g_sc = g_scales_raw[sub] & 0x3F;
                g_m  = g_scales_raw[sub + 4] & 0x3F;
            }
            else
            {
                g_sc = (g_scales_raw[sub + 4] & 0x0F) | ((g_scales_raw[sub - 4] >> 6) << 4);
                g_m  = (g_scales_raw[sub + 4] >> 4)   | ((g_scales_raw[sub] >> 6) << 4);
            }
            float g_scale = g_d * (float)g_sc;
            float g_min_val = g_dmin * (float)g_m;

            int pair = sub / 2;
            int is_odd = sub % 2;
            uint8_t g_byte_val = g_qs[pair * 32 + lane];
            int g_nibble = is_odd ? (g_byte_val >> 4) : (g_byte_val & 0x0F);
            int g_bit5 = (g_qh[lane] >> sub) & 1;
            int g_val = g_nibble | (g_bit5 << 4);
            float w_g = g_scale * (float)g_val - g_min_val;
            gate_acc += w_g * x_val;

            // Up sub
            int u_sc, u_m;
            if (sub < 4)
            {
                u_sc = u_scales_raw[sub] & 0x3F;
                u_m  = u_scales_raw[sub + 4] & 0x3F;
            }
            else
            {
                u_sc = (u_scales_raw[sub + 4] & 0x0F) | ((u_scales_raw[sub - 4] >> 6) << 4);
                u_m  = (u_scales_raw[sub + 4] >> 4)   | ((u_scales_raw[sub] >> 6) << 4);
            }
            float u_scale = u_d * (float)u_sc;
            float u_min_val = u_dmin * (float)u_m;

            uint8_t u_byte_val = u_qs[pair * 32 + lane];
            int u_nibble = is_odd ? (u_byte_val >> 4) : (u_byte_val & 0x0F);
            int u_bit5 = (u_qh[lane] >> sub) & 1;
            int u_val = u_nibble | (u_bit5 << 4);
            float w_u = u_scale * (float)u_val - u_min_val;
            up_acc += w_u * x_val;
        }
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── Q6_K: 210 bytes per 256 values ──────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_q6_k(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* g_row = gate_weight + (size_t)row * superblocks_per_row * 210;
    const uint8_t* u_row = up_weight   + (size_t)row * superblocks_per_row * 210;

    int l = lane;
    int isc = l / 16;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int sb = 0; sb < superblocks_per_row; sb++)
    {
        // Gate
        const uint8_t* g_block = g_row + (size_t)sb * 210;
        const uint8_t* g_ql = g_block;
        const uint8_t* g_qh_base = g_block + 128;
        const int8_t* g_scales = reinterpret_cast<const int8_t*>(g_block + 192);
        float g_d = to_float(*reinterpret_cast<const __half*>(g_block + 208));

        // Up
        const uint8_t* u_block = u_row + (size_t)sb * 210;
        const uint8_t* u_ql = u_block;
        const uint8_t* u_qh_base = u_block + 128;
        const int8_t* u_scales = reinterpret_cast<const int8_t*>(u_block + 192);
        float u_d = to_float(*reinterpret_cast<const __half*>(u_block + 208));

        for (int half_idx = 0; half_idx < 2; half_idx++)
        {
            int base_k = sb * 256 + half_idx * 128 + l;
            float x0 = to_float(x[base_k + 0 * 32]);
            float x1 = to_float(x[base_k + 1 * 32]);
            float x2 = to_float(x[base_k + 2 * 32]);
            float x3 = to_float(x[base_k + 3 * 32]);

            // Gate half
            const uint8_t* g_ql_half = g_ql + half_idx * 64;
            const uint8_t* g_qh_half = g_qh_base + half_idx * 32;
            const int8_t* g_sc_half = g_scales + half_idx * 8;

            int g_q0 = ((g_ql_half[l] & 0x0F) | (((g_qh_half[l] >> 0) & 3) << 4)) - 32;
            int g_q1 = ((g_ql_half[l + 32] & 0x0F) | (((g_qh_half[l] >> 2) & 3) << 4)) - 32;
            int g_q2 = ((g_ql_half[l] >> 4) | (((g_qh_half[l] >> 4) & 3) << 4)) - 32;
            int g_q3 = ((g_ql_half[l + 32] >> 4) | (((g_qh_half[l] >> 6) & 3) << 4)) - 32;

            float g_sc0 = g_d * (float)g_sc_half[isc + 0 * 2];
            float g_sc1 = g_d * (float)g_sc_half[isc + 1 * 2];
            float g_sc2 = g_d * (float)g_sc_half[isc + 2 * 2];
            float g_sc3 = g_d * (float)g_sc_half[isc + 3 * 2];

            gate_acc += (g_sc0 * (float)g_q0) * x0 + (g_sc1 * (float)g_q1) * x1 +
                        (g_sc2 * (float)g_q2) * x2 + (g_sc3 * (float)g_q3) * x3;

            // Up half
            const uint8_t* u_ql_half = u_ql + half_idx * 64;
            const uint8_t* u_qh_half = u_qh_base + half_idx * 32;
            const int8_t* u_sc_half = u_scales + half_idx * 8;

            int u_q0 = ((u_ql_half[l] & 0x0F) | (((u_qh_half[l] >> 0) & 3) << 4)) - 32;
            int u_q1 = ((u_ql_half[l + 32] & 0x0F) | (((u_qh_half[l] >> 2) & 3) << 4)) - 32;
            int u_q2 = ((u_ql_half[l] >> 4) | (((u_qh_half[l] >> 4) & 3) << 4)) - 32;
            int u_q3 = ((u_ql_half[l + 32] >> 4) | (((u_qh_half[l] >> 6) & 3) << 4)) - 32;

            float u_sc0 = u_d * (float)u_sc_half[isc + 0 * 2];
            float u_sc1 = u_d * (float)u_sc_half[isc + 1 * 2];
            float u_sc2 = u_d * (float)u_sc_half[isc + 2 * 2];
            float u_sc3 = u_d * (float)u_sc_half[isc + 3 * 2];

            up_acc += (u_sc0 * (float)u_q0) * x0 + (u_sc1 * (float)u_q1) * x1 +
                      (u_sc2 * (float)u_q2) * x2 + (u_sc3 * (float)u_q3) * x3;
        }
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── FP16: 2 bytes per value ─────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_f16(
    const __half* __restrict__ gate_weight,
    const __half* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const __half* __restrict__ x,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    int wave_in_block = threadIdx.x / WAVE;
    int lane = threadIdx.x % WAVE;
    int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const __half* g_row = gate_weight + (size_t)row * k;
    const __half* u_row = up_weight   + (size_t)row * k;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int col = lane; col < k; col += WAVE)
    {
        float x_val = to_float(x[col]);
        gate_acc += to_float(g_row[col]) * x_val;
        up_acc   += to_float(u_row[col]) * x_val;
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── MMVQ Fused Gate+Up (H3.2): Q8_0 weights with Q8_1 activation ────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_mmvq_q8_0(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int blocks_per_row = k / 32;
    const uint8_t* g_row = gate_weight + (size_t)row * blocks_per_row * 34;
    const uint8_t* u_row = up_weight   + (size_t)row * blocks_per_row * 34;

    const int block_half = lane & 1;       // 0: first 16 bytes, 1: second 16 bytes
    const int block_offset = lane >> 1;     // block within tile of 16 blocks (0..15)

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int base_b = 0; base_b < blocks_per_row; base_b += 16)
    {
        int b = base_b + block_offset;
        if (b < blocks_per_row)
        {
            const uint8_t* x_blk = x_q8_1 + (size_t)b * 36;
            int4 x_v;
            memcpy(&x_v, x_blk + 4 + block_half * 16, sizeof(int4));
            float d_x = to_float(*reinterpret_cast<const __half*>(x_blk));

            // Gate
            const uint8_t* g_blk = g_row + (size_t)b * 34;
            int4 g_v;
            memcpy(&g_v, g_blk + 2 + block_half * 16, sizeof(int4));
            int g_dot = 0;
            g_dot = __builtin_amdgcn_sudot4(true, g_v.x, true, x_v.x, g_dot, false);
            g_dot = __builtin_amdgcn_sudot4(true, g_v.y, true, x_v.y, g_dot, false);
            g_dot = __builtin_amdgcn_sudot4(true, g_v.z, true, x_v.z, g_dot, false);
            g_dot = __builtin_amdgcn_sudot4(true, g_v.w, true, x_v.w, g_dot, false);
            float g_d = to_float(*reinterpret_cast<const __half*>(g_blk));
            gate_acc += (g_d * d_x) * (float)g_dot;

            // Up
            const uint8_t* u_blk = u_row + (size_t)b * 34;
            int4 u_v;
            memcpy(&u_v, u_blk + 2 + block_half * 16, sizeof(int4));
            int u_dot = 0;
            u_dot = __builtin_amdgcn_sudot4(true, u_v.x, true, x_v.x, u_dot, false);
            u_dot = __builtin_amdgcn_sudot4(true, u_v.y, true, x_v.y, u_dot, false);
            u_dot = __builtin_amdgcn_sudot4(true, u_v.z, true, x_v.z, u_dot, false);
            u_dot = __builtin_amdgcn_sudot4(true, u_v.w, true, x_v.w, u_dot, false);
            float u_d = to_float(*reinterpret_cast<const __half*>(u_blk));
            up_acc += (u_d * d_x) * (float)u_dot;
        }
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}

// ── MMVQ Fused Gate+Up (H3.2): Q4_K weights with Q8_1 activation ────
extern "C" __global__ void __launch_bounds__(256)
fused_gate_up_swiglu_mmvq_q4_k(
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ up_weight,
    const __half* __restrict__ gate_bias,
    const __half* __restrict__ up_bias,
    const uint8_t* __restrict__ x_q8_1,
    __half* __restrict__ output,
    const int intermediate_size,
    const int k)
{
    const int wave_in_block = threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int row = blockIdx.x * (blockDim.x / WAVE) + wave_in_block;
    if (row >= intermediate_size) return;

    const int num_sb = k / 256;
    const uint8_t* g_row = gate_weight + (size_t)row * num_sb * 144;
    const uint8_t* u_row = up_weight   + (size_t)row * num_sb * 144;

    const int s = lane / 16;              // superblock 0 or 1
    const int t = lane % 16;              // 0..15 within superblock
    const int p = t / 4;                  // sub-block pair 0..3
    const int h = (t / 2) % 2;            // 0: first 16 elements, 1: second 16 elements
    const int nib_sel = t % 2;            // 0: low nibble (sub-block 2p), 1: high nibble (sub-block 2p + 1)
    const int j = 2 * p + nib_sel;        // sub-block index (0..7)

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (int base_sb = 0; base_sb < num_sb; base_sb += 2)
    {
        int sb = base_sb + s;
        if (sb < num_sb)
        {
            // ── Activation (loaded once for both Gate and Up) ──
            int act_b = sb * 8 + j;
            const uint8_t* act_ptr = x_q8_1 + (size_t)act_b * 36 + 4 + h * 16;
            int4 act;
            memcpy(&act, act_ptr, sizeof(int4));

            const uint8_t* act_blk = x_q8_1 + (size_t)act_b * 36;
            float d_x = to_float(*reinterpret_cast<const __half*>(act_blk));
            float s_x = to_float(*reinterpret_cast<const __half*>(act_blk + 2));

            // ── Gate projection ──
            const uint8_t* g_sb_ptr = g_row + (size_t)sb * 144;
            const uint8_t* g_qs_ptr = g_sb_ptr + 16 + p * 32 + h * 16;
            int4 g_raw_qs;
            memcpy(&g_raw_qs, g_qs_ptr, sizeof(int4));

            int4 g_nibbles;
            if (nib_sel == 0)
            {
                g_nibbles.x = g_raw_qs.x & 0x0F0F0F0F;
                g_nibbles.y = g_raw_qs.y & 0x0F0F0F0F;
                g_nibbles.z = g_raw_qs.z & 0x0F0F0F0F;
                g_nibbles.w = g_raw_qs.w & 0x0F0F0F0F;
            }
            else
            {
                g_nibbles.x = (g_raw_qs.x >> 4) & 0x0F0F0F0F;
                g_nibbles.y = (g_raw_qs.y >> 4) & 0x0F0F0F0F;
                g_nibbles.z = (g_raw_qs.z >> 4) & 0x0F0F0F0F;
                g_nibbles.w = (g_raw_qs.w >> 4) & 0x0F0F0F0F;
            }

            int g_dot = 0;
            g_dot = __builtin_amdgcn_sudot4(false, g_nibbles.x, true, act.x, g_dot, false);
            g_dot = __builtin_amdgcn_sudot4(false, g_nibbles.y, true, act.y, g_dot, false);
            g_dot = __builtin_amdgcn_sudot4(false, g_nibbles.z, true, act.z, g_dot, false);
            g_dot = __builtin_amdgcn_sudot4(false, g_nibbles.w, true, act.w, g_dot, false);

            const uint8_t* g_scales = g_sb_ptr + 4;
            uint8_t g_sc_raw, g_mn_raw;
            if (j < 4)
            {
                g_sc_raw = g_scales[j] & 63;
                g_mn_raw = g_scales[j + 4] & 63;
            }
            else
            {
                g_sc_raw = (g_scales[j + 4] & 0x0F) | ((g_scales[j - 4] >> 6) << 4);
                g_mn_raw = (g_scales[j + 4] >> 4)   | ((g_scales[j] >> 6) << 4);
            }

            float g_d    = to_float(*reinterpret_cast<const __half*>(g_sb_ptr));
            float g_dmin = to_float(*reinterpret_cast<const __half*>(g_sb_ptr + 2));
            float g_sc = g_d * (float)g_sc_raw;
            float g_mn = g_dmin * (float)g_mn_raw;

            float g_lane_val = (g_sc * d_x) * (float)g_dot;
            if (h == 0)
            {
                g_lane_val -= g_mn * s_x;
            }
            gate_acc += g_lane_val;

            // ── Up projection ──
            const uint8_t* u_sb_ptr = u_row + (size_t)sb * 144;
            const uint8_t* u_qs_ptr = u_sb_ptr + 16 + p * 32 + h * 16;
            int4 u_raw_qs;
            memcpy(&u_raw_qs, u_qs_ptr, sizeof(int4));

            int4 u_nibbles;
            if (nib_sel == 0)
            {
                u_nibbles.x = u_raw_qs.x & 0x0F0F0F0F;
                u_nibbles.y = u_raw_qs.y & 0x0F0F0F0F;
                u_nibbles.z = u_raw_qs.z & 0x0F0F0F0F;
                u_nibbles.w = u_raw_qs.w & 0x0F0F0F0F;
            }
            else
            {
                u_nibbles.x = (u_raw_qs.x >> 4) & 0x0F0F0F0F;
                u_nibbles.y = (u_raw_qs.y >> 4) & 0x0F0F0F0F;
                u_nibbles.z = (u_raw_qs.z >> 4) & 0x0F0F0F0F;
                u_nibbles.w = (u_raw_qs.w >> 4) & 0x0F0F0F0F;
            }

            int u_dot = 0;
            u_dot = __builtin_amdgcn_sudot4(false, u_nibbles.x, true, act.x, u_dot, false);
            u_dot = __builtin_amdgcn_sudot4(false, u_nibbles.y, true, act.y, u_dot, false);
            u_dot = __builtin_amdgcn_sudot4(false, u_nibbles.z, true, act.z, u_dot, false);
            u_dot = __builtin_amdgcn_sudot4(false, u_nibbles.w, true, act.w, u_dot, false);

            const uint8_t* u_scales = u_sb_ptr + 4;
            uint8_t u_sc_raw, u_mn_raw;
            if (j < 4)
            {
                u_sc_raw = u_scales[j] & 63;
                u_mn_raw = u_scales[j + 4] & 63;
            }
            else
            {
                u_sc_raw = (u_scales[j + 4] & 0x0F) | ((u_scales[j - 4] >> 6) << 4);
                u_mn_raw = (u_scales[j + 4] >> 4)   | ((u_scales[j] >> 6) << 4);
            }

            float u_d    = to_float(*reinterpret_cast<const __half*>(u_sb_ptr));
            float u_dmin = to_float(*reinterpret_cast<const __half*>(u_sb_ptr + 2));
            float u_sc = u_d * (float)u_sc_raw;
            float u_mn = u_dmin * (float)u_mn_raw;

            float u_lane_val = (u_sc * d_x) * (float)u_dot;
            if (h == 0)
            {
                u_lane_val -= u_mn * s_x;
            }
            up_acc += u_lane_val;
        }
    }

    float gate_sum = wave_sum(gate_acc);
    float up_sum   = wave_sum(up_acc);

    if (lane == 0)
    {
        if (gate_bias != nullptr) gate_sum += to_float(gate_bias[row]);
        if (up_bias != nullptr)   up_sum   += to_float(up_bias[row]);

        float silu_gate = gate_sum / (1.0f + expf(-gate_sum));
        output[row] = to_half(silu_gate * up_sum);
    }
}
