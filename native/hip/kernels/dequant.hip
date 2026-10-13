// dequant.hip - Dequantization kernels for DotLLM.Hip (H2.3).
// Converts quantized weight blocks (Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K) to FP16.
#include "rdna_common.h"
#include <stdint.h>

#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 34

#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18

#define Q5_0_BLOCK_SIZE 32
#define Q5_0_BLOCK_BYTES 22

#define Q4_K_SUPER_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144

#define Q5_K_SUPER_BLOCK_SIZE 256
#define Q5_K_BLOCK_BYTES 176

#define Q6_K_SUPER_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES 210

// ── Q8_0 ────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
dequant_q8_0_f16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q8_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q8_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q8_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q8_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q8_0_BLOCK_BYTES;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        int8_t q = reinterpret_cast<const int8_t*>(block + 2)[lane];

        dst[(size_t)block_idx * Q8_0_BLOCK_SIZE + lane] = to_half(d * (float)q);
    }
}

// ── Q4_0 ────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
dequant_q4_0_f16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q4_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q4_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q4_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q4_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q4_0_BLOCK_BYTES;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        const uint8_t* qs = block + 2;

        int byte_idx = lane / 2;
        uint8_t packed = qs[byte_idx];
        int val = (lane & 1) ? ((int)(packed >> 4) - 8) : ((int)(packed & 0x0F) - 8);

        dst[(size_t)block_idx * Q4_0_BLOCK_SIZE + lane] = to_half(d * (float)val);
    }
}

// ── Q5_0 ────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
dequant_q5_0_f16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    const int total_blocks)
{
    int lane = threadIdx.x % Q5_0_BLOCK_SIZE;
    int warp_in_block = threadIdx.x / Q5_0_BLOCK_SIZE;
    int warps_per_grid = (gridDim.x * blockDim.x) / Q5_0_BLOCK_SIZE;
    int start_block = blockIdx.x * (blockDim.x / Q5_0_BLOCK_SIZE) + warp_in_block;

    for (int block_idx = start_block; block_idx < total_blocks; block_idx += warps_per_grid)
    {
        const uint8_t* block = src + (size_t)block_idx * Q5_0_BLOCK_BYTES;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8) |
                          ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
        const uint8_t* qs = block + 6;

        int j = lane < 16 ? lane : lane - 16;
        uint8_t packed = qs[j];
        int nibble = (lane < 16) ? (packed & 0x0F) : (packed >> 4);
        int high_bit = (qh >> lane) & 1;
        int val = (nibble | (high_bit << 4)) - 16;

        dst[(size_t)block_idx * Q5_0_BLOCK_SIZE + lane] = to_half(d * (float)val);
    }
}

// ── Q4_K ────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
dequant_q4_k_f16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q4_K_BLOCK_BYTES;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        float dmin = to_float(*reinterpret_cast<const __half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        int pair = t / 64;
        int pos_in_pair = t % 64;
        int is_odd = pos_in_pair / 32;
        int j = pos_in_pair % 32;

        int sb_even = pair * 2;
        int sb_odd = pair * 2 + 1;
        int sb_cur = is_odd ? sb_odd : sb_even;

        int sc, m;
        if (sb_cur < 4)
        {
            sc = scales_raw[sb_cur] & 0x3F;
            m = scales_raw[sb_cur + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sb_cur + 4] & 0x0F) | ((scales_raw[sb_cur - 4] >> 6) << 4);
            m = (scales_raw[sb_cur + 4] >> 4) | ((scales_raw[sb_cur] >> 6) << 4);
        }

        uint8_t byte_val = qs[pair * 32 + j];
        int nibble = is_odd ? (byte_val >> 4) : (byte_val & 0x0F);

        float result = d * (float)sc * (float)nibble - dmin * (float)m;
        dst[(size_t)sb_idx * Q4_K_SUPER_BLOCK_SIZE + t] = to_half(result);
    }
}

// ── Q5_K ────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
dequant_q5_k_f16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q5_K_BLOCK_BYTES;
        float d = to_float(*reinterpret_cast<const __half*>(block));
        float dmin = to_float(*reinterpret_cast<const __half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;

        int sub = t / 32;
        int pos = t % 32;

        int sc, m;
        if (sub < 4)
        {
            sc = scales_raw[sub] & 0x3F;
            m = scales_raw[sub + 4] & 0x3F;
        }
        else
        {
            sc = (scales_raw[sub + 4] & 0x0F) | ((scales_raw[sub - 4] >> 6) << 4);
            m = (scales_raw[sub + 4] >> 4) | ((scales_raw[sub] >> 6) << 4);
        }

        float scale = d * (float)sc;
        float min_val = dmin * (float)m;

        int pair = sub / 2;
        int is_odd = sub % 2;
        uint8_t byte_val = qs[pair * 32 + pos];
        int nibble = is_odd ? (byte_val >> 4) : (byte_val & 0x0F);
        int bit = (qh[pos] >> sub) & 1;
        int val = nibble | (bit << 4);

        dst[(size_t)sb_idx * Q5_K_SUPER_BLOCK_SIZE + t] = to_half(scale * (float)val - min_val);
    }
}

// ── Q6_K ────────────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256)
dequant_q6_k_f16(
    const uint8_t* __restrict__ src,
    __half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q6_K_BLOCK_BYTES;
        const uint8_t* ql = block;
        const uint8_t* qh_base = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = to_float(*reinterpret_cast<const __half*>(block + 208));

        int half_idx = t / 128;
        int pos_in_half = t % 128;

        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh_base + half_idx * 32;
        const int8_t* sc_half = scales + half_idx * 8;

        int group = pos_in_half / 32;
        int l = pos_in_half % 32;
        int isc = l / 16;

        int q_val;
        switch (group)
        {
            case 0:
                q_val = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
                break;
            case 1:
                q_val = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
                break;
            case 2:
                q_val = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32;
                break;
            default:
                q_val = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32;
                break;
        }

        float sc = d * (float)sc_half[isc + group * 2];
        dst[(size_t)sb_idx * Q6_K_SUPER_BLOCK_SIZE + t] = to_half(sc * (float)q_val);
    }
}
