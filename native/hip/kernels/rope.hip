// rope.hip - Rotary Position Embedding (RoPE) kernel for DotLLM.Hip (H2.2).
// In-place rotation on Q and K with inline cos/sin calculation.
// Supports standard (norm) and NeoX styles.
#include "rdna_common.h"

extern "C" __global__ void __launch_bounds__(256)
rope_f16(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const int* __restrict__ positions,  // [seqLen]
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int rope_dim,
    const float theta,
    const int rope_type)  // 0 = standard, 1 = neox
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    // Process Q
    if (idx < total_q_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        int pos = positions[t];
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int q_stride = num_heads * head_dim;
        int base_idx = t * q_stride + head * head_dim;

        int i0, i1;
        if (rope_type == 1 || rope_type == 2) // neox: [pair, pair + half_rope]
        {
            i0 = base_idx + pair;
            i1 = base_idx + pair + half_rope;
        }
        else // standard: [2*pair, 2*pair + 1]
        {
            i0 = base_idx + 2 * pair;
            i1 = base_idx + 2 * pair + 1;
        }

        float v0 = to_float(q[i0]);
        float v1 = to_float(q[i1]);
        q[i0] = to_half(v0 * cos_val - v1 * sin_val);
        q[i1] = to_half(v0 * sin_val + v1 * cos_val);
    }

    // Process K
    if (idx < total_k_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        int t = remainder / num_kv_heads;

        int pos = positions[t];
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int k_stride = num_kv_heads * head_dim;
        int base_idx = t * k_stride + head * head_dim;

        int i0, i1;
        if (rope_type == 1 || rope_type == 2)
        {
            i0 = base_idx + pair;
            i1 = base_idx + pair + half_rope;
        }
        else
        {
            i0 = base_idx + 2 * pair;
            i1 = base_idx + 2 * pair + 1;
        }

        float v0 = to_float(k[i0]);
        float v1 = to_float(k[i1]);
        k[i0] = to_half(v0 * cos_val - v1 * sin_val);
        k[i1] = to_half(v0 * sin_val + v1 * cos_val);
    }
}

extern "C" __global__ void __launch_bounds__(256)
rope_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const int* __restrict__ positions,
    const int seq_len,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int rope_dim,
    const float theta,
    const int rope_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_rope = rope_dim / 2;
    int total_q_pairs = seq_len * num_heads * half_rope;
    int total_k_pairs = seq_len * num_kv_heads * half_rope;

    if (idx < total_q_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_heads;
        int t = remainder / num_heads;

        int pos = positions[t];
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int q_stride = num_heads * head_dim;
        int base_idx = t * q_stride + head * head_dim;

        int i0 = (rope_type == 1 || rope_type == 2) ? (base_idx + pair) : (base_idx + 2 * pair);
        int i1 = (rope_type == 1 || rope_type == 2) ? (base_idx + pair + half_rope) : (base_idx + 2 * pair + 1);

        float v0 = q[i0];
        float v1 = q[i1];
        q[i0] = v0 * cos_val - v1 * sin_val;
        q[i1] = v0 * sin_val + v1 * cos_val;
    }

    if (idx < total_k_pairs)
    {
        int pair = idx % half_rope;
        int remainder = idx / half_rope;
        int head = remainder % num_kv_heads;
        int t = remainder / num_kv_heads;

        int pos = positions[t];
        float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)rope_dim);
        float angle = (float)pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        int k_stride = num_kv_heads * head_dim;
        int base_idx = t * k_stride + head * head_dim;

        int i0 = (rope_type == 1 || rope_type == 2) ? (base_idx + pair) : (base_idx + 2 * pair);
        int i1 = (rope_type == 1 || rope_type == 2) ? (base_idx + pair + half_rope) : (base_idx + 2 * pair + 1);

        float v0 = k[i0];
        float v1 = k[i1];
        k[i0] = v0 * cos_val - v1 * sin_val;
        k[i1] = v0 * sin_val + v1 * cos_val;
    }
}
