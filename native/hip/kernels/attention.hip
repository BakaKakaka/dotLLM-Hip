// attention.hip - Baseline tiled attention kernel with online softmax for DotLLM.Hip (H2.4).
// Q[seqQ, numHeads * headDim], K[seqKv, numKvHeads * headDim], V same as K.
// FP16 input/output with FP32 accumulation.
// GQA: num_heads / num_kv_heads group sharing.
// Causal masking + sliding window.
// One block per (query_token, query_head) pair.
#include "rdna_common.h"

#define TILE_KV 256

extern "C" __global__ void __launch_bounds__(256)
attention_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window)
{
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;
    int pos_q = position_offset + tq;

    // Shared memory layout:
    //   q_shared[head_dim]
    //   score_tile[TILE_KV]
    //   out_accum[head_dim]
    //   warp_scratch[32]
    extern __shared__ float smem[];
    float* q_shared     = smem;
    float* score_tile   = smem + head_dim;
    float* out_accum    = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    // Step 1: Load Q vector into shared memory
    const __half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = to_float(q_vec[d]);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    // Step 2: Process KV in tiles with online softmax
    float running_max = -1e38f;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // 2a. Compute Q·K scores for this tile
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;

            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv >= sliding_window))
            {
                score_tile[t] = -1e38f;
                continue;
            }

            const __half* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * to_float(k_vec[d]);

            score_tile[t] = score * scale;
        }
        __syncthreads();

        // 2b. Find tile max
        float local_max = -1e38f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            local_max = device_max(local_max, score_tile[t]);

        float tile_max = block_all_max(local_max, warp_scratch);

        // 2c. Online softmax: rescale running accumulators
        float new_max = device_max(running_max, tile_max);
        float correction = (running_max > -1e37f) ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        // 2d. Compute attention weights: exp(score - max)
        float local_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -1e37f) ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            local_sum += w;
        }

        float tile_sum = block_all_sum(local_sum, warp_scratch);
        running_sum += tile_sum;

        // 2e. Accumulate weighted V
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    const __half* v_vec = v + (size_t)tkv * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * to_float(v_vec[d]);
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Step 3: Final normalize and write output
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    __half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = to_half(out_accum[d] * sum_inv);
}

extern "C" __global__ void __launch_bounds__(256)
attention_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window)
{
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;
    int pos_q = position_offset + tq;

    extern __shared__ float smem[];
    float* q_shared     = smem;
    float* score_tile   = smem + head_dim;
    float* out_accum    = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    const float* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = q_vec[d];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -1e38f;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;

            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv >= sliding_window))
            {
                score_tile[t] = -1e38f;
                continue;
            }

            const float* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * k_vec[d];

            score_tile[t] = score * scale;
        }
        __syncthreads();

        float local_max = -1e38f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            local_max = device_max(local_max, score_tile[t]);

        float tile_max = block_all_max(local_max, warp_scratch);

        float new_max = device_max(running_max, tile_max);
        float correction = (running_max > -1e37f) ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        float local_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -1e37f) ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            local_sum += w;
        }

        float tile_sum = block_all_sum(local_sum, warp_scratch);
        running_sum += tile_sum;

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    const float* v_vec = v + (size_t)tkv * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * v_vec[d];
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    float* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = out_accum[d] * sum_inv;
}

// ── Fast Coalesced Decode Attention (H3.4) ───────────────────────────
// Specialized for seq_q = 1. Eliminates uncoalesced strided loads by having
// waves cooperate across the head_dim vector (100% coalesced contiguous loads).
// Padded LDS layout (132 floats stride) eliminates bank conflicts.
extern "C" __global__ void __launch_bounds__(256)
attention_decode_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window)
{
    const int hq = blockIdx.x;
    if (hq >= num_heads) return;

    const int group_size = num_heads / num_kv_heads;
    const int hkv = hq / group_size;
    const float scale = rsqrtf((float)head_dim);

    const int kv_stride = num_kv_heads * head_dim;
    const int pos_q = position_offset;

    const int wave_id = threadIdx.x / WAVE; // 0..7
    const int lane = threadIdx.x % WAVE;    // 0..31

    const int elems_per_lane = head_dim / WAVE; // 4 for head_dim=128, 2 for head_dim=64

    // Load Q vector into registers (coalesced contiguous read across 32 lanes)
    float q_regs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const __half* q_vec = q + hq * head_dim + lane * elems_per_lane;
    for (int e = 0; e < elems_per_lane; e++)
    {
        q_regs[e] = to_float(q_vec[e]);
    }

    __shared__ float s_score[256];
    __shared__ float s_wave_acc[8][132]; // padded with 4 floats to eliminate LDS bank conflicts
    __shared__ float s_warp_scratch[32];

    float acc_head = 0.0f;
    float running_max = -1e38f;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int tile_len = seq_kv - t_start;
        if (tile_len > TILE_KV) tile_len = TILE_KV;

        // Phase 1: Compute Q·K scores for tokens in this tile (vectorized 64-bit K loads)
        for (int t = wave_id; t < tile_len; t += 8)
        {
            int tkv = t_start + t;
            if (tkv <= pos_q && (sliding_window <= 0 || pos_q - tkv < sliding_window))
            {
                const __half* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim + lane * elems_per_lane;
                float dot = 0.0f;
                if (elems_per_lane == 4)
                {
                    int2 k_raw = *reinterpret_cast<const int2*>(k_vec);
                    const __half* k_ptr = reinterpret_cast<const __half*>(&k_raw);
                    dot = q_regs[0] * to_float(k_ptr[0])
                        + q_regs[1] * to_float(k_ptr[1])
                        + q_regs[2] * to_float(k_ptr[2])
                        + q_regs[3] * to_float(k_ptr[3]);
                }
                else
                {
                    for (int e = 0; e < elems_per_lane; e++)
                        dot += q_regs[e] * to_float(k_vec[e]);
                }
                float sum_dot = wave_sum(dot);
                if (lane == 0)
                    s_score[t] = sum_dot * scale;
            }
            else
            {
                if (lane == 0)
                    s_score[t] = -1e38f;
            }
        }
        __syncthreads();

        // Phase 2: Online softmax update
        float local_max = -1e38f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            local_max = device_max(local_max, s_score[t]);
        float tile_max = block_all_max(local_max, s_warp_scratch);

        float new_max = device_max(running_max, tile_max);
        float correction = (running_max > -1e37f) ? expf(running_max - new_max) : 0.0f;
        running_sum *= correction;
        running_max = new_max;

        if (threadIdx.x < head_dim)
            acc_head *= correction;
        __syncthreads();

        // Convert scores to weights
        float local_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (s_score[t] > -1e37f) ? expf(s_score[t] - running_max) : 0.0f;
            s_score[t] = w;
            local_sum += w;
        }
        float tile_sum = block_all_sum(local_sum, s_warp_scratch);
        running_sum += tile_sum;
        __syncthreads();

        // Phase 3: Accumulate V for this tile (vectorized 64-bit V loads)
        float wave_v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int t = wave_id; t < tile_len; t += 8)
        {
            float w = s_score[t];
            if (w > 0.0f)
            {
                const __half* v_vec = v + (size_t)(t_start + t) * kv_stride + hkv * head_dim + lane * elems_per_lane;
                if (elems_per_lane == 4)
                {
                    int2 v_raw = *reinterpret_cast<const int2*>(v_vec);
                    const __half* v_ptr = reinterpret_cast<const __half*>(&v_raw);
                    wave_v[0] += w * to_float(v_ptr[0]);
                    wave_v[1] += w * to_float(v_ptr[1]);
                    wave_v[2] += w * to_float(v_ptr[2]);
                    wave_v[3] += w * to_float(v_ptr[3]);
                }
                else
                {
                    for (int e = 0; e < elems_per_lane; e++)
                        wave_v[e] += w * to_float(v_vec[e]);
                }
            }
        }

        for (int e = 0; e < elems_per_lane; e++)
            s_wave_acc[wave_id][lane * elems_per_lane + e] = wave_v[e];
        __syncthreads();

        // Parallel reduction of all head_dim elements across 8 waves (0 bank conflicts)
        if (threadIdx.x < head_dim)
        {
            int col = threadIdx.x;
            float sum_v = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; w++)
                sum_v += s_wave_acc[w][col];
            acc_head += sum_v;
        }
        __syncthreads();
    }

    // Phase 4: Final normalize and write output
    if (threadIdx.x < head_dim)
    {
        float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;
        output[hq * head_dim + threadIdx.x] = to_half(acc_head * sum_inv);
    }
}

