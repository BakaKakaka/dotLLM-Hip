// gated_delta_net.hip - Gated DeltaNet linear attention and full attention gating kernels
// Optimized for AMD RDNA3 (gfx1100) and RDNA4 (gfx1201) wave32 execution.
#include "rdna_common.h"

// ---------------------------------------------------------------------------
// 1. Causal Conv1D + SiLU (depthwise, kernel_size=4, decode step)
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256)
causal_conv1d_silu_f16(
    const __half* __restrict__ input,       // [channels]
    const float* __restrict__ weights,     // [channels, 4]
    float* __restrict__ state,             // [channels, 3]
    __half* __restrict__ output,           // [channels]
    const int channels)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    int w_base = c * 4;
    int s_base = c * 3;

    float s0 = state[s_base + 0];
    float s1 = state[s_base + 1];
    float s2 = state[s_base + 2];
    float x = to_float(input[c]);

    float conv = weights[w_base + 0] * s0
               + weights[w_base + 1] * s1
               + weights[w_base + 2] * s2
               + weights[w_base + 3] * x;

    // Shift state
    state[s_base + 0] = s1;
    state[s_base + 1] = s2;
    state[s_base + 2] = x;

    // SiLU(conv) = conv * sigmoid(conv)
    float sig = 1.0f / (1.0f + expf(-conv));
    output[c] = to_half(conv * sig);
}

// ---------------------------------------------------------------------------
// 2. L2 norm per head on Q and K, scale Q by 1/sqrt(headDim), and GQA repeat
// ---------------------------------------------------------------------------
// Grid: (num_k_heads, 1)
// Block: 128 threads (4 wave32 wavefronts)
extern "C" __global__ void __launch_bounds__(128)
l2_norm_and_gqa_repeat_f16(
    const __half* __restrict__ q_in,       // [num_k_heads, 128]
    const __half* __restrict__ k_in,       // [num_k_heads, 128]
    __half* __restrict__ q_out,            // [num_v_heads, 128]
    __half* __restrict__ k_out,            // [num_v_heads, 128]
    const int num_k_heads,
    const int num_v_heads,
    const int head_dim,                    // 128
    const float eps)
{
    int hk = blockIdx.x;
    if (hk >= num_k_heads) return;

    int tid = threadIdx.x;
    int wave_id = tid / WAVE;
    int lane_id = tid % WAVE;

    float qi = to_float(q_in[hk * head_dim + tid]);
    float ki = to_float(k_in[hk * head_dim + tid]);

    float q_sq = qi * qi;
    float k_sq = ki * ki;

    // Intra-wave reduction
    float q_wave_sum = wave_sum(q_sq);
    float k_wave_sum = wave_sum(k_sq);

    __shared__ float s_q_sum[4];
    __shared__ float s_k_sum[4];

    if (lane_id == 0)
    {
        s_q_sum[wave_id] = q_wave_sum;
        s_k_sum[wave_id] = k_wave_sum;
    }
    __syncthreads();

    // Block reduction on first wave
    __shared__ float s_q_scale;
    __shared__ float s_k_scale;

    if (tid == 0)
    {
        float total_q_sq = s_q_sum[0] + s_q_sum[1] + s_q_sum[2] + s_q_sum[3];
        float total_k_sq = s_k_sum[0] + s_k_sum[1] + s_k_sum[2] + s_k_sum[3];

        float q_scale_factor = 1.0f / sqrtf((float)head_dim);
        s_q_scale = rsqrtf(total_q_sq + eps) * q_scale_factor;
        s_k_scale = rsqrtf(total_k_sq + eps);
    }
    __syncthreads();

    float q_normed = qi * s_q_scale;
    float k_normed = ki * s_k_scale;

    int repeat = num_v_heads / num_k_heads;
    for (int r = 0; r < repeat; r++)
    {
        int hv = r * num_k_heads + hk;
        q_out[hv * head_dim + tid] = to_half(q_normed);
        k_out[hv * head_dim + tid] = to_half(k_normed);
    }
}


// ---------------------------------------------------------------------------
// 3. Recurrence Gates Computation
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(32)
compute_ssm_gates(
    const float* __restrict__ beta_raw,    // [num_v_heads]
    const float* __restrict__ alpha_raw,   // [num_v_heads]
    const float* __restrict__ ssm_a,       // [num_v_heads] (negative in GGUF)
    const float* __restrict__ ssm_dt_bias, // [num_v_heads]
    float* __restrict__ beta_out,          // [num_v_heads]
    float* __restrict__ g_out,             // [num_v_heads]
    const int num_v_heads)
{
    int h = threadIdx.x;
    if (h >= num_v_heads) return;

    float b = beta_raw[h];
    beta_out[h] = 1.0f / (1.0f + expf(-b));

    float u = alpha_raw[h] + ssm_dt_bias[h];
    float sp = u > 20.0f ? u : logf(1.0f + expf(u));

    // ssm_a is already negative in GGUF
    g_out[h] = expf(ssm_a[h] * sp);
}

// ---------------------------------------------------------------------------
// 3b. Recurrence Gates Computation (FP16 inputs from linear projection)
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(32)
compute_ssm_gates_f16(
    const __half* __restrict__ beta_raw,    // [num_v_heads]
    const __half* __restrict__ alpha_raw,   // [num_v_heads]
    const float* __restrict__ ssm_a,       // [num_v_heads] (negative in GGUF)
    const float* __restrict__ ssm_dt_bias, // [num_v_heads]
    float* __restrict__ beta_out,          // [num_v_heads]
    float* __restrict__ g_out,             // [num_v_heads]
    const int num_v_heads)
{
    int h = threadIdx.x;
    if (h >= num_v_heads) return;

    float b = to_float(beta_raw[h]);
    beta_out[h] = 1.0f / (1.0f + expf(-b));

    float u = to_float(alpha_raw[h]) + ssm_dt_bias[h];
    float sp = u > 20.0f ? u : logf(1.0f + expf(u));

    // ssm_a is already negative in GGUF
    g_out[h] = expf(ssm_a[h] * sp);
}

// ---------------------------------------------------------------------------
// 4. Recurrent Delta Rule (Single Token Step)
// ---------------------------------------------------------------------------
// Grid: (num_v_heads, 1)
// Block: 128 threads (4 wave32 wavefronts)
// Thread tid owns column j = tid of the state matrix S_{i, j} (i in [0, 127])
// keeping all 128 elements in VGPR registers.
extern "C" __global__ void __launch_bounds__(128)
gated_delta_net_recurrent_f16(
    const __half* __restrict__ q,       // [num_v_heads, 128]
    const __half* __restrict__ k,       // [num_v_heads, 128]
    const __half* __restrict__ v,       // [num_v_heads, 128]
    const float* __restrict__ beta,     // [num_v_heads]
    const float* __restrict__ g,        // [num_v_heads]
    float* __restrict__ state,          // [num_v_heads, 128, 128]
    __half* __restrict__ out,           // [num_v_heads, 128]
    const int num_v_heads,
    const int head_k_dim,               // 128
    const int head_v_dim)               // 128
{
    int h = blockIdx.x;
    if (h >= num_v_heads) return;

    int j = threadIdx.x; // column owned by this thread

    __shared__ float s_k[128];
    __shared__ float s_q[128];

    // Load k and q into LDS
    s_k[j] = to_float(k[h * 128 + j]);
    s_q[j] = to_float(q[h * 128 + j]);

    float gh = g[h];
    float betah = beta[h];
    float vj = to_float(v[h * 128 + j]);

    // Base pointer for this head's state matrix
    float* s_head = state + h * 16384; // 128 * 128 = 16384

    // Load thread j's entire column into VGPRs (128 floats)
    float col[128];
    #pragma unroll 4
    for (int i = 0; i < 128; i++)
    {
        col[i] = s_head[i * 128 + j];
    }

    __syncthreads();

    // Step 1: retrieved r_j = sum_{i=0}^{127} (gh * S_{i, j}) * k_i
    // and simultaneously decay col[i] in registers
    float rj = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < 128; i++)
    {
        float decayed = col[i] * gh;
        col[i] = decayed;
        rj += decayed * s_k[i];
    }

    // Step 2: delta_j = beta * (v_j - r_j)
    float deltaj = betah * (vj - rj);

    // Step 3: update S_{i, j} <- col[i] + k_i * deltaj
    // and accumulate output o_j = sum_{i=0}^{127} S_{i, j} * q_i
    float oj = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < 128; i++)
    {
        float updated = col[i] + s_k[i] * deltaj;
        col[i] = updated;
        oj += updated * s_q[i];
    }

    // Step 4: Write updated column back to global VRAM (coalesced!)
    #pragma unroll 4
    for (int i = 0; i < 128; i++)
    {
        s_head[i * 128 + j] = col[i];
    }

    // Step 5: Write output element
    out[h * 128 + j] = to_half(oj);
}

// ---------------------------------------------------------------------------
// 5. Gated RMSNorm + SiLU across head_v_dim (128)
// ---------------------------------------------------------------------------
// Grid: (num_v_heads, 1)
// Block: 128 threads (4 wave32 wavefronts)
extern "C" __global__ void __launch_bounds__(128)
gated_rmsnorm_silu_f16(
    const __half* __restrict__ o,          // [num_v_heads, 128]
    const __half* __restrict__ z,          // [num_v_heads, 128]
    const float* __restrict__ norm_weight, // [head_v_dim = 128]
    __half* __restrict__ out,              // [num_v_heads, 128]
    const int num_v_heads,
    const int head_v_dim,                  // 128
    const float eps)
{
    int h = blockIdx.x;
    if (h >= num_v_heads) return;

    int tid = threadIdx.x;
    int wave_id = tid / WAVE;
    int lane_id = tid % WAVE;

    float oi = to_float(o[h * head_v_dim + tid]);
    float o_sq = oi * oi;

    float wave_sum_sq = wave_sum(o_sq);

    __shared__ float s_sum[4];
    if (lane_id == 0)
    {
        s_sum[wave_id] = wave_sum_sq;
    }
    __syncthreads();

    __shared__ float s_inv_rms;
    if (tid == 0)
    {
        float total_sum_sq = s_sum[0] + s_sum[1] + s_sum[2] + s_sum[3];
        s_inv_rms = rsqrtf(total_sum_sq / (float)head_v_dim + eps);
    }
    __syncthreads();

    float zi = to_float(z[h * head_v_dim + tid]);
    float silu_z = zi * (1.0f / (1.0f + expf(-zi)));
    float w = norm_weight[tid];

    float y = (oi * s_inv_rms) * w * silu_z;
    out[h * head_v_dim + tid] = to_half(y);
}

// ---------------------------------------------------------------------------
// 6. Full Attention De-interleave Q and Gate
// ---------------------------------------------------------------------------
// Grid: (num_heads, 1)
// Block: 256 threads (since head_dim = 256)
extern "C" __global__ void __launch_bounds__(256)
deinterleave_q_gate_f16(
    const __half* __restrict__ qg,         // [num_heads, head_dim * 2]
    __half* __restrict__ q,                // [num_heads, head_dim]
    __half* __restrict__ gate,             // [num_heads, head_dim]
    const int num_heads,
    const int head_dim)
{
    int h = blockIdx.x;
    if (h >= num_heads) return;

    int j = threadIdx.x;
    if (j >= head_dim) return;

    int stride = head_dim * 2;
    int in_offset = h * stride + j;

    q[h * head_dim + j] = qg[in_offset];
    gate[h * head_dim + j] = qg[in_offset + head_dim];
}

// ---------------------------------------------------------------------------
// 7. Mul Sigmoid Gate for Attention Output
// ---------------------------------------------------------------------------
// Grid: ((total_elements + 255) / 256, 1)
// Block: 256 threads
extern "C" __global__ void __launch_bounds__(256)
mul_sigmoid_gate_f16(
    __half* __restrict__ attn_out,         // [total_elements] in-place
    const __half* __restrict__ gate,       // [total_elements]
    const int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float v = to_float(attn_out[idx]);
    float g = to_float(gate[idx]);
    float sig = 1.0f / (1.0f + expf(-g));
    attn_out[idx] = to_half(v * sig);
}
