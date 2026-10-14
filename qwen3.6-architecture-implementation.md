
---

# 🛠️ Agent Implementation Spec: Qwen 3.5 Architecture (`qwen35` / `qwen35moe`) in `dotLLM-Hip`

## 1. Context & Objective
Your objective is to implement end-to-end support for the **Qwen 3.5** architecture (`qwen35` dense and `qwen35moe` Mixture-of-Experts) in this repository (`dotLLM-Hip`). 
Before writing any code, inspect `CLAUDE.md`, `GEMINI.md`, `src/`, and `native/` to understand the exact project layout, how `DotLLM.Core`, `DotLLM.Models`, `DotLLM.Engine`, and the HIP backend (targeting `gfx1100` and `gfx1201`) interact. Follow the repository's Zero-GC and unmanaged memory principles (`NativeMemory.AlignedAlloc` / device pointers) on the hot path.

---

## 2. Architectural Overview of Qwen 3.5
Unlike Qwen 2.5/3.0, **Qwen 3.5** is a hybrid linear-attention + full-attention architecture (inherited from Qwen3-Next):
1. **3:1 Hybrid Layer Stack:** Controlled by `full_attention_interval` (default `4`). Layers where `(layer_idx + 1) % full_attention_interval != 0` (e.g., layers 0, 1, 2, 4, 5, 6...) are **Gated DeltaNet** (recurrent/SSM-like linear attention) layers. Every 4th layer (indices 3, 7, 11...) is a **Full Attention** layer.
2. **Gated DeltaNet (Linear Attention):**
   - Uses a depthwise 1D causal convolution (`kernel_size = 4`) + SiLU instead of RoPE.
   - Uses L2-normalization on $Q$ and $K$.
   - Maintains a fixed-size recurrent state matrix $S \in \mathbb{R}^{H_v \times d_k \times d_v}$ per layer and a 1D conv sliding window state of size `conv_kernel - 1` (or `conv_kernel`).
   - Applies a SiLU-gated RMSNorm (`RMSNorm(x) * SiLU(z)`) before the output projection.
3. **Sigmoid-Gated Full Attention:**
   - The $Q$ projection (`attn_q.weight`) outputs **2×** the normal dimension: `[num_heads * head_dim * 2, hidden_size]`.
   - The output is de-interleaved into $Q$ and a `gate`.
   - Applies per-head `attn_q_norm` and `attn_k_norm` (RMSNorm) before RoPE.
   - Uses partial Rotary Position Embeddings (e.g., `rope.dimension_count = 64` out of `head_dim = 256`, NeoX-style / M-RoPE sections).
   - Multiplies the SDPA output by `Sigmoid(gate)` before `attn_output.weight`.

---

## 3. Step-by-Step Implementation Plan

### Phase 1: GGUF Metadata & Config (`DotLLM.Core` & `DotLLM.Models`)
1. Register GGUF architecture strings: `"qwen35"`, `"qwen35moe"`, and `"qwen3next"`.
2. Extend `ModelConfig` (or equivalent config structure) to parse the following GGUF metadata keys (prefixed with `{arch}.`, e.g., `qwen35.`):
   - `full_attention_interval` (uint32, default `4`)
   - `rope.dimension_count` (uint32, e.g., `64`)
   - `rope.freq_base` (float, e.g., `1e7` or `1e6`)
   - `rope.dimension_sections` (int32[4], e.g., `[11, 11, 10, 0]`)
   - `ssm.conv_kernel` (uint32, default `4`)
   - `ssm.inner_size` (uint32, equals `num_v_heads * head_v_dim`, e.g., `4096` or `6144`)
   - `ssm.state_size` (uint32, equals `head_k_dim`, e.g., `128`)
   - `ssm.group_count` (uint32, equals `num_k_heads`, e.g., `16`)
   - `ssm.time_step_rank` (uint32, equals `num_v_heads`, e.g., `32` or `48`)
   - *(If implementing MoE)*: `expert_count` (256), `expert_used_count` (8), `expert_feed_forward_length`, `expert_shared_feed_forward_length`.

### Phase 2: Weight Loading & Tensor Mapping (`DotLLM.Models`)
Update the weight loader for `qwen35` blocks. Because layers alternate, inspect `(layer_idx + 1) % full_attention_interval == 0`:

**Common per-layer weights:**
- `blk.{L}.attn_norm.weight` `[hidden]`
- `blk.{L}.post_attention_norm.weight` `[hidden]`
- Dense FFN: `blk.{L}.ffn_gate.weight`, `blk.{L}.ffn_up.weight`, `blk.{L}.ffn_down.weight`
- MoE FFN (if `qwen35moe`): `ffn_gate_inp.weight`, `ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight`, plus shared expert weights.

**For Full Attention layers (`(L + 1) % interval == 0`):**
- `blk.{L}.attn_q.weight` `[num_q_heads * head_dim * 2, hidden]` *(Note the 2× size for Q + Gate!)*
- `blk.{L}.attn_k.weight` `[num_kv_heads * head_dim, hidden]`
- `blk.{L}.attn_v.weight` `[num_kv_heads * head_dim, hidden]`
- `blk.{L}.attn_q_norm.weight` `[head_dim]`
- `blk.{L}.attn_k_norm.weight` `[head_dim]`
- `blk.{L}.attn_output.weight` `[hidden, num_q_heads * head_dim]`

**For Gated DeltaNet layers (`(L + 1) % interval != 0`):**
- `blk.{L}.attn_qkv.weight` `[qk_dim * 2 + v_dim, hidden]` where `qk_dim = num_k_heads * state_size` and `v_dim = inner_size`.
- `blk.{L}.attn_gate.weight` `[inner_size, hidden]` (the $z$ gate projection)
- `blk.{L}.ssm_beta.weight` `[num_v_heads, hidden]`
- `blk.{L}.ssm_alpha.weight` `[num_v_heads, hidden]`
- `blk.{L}.ssm_conv1d.weight` `[conv_kernel, qk_dim * 2 + v_dim]` (F32)
- `blk.{L}.ssm_dt.bias` `[num_v_heads]` (F32)
- `blk.{L}.ssm_a` `[num_v_heads]` (F32, Note: in GGUF this is often stored as `-exp(A_log)` or `A_log`—verify against llama.cpp's `llm_build_qwen3next` / `qwen35` convention where `g = ssm_a * softplus(alpha + dt_bias)`).
- `blk.{L}.ssm_norm.weight` `[head_v_dim]` (F32, Gated RMSNorm weight)
- `blk.{L}.ssm_out.weight` `[hidden, inner_size]`

### Phase 3: Forward Graph & Mathematical Specification

#### A. Gated DeltaNet Layer Forward (`(L + 1) % interval != 0`)
Given input $x$ after `attn_norm`:
1. **Projections:**
   - `qkv = x @ attn_qkv.weight` $\rightarrow$ shape `[B, T, 2 * qk_dim + v_dim]`
   - `z = x @ attn_gate.weight` $\rightarrow$ shape `[B, T, v_dim]`
   - `beta_raw = x @ ssm_beta.weight` $\rightarrow$ shape `[B, T, H_v]`
   - `alpha_raw = x @ ssm_alpha.weight` $\rightarrow$ shape `[B, T, H_v]`
2. **Causal Conv1D + SiLU:**
   - Apply depthwise 1D causal convolution (`kernel = 4`) over `qkv` along the sequence dimension using `ssm_conv1d.weight` and the cached `conv_state` (last 3 tokens of `qkv`).
   - Apply $\text{SiLU}(x) = x \cdot \sigma(x)$ in-place.
   - Split `qkv` into $Q \in [H_k, d_k]$, $K \in [H_k, d_k]$, and $V \in [H_v, d_v]$.
3. **Head Expansion & Normalization:**
   - Apply L2 Normalization per head to $Q$ and $K$: $\tilde{q} = \frac{q}{\sqrt{\sum q_i^2 + \epsilon}}$, $\tilde{k} = \frac{k}{\sqrt{\sum k_i^2 + \epsilon}}$ ($\epsilon = 10^{-6}$).
   - Scale $\tilde{q} \leftarrow \tilde{q} \cdot \frac{1}{\sqrt{d_k}}$.
   - Repeat/interleave $Q$ and $K$ heads by `H_v / H_k` (e.g., $32 / 16 = 2$) to match $H_v$.
4. **Gates Computation:**
   - $\beta_t = \text{sigmoid}(\text{beta\_raw}_t)$
   - $g_t = \exp(\text{ssm\_a} \cdot \text{softplus}(\text{alpha\_raw}_t + \text{ssm\_dt.bias}))$ *(Note: if `ssm_a` in GGUF is already negative `-exp(A_log)`, multiply directly; if positive `A_log`, use `-exp(ssm_a)`)*.
5. **Delta Rule Recurrent Update (per head $h \in [0, H_v)$, state $S \in \mathbb{R}^{d_k \times d_v}$):**
   For each token step $t$:
   $$S \leftarrow g_t \cdot S$$
   $$\text{retrieved} = S^T \tilde{k}_t \quad (\text{or } \sum_{i=0}^{d_k-1} S_{i, j} \tilde{k}_{t, i})$$
   $$\delta = \beta_t \cdot (v_t - \text{retrieved})$$
   $$S \leftarrow S + \tilde{k}_t \otimes \delta$$
   $$o_t = S^T \tilde{q}_t$$
6. **Gated RMSNorm & Output:**
   - Per V-head ($d_v = 128$), apply RMSNorm using `ssm_norm.weight` and multiply by $\text{SiLU}(z_t)$:
     $$y_t = \text{RMSNorm}(o_t, \text{ssm\_norm.weight}) \odot \text{SiLU}(z_t)$$
   - Project back to residual stream: `out = y @ ssm_out.weight`.

#### B. Full Attention Layer Forward (`(L + 1) % interval == 0`)
Given input $x$ after `attn_norm`:
1. **Projections & De-interleaving:**
   - `qg = x @ attn_q.weight` $\rightarrow$ reshape to `[num_q_heads, 2, head_dim]` (or `[num_q_heads, head_dim * 2]` depending on GGUF layout—verify with llama.cpp `qwen35` Q+gate de-interleave) and split into `q` `[num_q_heads, head_dim]` and `gate` `[num_q_heads * head_dim]`.
   - `k = x @ attn_k.weight`, `v = x @ attn_v.weight`.
2. **QK-Norm & Partial RoPE:**
   - Apply RMSNorm (`attn_q_norm.weight` and `attn_k_norm.weight`) per head across `head_dim`.
   - Apply NeoX-style RoPE **only** to the first `rope.dimension_count` (e.g., 64) elements of each head; leave the remaining `head_dim - rope_dim` (e.g., 192) elements unchanged.
3. **Attention & Sigmoid Gate:**
   - Store $K, V$ into the layer's KV-cache and compute GQA Scaled Dot-Product Attention $\rightarrow$ `attn_out`.
   - Multiply elementwise: `attn_out = attn_out * Sigmoid(gate)`.
   - Project: `out = attn_out @ attn_output.weight`.

---

### Phase 4: Hybrid Cache & State Management (`DotLLM.Engine`)
1. **Avoid Wasting VRAM on KV Cache:** Only `1 / full_attention_interval` (25%) of layers use a KV cache. Update the KV-cache / `PagedAttention` allocator so non-attention layers allocate `0` KV-cache pages.
2. **Recurrent State Pool:** Create a state holder for the Gated DeltaNet layers:
   - `ConvState`: `[num_deltanet_layers, conv_dim, conv_kernel - 1]` (or `conv_kernel` ring buffer) in `float` or `fp16`.
   - `SsmState`: `[num_deltanet_layers, num_v_heads, head_k_dim, head_v_dim]` (e.g., `24 * 32 * 128 * 128` floats/fp16 $\approx 12\text{--}24\text{ MB}$ per sequence).
3. **Speculative Decoding / Prefix Caching Hook:** If speculative decoding or rollback is used in `DotLLM.Engine`, ensure `ConvState` and `SsmState` support snapshotting and rollback on token rejection (since recurrent state mutates in-place unlike a pointer-indexed KV cache).

---

### Phase 5: Radeon HIP Backend Kernels (`gfx1100` & `gfx1201`)
Inspect the existing HIP kernel compilation pipeline in `native/` and the `gfx1201_*.amdisa` / HIP sources. Both **RDNA 3 (`gfx1100`)** and **RDNA 4 (`gfx1201`)** execute in **wave32** (`warpSize == 32`) by default and have 64 KB of LDS per WGP/CU.

Implement and expose the following HIP kernels (plus CPU SIMD fallbacks in `DotLLM.Cpu` using `Vector256`/`Vector512` and `TensorPrimitives`):

1. **`gated_delta_net_recurrent_hip` (Critical Hot Path for Decode & Prefill):**
   - **Grid:** `(num_v_heads, batch_size)`
   - **Block:** `128` threads (4 wave32 wavefronts per block, since $d_k = 128, d_v = 128$).
   - **Mapping:** Each thread $tid \in [0, 128)$ owns one column $j = tid$ of the state matrix $S_{i, j}$ ($i \in [0, 128)$), keeping all 128 floats of its column in VGPRs (or fast LDS) across the token loop!
   - **Execution per token $t$:**
     - Load $\tilde{k}_t[0..127]$ and $\tilde{q}_t[0..127]$ into `__shared__` memory (LDS).
     - Thread $j$ computes $r_j = \sum_{i=0}^{127} (g_t \cdot S_{i,j}) \cdot \tilde{k}_t[i]$ directly in registers.
     - Thread $j$ computes $\delta_j = \beta_t \cdot (v_t[j] - r_j)$.
     - Thread $j$ updates its column $S_{i,j} \leftarrow g_t \cdot S_{i,j} + \tilde{k}_t[i] \cdot \delta_j$ and simultaneously accumulates $o_t[j] = \sum_{i=0}^{127} S_{i,j} \cdot \tilde{q}_t[i]$.
     - Write updated $S_{i,j}$ back to global VRAM after the sequence/token loop completes.
   - *Why this layout?* Zero cross-thread reductions are needed along $d_k$ if each thread owns a full $d_k$-column for a single $j \in d_v$! It avoids `__shfl_xor` overhead and achieves near-peak memory bandwidth on `gfx1100` and `gfx1201`.

2. **`causal_conv1d_silu_hip`:**
   - Fused depthwise 1D convolution (`K=4`) + state update + SiLU activation.
   - 1 thread per channel $c \in [0, \text{conv\_dim})$. Updates the 3-element state window and outputs `silu(sum)`.

3. **`l2_norm_and_gqa_repeat_hip`:**
   - Per-head L2 norm over `head_k_dim = 128` using wave32 shuffle reductions (`__shfl_xor(val, offset, 32)` + 4-warp shared memory reduction).

4. **`gated_rmsnorm_silu_hip`:**
   - Computes `RMSNorm(x, weight, eps) * SiLU(z)` across `head_v_dim = 128` per head.

5. **`deinterleave_q_gate_and_qknorm_rope_hip`:**
   - Splits `attn_q` output into `Q` and `Gate`.
   - Applies per-head RMSNorm to `Q` and `K`.
   - Applies partial RoPE (`rope_dim = 64`) to `Q` and `K`.
   - After attention, fuse or run `mul_sigmoid_gate_hip`: `attn_out[i] *= 1.0f / (1.0f + expf(-gate[i]))`.

---

## 4. Verification & Testing Checklist for the Agent
1. **Unit Tests (`tests/`):**
   - Add unit tests for `GatedDeltaNet` single-step recurrence and multi-token prefill comparing the C# CPU SIMD implementation against the HIP GPU kernel on `gfx1100` / `gfx1201`.
   - Add unit tests for Partial RoPE (`rope_dim < head_dim`) and `RMSNorm + SiLU` gating.
2. **ISA / Compilation Verification:**
   - Compile the new HIP kernels for `--offload-arch=gfx1100` and `--offload-arch=gfx1201`. Verify there are no VGPR spills in `gated_delta_net_recurrent_hip` (check `.amdisa` or `--save-temps` output if applicable).
3. **End-to-End Smoke Test:**
   - Pull a small Qwen 3.5 GGUF model (e.g., `Qwen3.5-0.8B-GGUF` or `unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-9B-GGUF` in `Q4_K_M` or `Q8_0`).
   - Run `dotllm run <model> -p "The capital of France is" -n 32` and verify coherent, non-garbled output on both CPU and HIP backends.

---

### 💡 Tips till dig innan du kör agenten:
* **Dense vs MoE först:** Be gärna agenten att börja med **Dense-varianten (`qwen35`, t.ex. Qwen3.5-0.8B, 4B eller 9B)** i en första PR/commit, och därefter lägga till `qwen35moe` (35B-A3B) om du även vill köra MoE-modellerna. Då blir det mycket enklare att felsöka Gated DeltaNet-kärnorna isolerat!
* **Testmodell:** `Qwen3.5-0.8B-GGUF` (eller `4B`) laddar på bråkdelen av en sekund och är perfekt för agenten att köra snabba röktester mot under utvecklingen.Här är en komplett och tekniskt detaljerad implementeringsguide (utformad som en instruktion/specifikation som du kan ge direkt till din kodningsagent, t.ex. Claude Code, Cursor eller liknande). 

Eftersom **Qwen 3.5** (och Qwen 3.6/Next-familjen) inte är en vanlig Transformer, utan en **hybridarkitektur (3:1 Gated DeltaNet + Sigmoid-Gated Full Attention)**, krävs förändringar i både GGUF-inläsning, state-hantering (KV-cache + rekurrent state) och nya HIP-kernels för `gfx1100` (RDNA 3) och `gfx1201` (RDNA 4).

Kopiera texten nedan och ge den till din agent!

---

# 🛠️ Agent Implementation Spec: Qwen 3.5 Architecture (`qwen35` / `qwen35moe`) in `dotLLM-Hip`

## 1. Context & Objective
Your objective is to implement end-to-end support for the **Qwen 3.5** architecture (`qwen35` dense and `qwen35moe` Mixture-of-Experts) in this repository (`dotLLM-Hip`). 
Before writing any code, inspect `CLAUDE.md`, `GEMINI.md`, `src/`, and `native/` to understand the exact project layout, how `DotLLM.Core`, `DotLLM.Models`, `DotLLM.Engine`, and the HIP backend (targeting `gfx1100` and `gfx1201`) interact. Follow the repository's Zero-GC and unmanaged memory principles (`NativeMemory.AlignedAlloc` / device pointers) on the hot path.

---

## 2. Architectural Overview of Qwen 3.5
Unlike Qwen 2.5/3.0, **Qwen 3.5** is a hybrid linear-attention + full-attention architecture (inherited from Qwen3-Next):
1. **3:1 Hybrid Layer Stack:** Controlled by `full_attention_interval` (default `4`). Layers where `(layer_idx + 1) % full_attention_interval != 0` (e.g., layers 0, 1, 2, 4, 5, 6...) are **Gated DeltaNet** (recurrent/SSM-like linear attention) layers. Every 4th layer (indices 3, 7, 11...) is a **Full Attention** layer.
2. **Gated DeltaNet (Linear Attention):**
   - Uses a depthwise 1D causal convolution (`kernel_size = 4`) + SiLU instead of RoPE.
   - Uses L2-normalization on $Q$ and $K$.
   - Maintains a fixed-size recurrent state matrix $S \in \mathbb{R}^{H_v \times d_k \times d_v}$ per layer and a 1D conv sliding window state of size `conv_kernel - 1` (or `conv_kernel`).
   - Applies a SiLU-gated RMSNorm (`RMSNorm(x) * SiLU(z)`) before the output projection.
3. **Sigmoid-Gated Full Attention:**
   - The $Q$ projection (`attn_q.weight`) outputs **2×** the normal dimension: `[num_heads * head_dim * 2, hidden_size]`.
   - The output is de-interleaved into $Q$ and a `gate`.
   - Applies per-head `attn_q_norm` and `attn_k_norm` (RMSNorm) before RoPE.
   - Uses partial Rotary Position Embeddings (e.g., `rope.dimension_count = 64` out of `head_dim = 256`, NeoX-style / M-RoPE sections).
   - Multiplies the SDPA output by `Sigmoid(gate)` before `attn_output.weight`.

---

## 3. Step-by-Step Implementation Plan

### Phase 1: GGUF Metadata & Config (`DotLLM.Core` & `DotLLM.Models`)
1. Register GGUF architecture strings: `"qwen35"`, `"qwen35moe"`, and `"qwen3next"`.
2. Extend `ModelConfig` (or equivalent config structure) to parse the following GGUF metadata keys (prefixed with `{arch}.`, e.g., `qwen35.`):
   - `full_attention_interval` (uint32, default `4`)
   - `rope.dimension_count` (uint32, e.g., `64`)
   - `rope.freq_base` (float, e.g., `1e7` or `1e6`)
   - `rope.dimension_sections` (int32[4], e.g., `[11, 11, 10, 0]`)
   - `ssm.conv_kernel` (uint32, default `4`)
   - `ssm.inner_size` (uint32, equals `num_v_heads * head_v_dim`, e.g., `4096` or `6144`)
   - `ssm.state_size` (uint32, equals `head_k_dim`, e.g., `128`)
   - `ssm.group_count` (uint32, equals `num_k_heads`, e.g., `16`)
   - `ssm.time_step_rank` (uint32, equals `num_v_heads`, e.g., `32` or `48`)
   - *(If implementing MoE)*: `expert_count` (256), `expert_used_count` (8), `expert_feed_forward_length`, `expert_shared_feed_forward_length`.

### Phase 2: Weight Loading & Tensor Mapping (`DotLLM.Models`)
Update the weight loader for `qwen35` blocks. Because layers alternate, inspect `(layer_idx + 1) % full_attention_interval == 0`:

**Common per-layer weights:**
- `blk.{L}.attn_norm.weight` `[hidden]`
- `blk.{L}.post_attention_norm.weight` `[hidden]`
- Dense FFN: `blk.{L}.ffn_gate.weight`, `blk.{L}.ffn_up.weight`, `blk.{L}.ffn_down.weight`
- MoE FFN (if `qwen35moe`): `ffn_gate_inp.weight`, `ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight`, plus shared expert weights.

**For Full Attention layers (`(L + 1) % interval == 0`):**
- `blk.{L}.attn_q.weight` `[num_q_heads * head_dim * 2, hidden]` *(Note the 2× size for Q + Gate!)*
- `blk.{L}.attn_k.weight` `[num_kv_heads * head_dim, hidden]`
- `blk.{L}.attn_v.weight` `[num_kv_heads * head_dim, hidden]`
- `blk.{L}.attn_q_norm.weight` `[head_dim]`
- `blk.{L}.attn_k_norm.weight` `[head_dim]`
- `blk.{L}.attn_output.weight` `[hidden, num_q_heads * head_dim]`

**For Gated DeltaNet layers (`(L + 1) % interval != 0`):**
- `blk.{L}.attn_qkv.weight` `[qk_dim * 2 + v_dim, hidden]` where `qk_dim = num_k_heads * state_size` and `v_dim = inner_size`.
- `blk.{L}.attn_gate.weight` `[inner_size, hidden]` (the $z$ gate projection)
- `blk.{L}.ssm_beta.weight` `[num_v_heads, hidden]`
- `blk.{L}.ssm_alpha.weight` `[num_v_heads, hidden]`
- `blk.{L}.ssm_conv1d.weight` `[conv_kernel, qk_dim * 2 + v_dim]` (F32)
- `blk.{L}.ssm_dt.bias` `[num_v_heads]` (F32)
- `blk.{L}.ssm_a` `[num_v_heads]` (F32, Note: in GGUF this is often stored as `-exp(A_log)` or `A_log`—verify against llama.cpp's `llm_build_qwen3next` / `qwen35` convention where `g = ssm_a * softplus(alpha + dt_bias)`).
- `blk.{L}.ssm_norm.weight` `[head_v_dim]` (F32, Gated RMSNorm weight)
- `blk.{L}.ssm_out.weight` `[hidden, inner_size]`

### Phase 3: Forward Graph & Mathematical Specification

#### A. Gated DeltaNet Layer Forward (`(L + 1) % interval != 0`)
Given input $x$ after `attn_norm`:
1. **Projections:**
   - `qkv = x @ attn_qkv.weight` $\rightarrow$ shape `[B, T, 2 * qk_dim + v_dim]`
   - `z = x @ attn_gate.weight` $\rightarrow$ shape `[B, T, v_dim]`
   - `beta_raw = x @ ssm_beta.weight` $\rightarrow$ shape `[B, T, H_v]`
   - `alpha_raw = x @ ssm_alpha.weight` $\rightarrow$ shape `[B, T, H_v]`
2. **Causal Conv1D + SiLU:**
   - Apply depthwise 1D causal convolution (`kernel = 4`) over `qkv` along the sequence dimension using `ssm_conv1d.weight` and the cached `conv_state` (last 3 tokens of `qkv`).
   - Apply $\text{SiLU}(x) = x \cdot \sigma(x)$ in-place.
   - Split `qkv` into $Q \in [H_k, d_k]$, $K \in [H_k, d_k]$, and $V \in [H_v, d_v]$.
3. **Head Expansion & Normalization:**
   - Apply L2 Normalization per head to $Q$ and $K$: $\tilde{q} = \frac{q}{\sqrt{\sum q_i^2 + \epsilon}}$, $\tilde{k} = \frac{k}{\sqrt{\sum k_i^2 + \epsilon}}$ ($\epsilon = 10^{-6}$).
   - Scale $\tilde{q} \leftarrow \tilde{q} \cdot \frac{1}{\sqrt{d_k}}$.
   - Repeat/interleave $Q$ and $K$ heads by `H_v / H_k` (e.g., $32 / 16 = 2$) to match $H_v$.
4. **Gates Computation:**
   - $\beta_t = \text{sigmoid}(\text{beta\_raw}_t)$
   - $g_t = \exp(\text{ssm\_a} \cdot \text{softplus}(\text{alpha\_raw}_t + \text{ssm\_dt.bias}))$ *(Note: if `ssm_a` in GGUF is already negative `-exp(A_log)`, multiply directly; if positive `A_log`, use `-exp(ssm_a)`)*.
5. **Delta Rule Recurrent Update (per head $h \in [0, H_v)$, state $S \in \mathbb{R}^{d_k \times d_v}$):**
   For each token step $t$:
   $$S \leftarrow g_t \cdot S$$
   $$\text{retrieved} = S^T \tilde{k}_t \quad (\text{or } \sum_{i=0}^{d_k-1} S_{i, j} \tilde{k}_{t, i})$$
   $$\delta = \beta_t \cdot (v_t - \text{retrieved})$$
   $$S \leftarrow S + \tilde{k}_t \otimes \delta$$
   $$o_t = S^T \tilde{q}_t$$
6. **Gated RMSNorm & Output:**
   - Per V-head ($d_v = 128$), apply RMSNorm using `ssm_norm.weight` and multiply by $\text{SiLU}(z_t)$:
     $$y_t = \text{RMSNorm}(o_t, \text{ssm\_norm.weight}) \odot \text{SiLU}(z_t)$$
   - Project back to residual stream: `out = y @ ssm_out.weight`.

#### B. Full Attention Layer Forward (`(L + 1) % interval == 0`)
Given input $x$ after `attn_norm`:
1. **Projections & De-interleaving:**
   - `qg = x @ attn_q.weight` $\rightarrow$ reshape to `[num_q_heads, 2, head_dim]` (or `[num_q_heads, head_dim * 2]` depending on GGUF layout—verify with llama.cpp `qwen35` Q+gate de-interleave) and split into `q` `[num_q_heads, head_dim]` and `gate` `[num_q_heads * head_dim]`.
   - `k = x @ attn_k.weight`, `v = x @ attn_v.weight`.
2. **QK-Norm & Partial RoPE:**
   - Apply RMSNorm (`attn_q_norm.weight` and `attn_k_norm.weight`) per head across `head_dim`.
   - Apply NeoX-style RoPE **only** to the first `rope.dimension_count` (e.g., 64) elements of each head; leave the remaining `head_dim - rope_dim` (e.g., 192) elements unchanged.
3. **Attention & Sigmoid Gate:**
   - Store $K, V$ into the layer's KV-cache and compute GQA Scaled Dot-Product Attention $\rightarrow$ `attn_out`.
   - Multiply elementwise: `attn_out = attn_out * Sigmoid(gate)`.
   - Project: `out = attn_out @ attn_output.weight`.

---

### Phase 4: Hybrid Cache & State Management (`DotLLM.Engine`)
1. **Avoid Wasting VRAM on KV Cache:** Only `1 / full_attention_interval` (25%) of layers use a KV cache. Update the KV-cache / `PagedAttention` allocator so non-attention layers allocate `0` KV-cache pages.
2. **Recurrent State Pool:** Create a state holder for the Gated DeltaNet layers:
   - `ConvState`: `[num_deltanet_layers, conv_dim, conv_kernel - 1]` (or `conv_kernel` ring buffer) in `float` or `fp16`.
   - `SsmState`: `[num_deltanet_layers, num_v_heads, head_k_dim, head_v_dim]` (e.g., `24 * 32 * 128 * 128` floats/fp16 $\approx 12\text{--}24\text{ MB}$ per sequence).
3. **Speculative Decoding / Prefix Caching Hook:** If speculative decoding or rollback is used in `DotLLM.Engine`, ensure `ConvState` and `SsmState` support snapshotting and rollback on token rejection (since recurrent state mutates in-place unlike a pointer-indexed KV cache).

---

### Phase 5: Radeon HIP Backend Kernels (`gfx1100` & `gfx1201`)
Inspect the existing HIP kernel compilation pipeline in `native/` and the `gfx1201_*.amdisa` / HIP sources. Both **RDNA 3 (`gfx1100`)** and **RDNA 4 (`gfx1201`)** execute in **wave32** (`warpSize == 32`) by default and have 64 KB of LDS per WGP/CU.

Implement and expose the following HIP kernels (plus CPU SIMD fallbacks in `DotLLM.Cpu` using `Vector256`/`Vector512` and `TensorPrimitives`):

1. **`gated_delta_net_recurrent_hip` (Critical Hot Path for Decode & Prefill):**
   - **Grid:** `(num_v_heads, batch_size)`
   - **Block:** `128` threads (4 wave32 wavefronts per block, since $d_k = 128, d_v = 128$).
   - **Mapping:** Each thread $tid \in [0, 128)$ owns one column $j = tid$ of the state matrix $S_{i, j}$ ($i \in [0, 128)$), keeping all 128 floats of its column in VGPRs (or fast LDS) across the token loop!
   - **Execution per token $t$:**
     - Load $\tilde{k}_t[0..127]$ and $\tilde{q}_t[0..127]$ into `__shared__` memory (LDS).
     - Thread $j$ computes $r_j = \sum_{i=0}^{127} (g_t \cdot S_{i,j}) \cdot \tilde{k}_t[i]$ directly in registers.
     - Thread $j$ computes $\delta_j = \beta_t \cdot (v_t[j] - r_j)$.
     - Thread $j$ updates its column $S_{i,j} \leftarrow g_t \cdot S_{i,j} + \tilde{k}_t[i] \cdot \delta_j$ and simultaneously accumulates $o_t[j] = \sum_{i=0}^{127} S_{i,j} \cdot \tilde{q}_t[i]$.
     - Write updated $S_{i,j}$ back to global VRAM after the sequence/token loop completes.
   - *Why this layout?* Zero cross-thread reductions are needed along $d_k$ if each thread owns a full $d_k$-column for a single $j \in d_v$! It avoids `__shfl_xor` overhead and achieves near-peak memory bandwidth on `gfx1100` and `gfx1201`.

2. **`causal_conv1d_silu_hip`:**
   - Fused depthwise 1D convolution (`K=4`) + state update + SiLU activation.
   - 1 thread per channel $c \in [0, \text{conv\_dim})$. Updates the 3-element state window and outputs `silu(sum)`.

3. **`l2_norm_and_gqa_repeat_hip`:**
   - Per-head L2 norm over `head_k_dim = 128` using wave32 shuffle reductions (`__shfl_xor(val, offset, 32)` + 4-warp shared memory reduction).

4. **`gated_rmsnorm_silu_hip`:**
   - Computes `RMSNorm(x, weight, eps) * SiLU(z)` across `head_v_dim = 128` per head.

5. **`deinterleave_q_gate_and_qknorm_rope_hip`:**
   - Splits `attn_q` output into `Q` and `Gate`.
   - Applies per-head RMSNorm to `Q` and `K`.
   - Applies partial RoPE (`rope_dim = 64`) to `Q` and `K`.
   - After attention, fuse or run `mul_sigmoid_gate_hip`: `attn_out[i] *= 1.0f / (1.0f + expf(-gate[i]))`.

---

## 4. Verification & Testing Checklist for the Agent
1. **Unit Tests (`tests/`):**
   - Add unit tests for `GatedDeltaNet` single-step recurrence and multi-token prefill comparing the C# CPU SIMD implementation against the HIP GPU kernel on `gfx1100` / `gfx1201`.
   - Add unit tests for Partial RoPE (`rope_dim < head_dim`) and `RMSNorm + SiLU` gating.
2. **ISA / Compilation Verification:**
   - Compile the new HIP kernels for `--offload-arch=gfx1100` and `--offload-arch=gfx1201`. Verify there are no VGPR spills in `gated_delta_net_recurrent_hip` (check `.amdisa` or `--save-temps` output if applicable).
3. **End-to-End Smoke Test:**
   - Pull a small Qwen 3.5 GGUF model (e.g., `Qwen3.5-0.8B-GGUF` or `unsloth/Qwen3.5-4B-GGUF` / `Qwen3.5-9B-GGUF` in `Q4_K_M` or `Q8_0`).
   - Run `dotllm run <model> -p "The capital of France is" -n 32` and verify coherent, non-garbled output on both CPU and HIP backends.

---
