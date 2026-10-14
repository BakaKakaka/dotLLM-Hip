using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// CPU kernel implementations for Qwen 3.5 Gated DeltaNet (linear attention)
/// and sigmoid-gated full attention mechanisms.
/// </summary>
public static class GatedDeltaNet
{
    private const float Epsilon = 1e-6f;

    /// <summary>
    /// Computes depthwise 1D causal convolution (kernel_size=4) + in-place state update + SiLU activation for decode (1 token).
    /// </summary>
    /// <param name="input">Input channels of length <paramref name="channels"/>.</param>
    /// <param name="weights">Kernel weights [channels, 4] stored with kernel_size as inner dimension (length = channels * 4).</param>
    /// <param name="state">Sliding window state [channels, 3] storing the last 3 tokens per channel.</param>
    /// <param name="output">Output destination of length <paramref name="channels"/>.</param>
    /// <param name="channels">Number of channels (e.g. 8192).</param>
    [SkipLocalsInit]
    public static void CausalConv1dSiLu(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights,
        Span<float> state,
        Span<float> output,
        int channels)
    {
        for (int c = 0; c < channels; c++)
        {
            int wBase = c * 4;
            int sBase = c * 3;

            float s0 = state[sBase + 0];
            float s1 = state[sBase + 1];
            float s2 = state[sBase + 2];
            float x = input[c];

            float conv = weights[wBase + 0] * s0
                       + weights[wBase + 1] * s1
                       + weights[wBase + 2] * s2
                       + weights[wBase + 3] * x;

            // Update sliding window state
            state[sBase + 0] = s1;
            state[sBase + 1] = s2;
            state[sBase + 2] = x;

            // SiLU(conv) = conv * sigmoid(conv)
            float sigmoid = 1.0f / (1.0f + MathF.Exp(-conv));
            output[c] = conv * sigmoid;
        }
    }

    /// <summary>
    /// Applies per-head L2-normalization to Q and K, scales Q by 1/sqrt(headDim),
    /// and expands/repeats heads from <paramref name="numKHeads"/> to <paramref name="numVHeads"/>.
    /// </summary>
    [SkipLocalsInit]
    public static void L2NormAndGqaRepeat(
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        Span<float> qOut,
        Span<float> kOut,
        int numKHeads,
        int numVHeads,
        int headDim,
        float eps = Epsilon)
    {
        int repeatFactor = numVHeads / numKHeads;
        float qScaleFactor = 1.0f / MathF.Sqrt(headDim);

        for (int hk = 0; hk < numKHeads; hk++)
        {
            var qHead = q.Slice(hk * headDim, headDim);
            var kHead = k.Slice(hk * headDim, headDim);

            float qSumSq = TensorPrimitives.SumOfSquares(qHead);
            float kSumSq = TensorPrimitives.SumOfSquares(kHead);

            float qNorm = 1.0f / MathF.Sqrt(qSumSq + eps) * qScaleFactor;
            float kNorm = 1.0f / MathF.Sqrt(kSumSq + eps);

            for (int r = 0; r < repeatFactor; r++)
            {
                int hv = r * numKHeads + hk;
                var qDst = qOut.Slice(hv * headDim, headDim);
                var kDst = kOut.Slice(hv * headDim, headDim);

                TensorPrimitives.Multiply(qHead, qNorm, qDst);
                TensorPrimitives.Multiply(kHead, kNorm, kDst);
            }

        }
    }

    /// <summary>
    /// Computes recurrence gates beta and g for each head:
    /// <c>beta = sigmoid(beta_raw)</c> and <c>g = exp(ssm_a * softplus(alpha_raw + ssm_dt_bias))</c>.
    /// </summary>
    [SkipLocalsInit]
    public static void ComputeGates(
        ReadOnlySpan<float> betaRaw,
        ReadOnlySpan<float> alphaRaw,
        ReadOnlySpan<float> ssmA,
        ReadOnlySpan<float> ssmDtBias,
        Span<float> betaOut,
        Span<float> gOut,
        int numVHeads)
    {
        for (int h = 0; h < numVHeads; h++)
        {
            float b = betaRaw[h];
            betaOut[h] = 1.0f / (1.0f + MathF.Exp(-b));

            float u = alphaRaw[h] + ssmDtBias[h];
            float sp = u > 20.0f ? u : MathF.Log(1.0f + MathF.Exp(u));

            // In GGUF, ssm_a is already stored as negative values (-exp(A_log))
            gOut[h] = MathF.Exp(ssmA[h] * sp);
        }
    }

    /// <summary>
    /// Executes a single token step of the Gated DeltaNet recurrence rule for all heads.
    /// <c>S_h in R^{d_k x d_v}</c> state matrix is updated in-place:
    /// 1. S &lt;- g * S
    /// 2. r = S^T * k
    /// 3. delta = beta * (v - r)
    /// 4. S &lt;- S + k (x) delta
    /// 5. o = S^T * q
    /// </summary>
    /// <param name="q">Normalized and repeated Q vectors [numVHeads * headKDim].</param>
    /// <param name="k">Normalized and repeated K vectors [numVHeads * headKDim].</param>
    /// <param name="v">V vectors [numVHeads * headVDim].</param>
    /// <param name="beta">Beta gates [numVHeads].</param>
    /// <param name="g">Decay gates [numVHeads].</param>
    /// <param name="state">Recurrent state [numVHeads, headKDim, headVDim] updated in-place.</param>
    /// <param name="output">Output vectors [numVHeads * headVDim].</param>
    /// <param name="numVHeads">Number of V heads (e.g. 32).</param>
    /// <param name="headKDim">Head dimension of K and Q (e.g. 128).</param>
    /// <param name="headVDim">Head dimension of V (e.g. 128).</param>
    [SkipLocalsInit]
    public static unsafe void RecurrentDeltaRuleStep(
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> g,
        Span<float> state,
        Span<float> output,
        int numVHeads,
        int headKDim,
        int headVDim)
    {
        int stateHeadStride = headKDim * headVDim;

        fixed (float* pQ = q, pK = k, pV = v, pState = state, pOut = output)
        {
            for (int h = 0; h < numVHeads; h++)
            {
                float* qH = pQ + h * headKDim;
                float* kH = pK + h * headKDim;
                float* vH = pV + h * headVDim;
                float* sH = pState + h * stateHeadStride;
                float* oH = pOut + h * headVDim;

                float gh = g[h];
                float betah = beta[h];

                // Each column j in [0, headVDim-1] can be processed independently!
                for (int j = 0; j < headVDim; j++)
                {
                    // Compute retrieved r_j = sum_{i=0}^{headKDim-1} (g * S_{i, j}) * k_i
                    // and apply decay g to S_{i, j} simultaneously
                    float rj = 0.0f;
                    for (int i = 0; i < headKDim; i++)
                    {
                        int sIdx = i * headVDim + j;
                        float sij = sH[sIdx] * gh;
                        sH[sIdx] = sij;
                        rj += sij * kH[i];
                    }

                    // Delta: delta_j = beta * (v_j - r_j)
                    float deltaj = betah * (vH[j] - rj);

                    // Update S_{i, j} <- S_{i, j} + k_i * delta_j
                    // and accumulate output o_j = sum_{i=0}^{headKDim-1} S_{i, j} * q_i
                    float oj = 0.0f;
                    for (int i = 0; i < headKDim; i++)
                    {
                        int sIdx = i * headVDim + j;
                        float sij = sH[sIdx] + kH[i] * deltaj;
                        sH[sIdx] = sij;
                        oj += sij * qH[i];
                    }

                    oH[j] = oj;
                }
            }
        }
    }

    /// <summary>
    /// Applies Gated RMSNorm + SiLU across each V-head:
    /// <c>y_h = RMSNorm(o_h, ssm_norm_weight) * SiLU(z_h)</c>.
    /// </summary>
    [SkipLocalsInit]
    public static void GatedRmsNormSiLu(
        ReadOnlySpan<float> o,
        ReadOnlySpan<float> z,
        ReadOnlySpan<float> normWeight,
        Span<float> y,
        int numVHeads,
        int headVDim,
        float eps = Epsilon)
    {
        for (int h = 0; h < numVHeads; h++)
        {
            var oHead = o.Slice(h * headVDim, headVDim);
            var zHead = z.Slice(h * headVDim, headVDim);
            var yHead = y.Slice(h * headVDim, headVDim);

            float sumSq = TensorPrimitives.SumOfSquares(oHead);
            float rms = MathF.Sqrt(sumSq / headVDim + eps);
            float invRms = 1.0f / rms;

            for (int j = 0; j < headVDim; j++)
            {
                float normed = oHead[j] * invRms * normWeight[j];
                float zj = zHead[j];
                float silu = zj * (1.0f / (1.0f + MathF.Exp(-zj)));
                yHead[j] = normed * silu;
            }
        }
    }

    /// <summary>
    /// De-interleaves Q and Gate from the 2x dimension output of attn_q projection.
    /// Input shape is [numHeads, 2, headDim].
    /// First half of each head is Q, second half is Gate.
    /// </summary>
    [SkipLocalsInit]
    public static void DeinterleaveQGate(
        ReadOnlySpan<float> qg,
        Span<float> q,
        Span<float> gate,
        int numHeads,
        int headDim)
    {
        int stride = headDim * 2;
        for (int h = 0; h < numHeads; h++)
        {
            var src = qg.Slice(h * stride, stride);
            src.Slice(0, headDim).CopyTo(q.Slice(h * headDim, headDim));
            src.Slice(headDim, headDim).CopyTo(gate.Slice(h * headDim, headDim));
        }
    }

    /// <summary>
    /// Applies element-wise sigmoid gating to attention output:
    /// <c>attn_out[i] = attn_out[i] * sigmoid(gate[i])</c>.
    /// </summary>
    [SkipLocalsInit]
    public static void ApplySigmoidGate(
        Span<float> attnOut,
        ReadOnlySpan<float> gate)
    {
        for (int i = 0; i < attnOut.Length; i++)
        {
            float g = gate[i];
            float sig = 1.0f / (1.0f + MathF.Exp(-g));
            attnOut[i] *= sig;
        }
    }
}
