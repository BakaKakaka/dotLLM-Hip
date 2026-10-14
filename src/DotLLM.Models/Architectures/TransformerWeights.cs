using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Holds per-layer weight references for a single transformer layer.
/// Norm weights are dequantized to <c>float[]</c> at load time (small).
/// Linear projection weights remain as mmap pointers with their quantization type.
/// Bias arrays are nullable — null when the model has no biases (e.g. standard Llama/Mistral).
/// </summary>
internal readonly struct TransformerLayerWeights
{
    /// <summary>True if this layer is a Gated DeltaNet layer (Qwen 3.5 hybrid architecture).</summary>
    public readonly bool IsDeltaNet;
    /// <summary>Index into the recurrent DeltaNet state pool (0..numDeltaNetLayers-1). -1 for full attention layers.</summary>
    public readonly int DeltaNetIndex;

    /// <summary>Pre-attention RMSNorm weight [hiddenSize].</summary>
    public readonly float[] AttnNormWeight;

    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to Q after projection, before RoPE. Null when absent (e.g. Qwen2, Llama).</summary>
    public readonly float[]? QNormWeight;
    /// <summary>Optional QK-norm weight [headDim]. Applied per-head to K after projection, before RoPE. Null when absent.</summary>
    public readonly float[]? KNormWeight;

    /// <summary>Q projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint QWeight;
    public readonly QuantizationType QQuantType;
    public readonly int QOutputDim;
    public readonly int QInputDim;
    /// <summary>Optional Q projection bias [QOutputDim]. Null when absent.</summary>
    public readonly float[]? QBias;

    /// <summary>K projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint KWeight;
    public readonly QuantizationType KQuantType;
    public readonly int KOutputDim;
    public readonly int KInputDim;
    /// <summary>Optional K projection bias [KOutputDim]. Null when absent.</summary>
    public readonly float[]? KBias;

    /// <summary>V projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint VWeight;
    public readonly QuantizationType VQuantType;
    public readonly int VOutputDim;
    public readonly int VInputDim;
    /// <summary>Optional V projection bias [VOutputDim]. Null when absent.</summary>
    public readonly float[]? VBias;

    /// <summary>Output projection pointer, quantType, output dim, input dim.</summary>
    public readonly nint OWeight;
    public readonly QuantizationType OQuantType;
    public readonly int OOutputDim;
    public readonly int OInputDim;
    /// <summary>Optional output projection bias [OOutputDim]. Null when absent.</summary>
    public readonly float[]? OBias;

    // Gated DeltaNet linear attention weights
    public readonly nint QkvWeight;
    public readonly QuantizationType QkvQuantType;
    public readonly int QkvOutputDim;
    public readonly int QkvInputDim;

    public readonly nint SsmGateWeight;
    public readonly QuantizationType SsmGateQuantType;
    public readonly int SsmGateOutputDim;
    public readonly int SsmGateInputDim;

    public readonly nint SsmBetaWeight;
    public readonly QuantizationType SsmBetaQuantType;
    public readonly int SsmBetaOutputDim;
    public readonly int SsmBetaInputDim;

    public readonly nint SsmAlphaWeight;
    public readonly QuantizationType SsmAlphaQuantType;
    public readonly int SsmAlphaOutputDim;
    public readonly int SsmAlphaInputDim;

    public readonly nint SsmConv1dWeight;
    public readonly int SsmConv1dChannels;
    public readonly int SsmConv1dKernel;

    public readonly float[]? SsmDtBias;
    public readonly float[]? SsmA;
    public readonly float[]? SsmNormWeight;

    public readonly nint SsmOutWeight;
    public readonly QuantizationType SsmOutQuantType;
    public readonly int SsmOutOutputDim;
    public readonly int SsmOutInputDim;

    /// <summary>Pre-FFN RMSNorm weight [hiddenSize].</summary>
    public readonly float[] FfnNormWeight;

    /// <summary>SwiGLU gate projection.</summary>
    public readonly nint GateWeight;
    public readonly QuantizationType GateQuantType;
    public readonly int GateOutputDim;
    public readonly int GateInputDim;
    /// <summary>Optional gate projection bias [GateOutputDim]. Null when absent.</summary>
    public readonly float[]? GateBias;

    /// <summary>SwiGLU up projection.</summary>
    public readonly nint UpWeight;
    public readonly QuantizationType UpQuantType;
    public readonly int UpOutputDim;
    public readonly int UpInputDim;
    /// <summary>Optional up projection bias [UpOutputDim]. Null when absent.</summary>
    public readonly float[]? UpBias;

    /// <summary>Down projection.</summary>
    public readonly nint DownWeight;
    public readonly QuantizationType DownQuantType;
    public readonly int DownOutputDim;
    public readonly int DownInputDim;
    /// <summary>Optional down projection bias [DownOutputDim]. Null when absent.</summary>
    public readonly float[]? DownBias;

    public TransformerLayerWeights(
        float[] attnNormWeight,
        nint qWeight, QuantizationType qQuantType, int qOutputDim, int qInputDim,
        nint kWeight, QuantizationType kQuantType, int kOutputDim, int kInputDim,
        nint vWeight, QuantizationType vQuantType, int vOutputDim, int vInputDim,
        nint oWeight, QuantizationType oQuantType, int oOutputDim, int oInputDim,
        float[] ffnNormWeight,
        nint gateWeight, QuantizationType gateQuantType, int gateOutputDim, int gateInputDim,
        nint upWeight, QuantizationType upQuantType, int upOutputDim, int upInputDim,
        nint downWeight, QuantizationType downQuantType, int downOutputDim, int downInputDim,
        float[]? qBias = null, float[]? kBias = null, float[]? vBias = null, float[]? oBias = null,
        float[]? gateBias = null, float[]? upBias = null, float[]? downBias = null,
        float[]? qNormWeight = null, float[]? kNormWeight = null)
    {
        IsDeltaNet = false;
        DeltaNetIndex = -1;
        AttnNormWeight = attnNormWeight;
        QNormWeight = qNormWeight;
        KNormWeight = kNormWeight;
        QWeight = qWeight; QQuantType = qQuantType; QOutputDim = qOutputDim; QInputDim = qInputDim; QBias = qBias;
        KWeight = kWeight; KQuantType = kQuantType; KOutputDim = kOutputDim; KInputDim = kInputDim; KBias = kBias;
        VWeight = vWeight; VQuantType = vQuantType; VOutputDim = vOutputDim; VInputDim = vInputDim; VBias = vBias;
        OWeight = oWeight; OQuantType = oQuantType; OOutputDim = oOutputDim; OInputDim = oInputDim; OBias = oBias;
        FfnNormWeight = ffnNormWeight;
        GateWeight = gateWeight; GateQuantType = gateQuantType; GateOutputDim = gateOutputDim; GateInputDim = gateInputDim; GateBias = gateBias;
        UpWeight = upWeight; UpQuantType = upQuantType; UpOutputDim = upOutputDim; UpInputDim = upInputDim; UpBias = upBias;
        DownWeight = downWeight; DownQuantType = downQuantType; DownOutputDim = downOutputDim; DownInputDim = downInputDim; DownBias = downBias;

        QkvWeight = 0; QkvQuantType = default; QkvOutputDim = 0; QkvInputDim = 0;
        SsmGateWeight = 0; SsmGateQuantType = default; SsmGateOutputDim = 0; SsmGateInputDim = 0;
        SsmBetaWeight = 0; SsmBetaQuantType = default; SsmBetaOutputDim = 0; SsmBetaInputDim = 0;
        SsmAlphaWeight = 0; SsmAlphaQuantType = default; SsmAlphaOutputDim = 0; SsmAlphaInputDim = 0;
        SsmConv1dWeight = 0; SsmConv1dChannels = 0; SsmConv1dKernel = 0;
        SsmDtBias = null; SsmA = null; SsmNormWeight = null;
        SsmOutWeight = 0; SsmOutQuantType = default; SsmOutOutputDim = 0; SsmOutInputDim = 0;
    }

    private TransformerLayerWeights(
        int deltaNetIndex,
        float[] attnNormWeight,
        float[] ffnNormWeight,
        nint gateWeight, QuantizationType gateQuantType, int gateOutputDim, int gateInputDim,
        nint upWeight, QuantizationType upQuantType, int upOutputDim, int upInputDim,
        nint downWeight, QuantizationType downQuantType, int downOutputDim, int downInputDim,
        nint qkvWeight, QuantizationType qkvQuantType, int qkvOutputDim, int qkvInputDim,
        nint ssmGateWeight, QuantizationType ssmGateQuantType, int ssmGateOutputDim, int ssmGateInputDim,
        nint ssmBetaWeight, QuantizationType ssmBetaQuantType, int ssmBetaOutputDim, int ssmBetaInputDim,
        nint ssmAlphaWeight, QuantizationType ssmAlphaQuantType, int ssmAlphaOutputDim, int ssmAlphaInputDim,
        nint ssmConv1dWeight, int ssmConv1dChannels, int ssmConv1dKernel,
        float[] ssmDtBias, float[] ssmA, float[] ssmNormWeight,
        nint ssmOutWeight, QuantizationType ssmOutQuantType, int ssmOutOutputDim, int ssmOutInputDim,
        float[]? gateBias, float[]? upBias, float[]? downBias)
    {
        IsDeltaNet = true;
        DeltaNetIndex = deltaNetIndex;
        AttnNormWeight = attnNormWeight;
        FfnNormWeight = ffnNormWeight;
        GateWeight = gateWeight; GateQuantType = gateQuantType; GateOutputDim = gateOutputDim; GateInputDim = gateInputDim; GateBias = gateBias;
        UpWeight = upWeight; UpQuantType = upQuantType; UpOutputDim = upOutputDim; UpInputDim = upInputDim; UpBias = upBias;
        DownWeight = downWeight; DownQuantType = downQuantType; DownOutputDim = downOutputDim; DownInputDim = downInputDim; DownBias = downBias;

        QkvWeight = qkvWeight; QkvQuantType = qkvQuantType; QkvOutputDim = qkvOutputDim; QkvInputDim = qkvInputDim;
        SsmGateWeight = ssmGateWeight; SsmGateQuantType = ssmGateQuantType; SsmGateOutputDim = ssmGateOutputDim; SsmGateInputDim = ssmGateInputDim;
        SsmBetaWeight = ssmBetaWeight; SsmBetaQuantType = ssmBetaQuantType; SsmBetaOutputDim = ssmBetaOutputDim; SsmBetaInputDim = ssmBetaInputDim;
        SsmAlphaWeight = ssmAlphaWeight; SsmAlphaQuantType = ssmAlphaQuantType; SsmAlphaOutputDim = ssmAlphaOutputDim; SsmAlphaInputDim = ssmAlphaInputDim;
        SsmConv1dWeight = ssmConv1dWeight; SsmConv1dChannels = ssmConv1dChannels; SsmConv1dKernel = ssmConv1dKernel;
        SsmDtBias = ssmDtBias; SsmA = ssmA; SsmNormWeight = ssmNormWeight;
        SsmOutWeight = ssmOutWeight; SsmOutQuantType = ssmOutQuantType; SsmOutOutputDim = ssmOutOutputDim; SsmOutInputDim = ssmOutInputDim;

        QWeight = 0; QQuantType = default; QOutputDim = 0; QInputDim = 0; QBias = null;
        KWeight = 0; KQuantType = default; KOutputDim = 0; KInputDim = 0; KBias = null;
        VWeight = 0; VQuantType = default; VOutputDim = 0; VInputDim = 0; VBias = null;
        OWeight = 0; OQuantType = default; OOutputDim = 0; OInputDim = 0; OBias = null;
        QNormWeight = null; KNormWeight = null;
    }

    public static TransformerLayerWeights CreateDeltaNet(
        int deltaNetIndex,
        float[] attnNormWeight,
        float[] ffnNormWeight,
        nint gateWeight, QuantizationType gateQuantType, int gateOutputDim, int gateInputDim,
        nint upWeight, QuantizationType upQuantType, int upOutputDim, int upInputDim,
        nint downWeight, QuantizationType downQuantType, int downOutputDim, int downInputDim,
        nint qkvWeight, QuantizationType qkvQuantType, int qkvOutputDim, int qkvInputDim,
        nint ssmGateWeight, QuantizationType ssmGateQuantType, int ssmGateOutputDim, int ssmGateInputDim,
        nint ssmBetaWeight, QuantizationType ssmBetaQuantType, int ssmBetaOutputDim, int ssmBetaInputDim,
        nint ssmAlphaWeight, QuantizationType ssmAlphaQuantType, int ssmAlphaOutputDim, int ssmAlphaInputDim,
        nint ssmConv1dWeight, int ssmConv1dChannels, int ssmConv1dKernel,
        float[] ssmDtBias, float[] ssmA, float[] ssmNormWeight,
        nint ssmOutWeight, QuantizationType ssmOutQuantType, int ssmOutOutputDim, int ssmOutInputDim,
        float[]? gateBias = null, float[]? upBias = null, float[]? downBias = null)
    {
        return new TransformerLayerWeights(
            deltaNetIndex,
            attnNormWeight, ffnNormWeight,
            gateWeight, gateQuantType, gateOutputDim, gateInputDim,
            upWeight, upQuantType, upOutputDim, upInputDim,
            downWeight, downQuantType, downOutputDim, downInputDim,
            qkvWeight, qkvQuantType, qkvOutputDim, qkvInputDim,
            ssmGateWeight, ssmGateQuantType, ssmGateOutputDim, ssmGateInputDim,
            ssmBetaWeight, ssmBetaQuantType, ssmBetaOutputDim, ssmBetaInputDim,
            ssmAlphaWeight, ssmAlphaQuantType, ssmAlphaOutputDim, ssmAlphaInputDim,
            ssmConv1dWeight, ssmConv1dChannels, ssmConv1dKernel,
            ssmDtBias, ssmA, ssmNormWeight,
            ssmOutWeight, ssmOutQuantType, ssmOutOutputDim, ssmOutInputDim,
            gateBias, upBias, downBias);
    }
}

/// <summary>
/// Holds R4-interleaved weight buffers for all projections in a single transformer layer.
/// Disposed when the parent <see cref="TransformerWeights"/> is disposed.
/// </summary>
internal sealed class RepackedLayerWeights : IDisposable
{
    public WeightRepacking.RepackedWeight Q, K, V, O, Gate, Up, Down;

    /// <summary>Total unmanaged bytes held by this layer's R4 buffers. Unrepacked projections contribute 0.</summary>
    public long AllocatedBytes =>
        Q.AllocatedBytes + K.AllocatedBytes + V.AllocatedBytes + O.AllocatedBytes
        + Gate.AllocatedBytes + Up.AllocatedBytes + Down.AllocatedBytes;

    public void Dispose()
    {
        Q.Dispose(); K.Dispose(); V.Dispose(); O.Dispose();
        Gate.Dispose(); Up.Dispose(); Down.Dispose();
    }
}

/// <summary>
/// Organizes all weight tensor references from a loaded GGUF file for a transformer-family model.
/// Norm weights are dequantized to managed <c>float[]</c> at load time.
/// Linear projections remain as raw mmap pointers for zero-copy inference.
/// Optionally holds R4-interleaved weight buffers for improved cache locality in 4-row SIMD kernels.
/// </summary>
internal sealed class TransformerWeights : IDisposable
{
    /// <summary>Token embedding pointer and metadata.</summary>
    public nint TokenEmbedWeight { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public int VocabSize { get; }
    public int HiddenSize { get; }

    /// <summary>Per-layer weights.</summary>
    public TransformerLayerWeights[] Layers { get; }

    /// <summary>Final RMSNorm weight [hiddenSize].</summary>
    public float[] OutputNormWeight { get; }

    /// <summary>LM head (output projection) pointer and metadata.</summary>
    public nint OutputWeight { get; }
    public QuantizationType OutputQuantType { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }

    /// <summary>Per-layer R4-interleaved weights. Null until <see cref="RepackWeights"/> is called.</summary>
    public RepackedLayerWeights[]? RepackedLayers { get; private set; }

    /// <summary>R4-interleaved LM head weights. Null until <see cref="RepackWeights"/> is called or if type is not repackable.</summary>
    public WeightRepacking.RepackedWeight? RepackedOutput { get; private set; }

    /// <summary>
    /// Total unmanaged bytes held by R4-interleaved buffers. Zero until <see cref="RepackWeights"/> runs,
    /// and zero for models whose weights are not repackable (F32/F16).
    /// </summary>
    /// <remarks>
    /// These buffers are a second copy of the weights, held in committed memory alongside the
    /// memory-mapped originals — reporting that counts only the mapped file understates the
    /// process footprint by roughly 2x.
    /// </remarks>
    public long RepackedBytes { get; private set; }

    private TransformerWeights(
        nint tokenEmbedWeight, QuantizationType tokenEmbedQuantType, int vocabSize, int hiddenSize,
        TransformerLayerWeights[] layers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQuantType, int outputOutputDim, int outputInputDim)
    {
        TokenEmbedWeight = tokenEmbedWeight;
        TokenEmbedQuantType = tokenEmbedQuantType;
        VocabSize = vocabSize;
        HiddenSize = hiddenSize;
        Layers = layers;
        OutputNormWeight = outputNormWeight;
        OutputWeight = outputWeight;
        OutputQuantType = outputQuantType;
        OutputOutputDim = outputOutputDim;
        OutputInputDim = outputInputDim;
    }

    /// <summary>
    /// Loads all weight references from an opened GGUF file.
    /// Norm weights are dequantized to <c>float[]</c>. Linear projections stay as mmap pointers.
    /// </summary>
    public static TransformerWeights LoadFromGguf(GgufFile gguf, ModelConfig config)
    {
        nint dataBase = gguf.DataBasePointer;
        var tensors = gguf.TensorsByName;

        // Token embeddings
        var embDesc = tensors["token_embd.weight"];
        nint embPtr = dataBase + (nint)embDesc.DataOffset;

        // Per-layer weights
        var layers = new TransformerLayerWeights[config.NumLayers];
        int deltaNetCounter = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            layers[i] = LoadLayer(i, dataBase, tensors, config, ref deltaNetCounter);
        }

        // Output norm
        var outNormDesc = tensors["output_norm.weight"];
        float[] outputNormWeight = DequantizeNorm(dataBase, outNormDesc, config.HiddenSize);

        // LM head — may be tied to token embeddings
        nint outputPtr;
        QuantizationType outputQt;
        int outputM, outputK;

        if (tensors.TryGetValue("output.weight", out var outDesc))
        {
            outputPtr = dataBase + (nint)outDesc.DataOffset;
            outputQt = outDesc.QuantizationType;
            // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
            outputK = outDesc.Shape[0];
            outputM = outDesc.Shape[1];
        }
        else
        {
            // Tied embeddings: alias token_embd.weight
            outputPtr = embPtr;
            outputQt = embDesc.QuantizationType;
            outputK = embDesc.Shape[0];
            outputM = embDesc.Shape[1];
        }

        return new TransformerWeights(
            embPtr, embDesc.QuantizationType, config.VocabSize, config.HiddenSize,
            layers,
            outputNormWeight,
            outputPtr, outputQt, outputM, outputK);
    }

    /// <summary>
    /// Repacks all linear projection weights into R4 interleaved layout for improved
    /// cache locality in 4-row SIMD kernels. Skips token embeddings (random row access)
    /// and non-block-structured types (F32, F16).
    /// </summary>
    public void RepackWeights()
    {
        var repacked = new RepackedLayerWeights[Layers.Length];
        for (int i = 0; i < Layers.Length; i++)
        {
            ref readonly var lw = ref Layers[i];
            if (lw.IsDeltaNet)
            {
                repacked[i] = new RepackedLayerWeights
                {
                    Gate = TryRepack(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim),
                    Up = TryRepack(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim),
                    Down = TryRepack(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim),
                };
            }
            else
            {
                repacked[i] = new RepackedLayerWeights
                {
                    Q = TryRepack(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim),
                    K = TryRepack(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim),
                    V = TryRepack(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim),
                    O = TryRepack(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim),
                    Gate = TryRepack(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim),
                    Up = TryRepack(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim),
                    Down = TryRepack(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim),
                };
            }
        }
        RepackedLayers = repacked;

        if (WeightRepacking.IsRepackable(OutputQuantType))
            RepackedOutput = WeightRepacking.RepackR4(OutputWeight, OutputQuantType, OutputOutputDim, OutputInputDim);

        long total = 0;
        foreach (var rl in repacked)
            total += rl.AllocatedBytes;
        total += RepackedOutput?.AllocatedBytes ?? 0;
        RepackedBytes = total;
    }

    private static WeightRepacking.RepackedWeight TryRepack(nint ptr, QuantizationType qt, int m, int k)
    {
        if (!WeightRepacking.IsRepackable(qt))
            return default;
        return WeightRepacking.RepackR4(ptr, qt, m, k);
    }

    /// <summary>Frees all R4-interleaved weight buffers.</summary>
    public void Dispose()
    {
        if (RepackedLayers is not null)
        {
            foreach (var rl in RepackedLayers)
                rl.Dispose();
            RepackedLayers = null;
        }
        RepackedOutput?.Dispose();
        RepackedOutput = null;
    }

    private static TransformerLayerWeights LoadLayer(
        int layerIdx,
        nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        ModelConfig config,
        ref int deltaNetCounter)
    {
        string prefix = $"blk.{layerIdx}";
        int hiddenSize = config.HiddenSize;

        // Attention norm — dequantize to float[]
        var attnNormDesc = tensors[$"{prefix}.attn_norm.weight"];
        float[] attnNorm = DequantizeNorm(dataBase, attnNormDesc, hiddenSize);

        // FFN norm — check ffn_norm.weight or post_attention_norm.weight
        var ffnNormDesc = tensors.TryGetValue($"{prefix}.ffn_norm.weight", out var ffnDesc)
            ? ffnDesc
            : tensors[$"{prefix}.post_attention_norm.weight"];
        float[] ffnNorm = DequantizeNorm(dataBase, ffnNormDesc, hiddenSize);

        // FFN projections — check for fused gate+up (Phi-3 style: ffn_up.weight has 2x intermediate rows)
        nint gatePtr, upPtr, downPtr;
        QuantizationType gateQt, upQt, downQt;
        int gateM, gateK, upM, upK, downM, downK;
        float[]? gateBias, upBias, downBias;

        (downPtr, downQt, downM, downK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_down.weight"]);
        downBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_down.bias");

        if (tensors.TryGetValue($"{prefix}.ffn_gate.weight", out var gateDesc))
        {
            // Standard separate gate/up (Llama, Mistral, Qwen)
            (gatePtr, gateQt, gateM, gateK) = LoadLinear(dataBase, gateDesc);
            (upPtr, upQt, upM, upK) = LoadLinear(dataBase, tensors[$"{prefix}.ffn_up.weight"]);
            gateBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_gate.bias");
            upBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.ffn_up.bias");
        }
        else
        {
            // Fused gate+up in ffn_up.weight (Phi-3 style)
            var fusedDesc = tensors[$"{prefix}.ffn_up.weight"];
            nint fusedPtr = dataBase + (nint)fusedDesc.DataOffset;
            int inputDim = fusedDesc.Shape[0];
            int fusedOutputDim = fusedDesc.Shape[1];
            int halfDim = fusedOutputDim / 2;
            long rowBytes = Dequantize.RowByteSize(inputDim, fusedDesc.QuantizationType);

            gatePtr = fusedPtr; gateQt = fusedDesc.QuantizationType; gateM = halfDim; gateK = inputDim;
            upPtr = fusedPtr + (nint)(halfDim * rowBytes); upQt = fusedDesc.QuantizationType; upM = halfDim; upK = inputDim;

            if (tensors.TryGetValue($"{prefix}.ffn_up.bias", out var fusedBiasDesc))
            {
                nint biasPtr = dataBase + (nint)fusedBiasDesc.DataOffset;
                gateBias = new float[halfDim];
                upBias = new float[halfDim];
                Dequantize.ToFloat32(biasPtr, halfDim, fusedBiasDesc.QuantizationType, gateBias);
                Dequantize.ToFloat32(biasPtr + halfDim * sizeof(float), halfDim, fusedBiasDesc.QuantizationType, upBias);
            }
            else
            {
                gateBias = null;
                upBias = null;
            }
        }

        int fullAttnInterval = config.SsmConfig?.FullAttentionInterval ?? 4;
        bool isDeltaNet = config.Architecture == Architecture.Qwen35 && ((layerIdx + 1) % fullAttnInterval != 0);

        if (isDeltaNet)
        {
            int deltaNetIdx = deltaNetCounter++;
            var (qkvPtr, qkvQt, qkvM, qkvK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_qkv.weight"]);
            var (gateZPtr, gateZQt, gateZM, gateZK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_gate.weight"]);
            var (betaPtr, betaQt, betaM, betaK) = LoadLinear(dataBase, tensors[$"{prefix}.ssm_beta.weight"]);
            var (alphaPtr, alphaQt, alphaM, alphaK) = LoadLinear(dataBase, tensors[$"{prefix}.ssm_alpha.weight"]);

            var convDesc = tensors[$"{prefix}.ssm_conv1d.weight"];
            nint convPtr = dataBase + (nint)convDesc.DataOffset;
            int convKernel = convDesc.Shape[0];
            int convChannels = convDesc.Shape[1];

            int timeStepRank = config.SsmConfig?.TimeStepRank ?? 32;
            int stateSize = config.SsmConfig?.StateSize ?? 128;
            float[] ssmDtBias = DequantizeNorm(dataBase, tensors[$"{prefix}.ssm_dt.bias"], timeStepRank);
            float[] ssmA = DequantizeNorm(dataBase, tensors[$"{prefix}.ssm_a"], timeStepRank);
            float[] ssmNormWeight = DequantizeNorm(dataBase, tensors[$"{prefix}.ssm_norm.weight"], stateSize);

            var (ssmOutPtr, ssmOutQt, ssmOutM, ssmOutK) = LoadLinear(dataBase, tensors[$"{prefix}.ssm_out.weight"]);

            return TransformerLayerWeights.CreateDeltaNet(
                deltaNetIdx,
                attnNorm,
                ffnNorm,
                gatePtr, gateQt, gateM, gateK,
                upPtr, upQt, upM, upK,
                downPtr, downQt, downM, downK,
                qkvPtr, qkvQt, qkvM, qkvK,
                gateZPtr, gateZQt, gateZM, gateZK,
                betaPtr, betaQt, betaM, betaK,
                alphaPtr, alphaQt, alphaM, alphaK,
                convPtr, convChannels, convKernel,
                ssmDtBias, ssmA, ssmNormWeight,
                ssmOutPtr, ssmOutQt, ssmOutM, ssmOutK,
                gateBias, upBias, downBias);
        }

        // Standard or Qwen3.5 Full Attention layer
        nint qPtr, kPtr, vPtr;
        QuantizationType qQt, kQt, vQt;
        int qM, qK, kM, kK, vM, vK;

        if (config.Architecture != Architecture.Qwen35 && tensors.TryGetValue($"{prefix}.attn_qkv.weight", out var qkvDesc))
        {
            // Fused QKV (Phi-3 style)
            nint qkvPtr = dataBase + (nint)qkvDesc.DataOffset;
            int inputDim = qkvDesc.Shape[0];
            long rowBytes = Dequantize.RowByteSize(inputDim, qkvDesc.QuantizationType);
            int qDim = config.NumAttentionHeads * config.HeadDim;
            int kvDim = config.NumKvHeads * config.HeadDim;

            qPtr = qkvPtr; qQt = qkvDesc.QuantizationType; qM = qDim; qK = inputDim;
            kPtr = qkvPtr + (nint)(qDim * rowBytes); kQt = qkvDesc.QuantizationType; kM = kvDim; kK = inputDim;
            vPtr = qkvPtr + (nint)((qDim + kvDim) * rowBytes); vQt = qkvDesc.QuantizationType; vM = kvDim; vK = inputDim;
        }
        else
        {
            // Separate Q/K/V
            (qPtr, qQt, qM, qK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_q.weight"]);
            (kPtr, kQt, kM, kK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_k.weight"]);
            (vPtr, vQt, vM, vK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_v.weight"]);
        }

        var (oPtr, oQt, oM, oK) = LoadLinear(dataBase, tensors[$"{prefix}.attn_output.weight"]);

        float[]? qBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_q.bias");
        float[]? kBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_k.bias");
        float[]? vBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_v.bias");
        float[]? oBias = LoadOptionalBias(dataBase, tensors, $"{prefix}.attn_output.bias");

        float[]? qNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_q_norm.weight", config.HeadDim);
        float[]? kNormWeight = LoadOptionalNorm(dataBase, tensors, $"{prefix}.attn_k_norm.weight", config.HeadDim);

        return new TransformerLayerWeights(
            attnNorm,
            qPtr, qQt, qM, qK,
            kPtr, kQt, kM, kK,
            vPtr, vQt, vM, vK,
            oPtr, oQt, oM, oK,
            ffnNorm,
            gatePtr, gateQt, gateM, gateK,
            upPtr, upQt, upM, upK,
            downPtr, downQt, downM, downK,
            qBias, kBias, vBias, oBias,
            gateBias, upBias, downBias,
            qNormWeight, kNormWeight);
    }

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) LoadLinear(
        nint dataBase, GgufTensorDescriptor desc)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        // GGUF: Dimensions[0] = input dim (K), Dimensions[1] = output dim (M)
        int k = desc.Shape[0];
        int m = desc.Shape[1];
        return (ptr, desc.QuantizationType, m, k);
    }

    private static float[] DequantizeNorm(nint dataBase, GgufTensorDescriptor desc, int expectedSize)
    {
        nint ptr = dataBase + (nint)desc.DataOffset;
        float[] result = new float[expectedSize];
        Dequantize.ToFloat32(ptr, expectedSize, desc.QuantizationType, result);
        return result;
    }

    /// <summary>
    /// Loads an optional norm weight tensor. Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalNorm(nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string name, int expectedSize)
    {
        if (!tensors.TryGetValue(name, out var desc)) return null;
        return DequantizeNorm(dataBase, desc, expectedSize);
    }

    /// <summary>
    /// Loads an optional bias tensor (F32 in GGUF). Returns null when the tensor is absent.
    /// </summary>
    private static float[]? LoadOptionalBias(nint dataBase,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string name)
    {
        if (!tensors.TryGetValue(name, out var desc)) return null;
        int size = (int)desc.Shape.ElementCount;
        float[] result = new float[size];
        Dequantize.ToFloat32(dataBase + (nint)desc.DataOffset, size, desc.QuantizationType, result);
        return result;
    }
}
