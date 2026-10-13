using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Hip;

/// <summary>
/// Per-layer GPU weight pointers for DotLLM.Hip.
/// Quantized linear projections are uploaded exactly once and dispatched via quantized GEMV during decode
/// or on-the-fly dequant during prefill.
/// </summary>
internal readonly struct HipLayerWeights
{
    // FP16 dequantized weights on device [outputDim, inputDim] (0 when using on-the-fly dequant or GEMV)
    public readonly nint Q, K, V, O, Gate, Up, Down;

    // Original quantized weights on device (for decode quantized GEMV and on-the-fly prefill dequant)
    public readonly nint QQuant, KQuant, VQuant, OQuant, GateQuant, UpQuant, DownQuant;
    public readonly QuantizationType QQuantType, KQuantType, VQuantType, OQuantType;
    public readonly QuantizationType GateQuantType, UpQuantType, DownQuantType;
    public readonly int QOutputDim, QInputDim, KOutputDim, KInputDim;
    public readonly int VOutputDim, VInputDim, OOutputDim, OInputDim;
    public readonly int GateOutputDim, GateInputDim, UpOutputDim, UpInputDim;
    public readonly int DownOutputDim, DownInputDim;

    // Norm weights on device (FP16)
    public readonly nint AttnNormWeight, FfnNormWeight;
    public readonly nint QNormWeight, KNormWeight; // 0 when absent

    // Bias on device (FP16, 0 when absent)
    public readonly nint QBias, KBias, VBias, OBias;
    public readonly nint GateBias, UpBias, DownBias;

    public HipLayerWeights(
        nint q, int qOut, int qIn, nint k, int kOut, int kIn,
        nint v, int vOut, int vIn, nint o, int oOut, int oIn,
        nint gate, int gateOut, int gateIn, nint up, int upOut, int upIn,
        nint down, int downOut, int downIn,
        nint attnNorm, nint ffnNorm,
        nint qNorm, nint kNorm,
        nint qBias, nint kBias, nint vBias, nint oBias,
        nint gateBias, nint upBias, nint downBias,
        nint qQuant, QuantizationType qQt, nint kQuant, QuantizationType kQt,
        nint vQuant, QuantizationType vQt, nint oQuant, QuantizationType oQt,
        nint gateQuant, QuantizationType gateQt, nint upQuant, QuantizationType upQt,
        nint downQuant, QuantizationType downQt)
    {
        Q = q; QOutputDim = qOut; QInputDim = qIn;
        K = k; KOutputDim = kOut; KInputDim = kIn;
        V = v; VOutputDim = vOut; VInputDim = vIn;
        O = o; OOutputDim = oOut; OInputDim = oIn;
        Gate = gate; GateOutputDim = gateOut; GateInputDim = gateIn;
        Up = up; UpOutputDim = upOut; UpInputDim = upIn;
        Down = down; DownOutputDim = downOut; DownInputDim = downIn;
        AttnNormWeight = attnNorm; FfnNormWeight = ffnNorm;
        QNormWeight = qNorm; KNormWeight = kNorm;
        QBias = qBias; KBias = kBias; VBias = vBias; OBias = oBias;
        GateBias = gateBias; UpBias = upBias; DownBias = downBias;
        QQuant = qQuant; QQuantType = qQt; KQuant = kQuant; KQuantType = kQt;
        VQuant = vQuant; VQuantType = vQt; OQuant = oQuant; OQuantType = oQt;
        GateQuant = gateQuant; GateQuantType = gateQt;
        UpQuant = upQuant; UpQuantType = upQt;
        DownQuant = downQuant; DownQuantType = downQt;
    }
}

/// <summary>
/// Manages all model weights on GPU for DotLLM.Hip.
/// Implements strict single-copy uploads (no triple uploads or temporary leaks, §5.2)
/// and enforces VRAM budget limits with aborts when exceeding 95% (W3).
/// </summary>
internal sealed class HipWeights : IDisposable
{
    public HipLayerWeights[] Layers { get; }
    public nint TokenEmbedDevice { get; }
    public QuantizationType TokenEmbedQuantType { get; }
    public nint OutputNormWeight { get; }
    public nint OutputWeight { get; }
    public int OutputOutputDim { get; }
    public int OutputInputDim { get; }
    public nint OutputWeightQuant { get; }
    public QuantizationType OutputQuantType { get; }

    private readonly List<nint> _allAllocations;

    private HipWeights(HipLayerWeights[] layers, nint tokenEmbed, QuantizationType tokenEmbedQt,
                       nint outputNorm, nint outputWeight, int outputOutDim, int outputInDim,
                       nint outputWeightQuant, QuantizationType outputQt,
                       List<nint> allocs)
    {
        Layers = layers;
        TokenEmbedDevice = tokenEmbed;
        TokenEmbedQuantType = tokenEmbedQt;
        OutputNormWeight = outputNorm;
        OutputWeight = outputWeight;
        OutputOutputDim = outputOutDim;
        OutputInputDim = outputInDim;
        OutputWeightQuant = outputWeightQuant;
        OutputQuantType = outputQt;
        _allAllocations = allocs;
    }

    /// <summary>
    /// Uploads weights from CPU (GGUF mmap) to GPU. Exactly one quantized copy per weight is uploaded.
    /// </summary>
    public static HipWeights LoadFromGguf(TransformerWeights cpuWeights, ModelConfig config,
                                          HipKernels kernels, nint stream,
                                          out string? vramWarning)
    {
        // 1. Initial VRAM budget check
        HipApi.hipMemGetInfo(out nuint freeBefore, out nuint totalVram).ThrowOnError();
        if (totalVram > 0)
        {
            double usedRatio = 1.0 - ((double)freeBefore / (double)totalVram);
            if (usedRatio > 0.95)
            {
                throw new InvalidOperationException(
                    $"Cannot load model: GPU VRAM usage is already at {usedRatio * 100:F1}% " +
                    $"({(totalVram - freeBefore) / (1024 * 1024)} MB used / {totalVram / (1024 * 1024)} MB total). " +
                    $"Exceeds 95% threshold (W3).");
            }
        }

        // Estimate weight size
        long estimatedWeightBytes = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            ref readonly var lw = ref cpuWeights.Layers[i];
            estimatedWeightBytes += Dequantize.RowByteSize(lw.QInputDim, lw.QQuantType) * lw.QOutputDim;
            estimatedWeightBytes += Dequantize.RowByteSize(lw.KInputDim, lw.KQuantType) * lw.KOutputDim;
            estimatedWeightBytes += Dequantize.RowByteSize(lw.VInputDim, lw.VQuantType) * lw.VOutputDim;
            estimatedWeightBytes += Dequantize.RowByteSize(lw.OInputDim, lw.OQuantType) * lw.OOutputDim;
            estimatedWeightBytes += Dequantize.RowByteSize(lw.GateInputDim, lw.GateQuantType) * lw.GateOutputDim;
            estimatedWeightBytes += Dequantize.RowByteSize(lw.UpInputDim, lw.UpQuantType) * lw.UpOutputDim;
            estimatedWeightBytes += Dequantize.RowByteSize(lw.DownInputDim, lw.DownQuantType) * lw.DownOutputDim;
        }

        vramWarning = null;
        if (totalVram > 0 && estimatedWeightBytes > (long)freeBefore)
        {
            long modelMb = estimatedWeightBytes / (1024 * 1024);
            long freeMb = (long)freeBefore / (1024 * 1024);
            long totalMb = (long)totalVram / (1024 * 1024);
            vramWarning = $"Model weights (~{modelMb} MB) exceed currently free VRAM ({freeMb}/{totalMb} MB free).";
        }

        var allocs = new List<nint>();

        // 2. Token embeddings
        nint tokenEmbed;
        var tokenEmbedQt = cpuWeights.TokenEmbedQuantType;
        if (tokenEmbedQt is QuantizationType.F32 or QuantizationType.F16 or QuantizationType.Q8_0)
        {
            long embedBytes = Dequantize.RowByteSize(config.HiddenSize, tokenEmbedQt) * config.VocabSize;
            tokenEmbed = AllocAndUpload(cpuWeights.TokenEmbedWeight, (nuint)embedBytes, allocs);
        }
        else
        {
            // For other formats, dequant entire table to FP16 at load time
            tokenEmbed = UploadAndDequant(cpuWeights.TokenEmbedWeight, tokenEmbedQt,
                config.VocabSize, config.HiddenSize, allocs, kernels, stream);
            tokenEmbedQt = QuantizationType.F16;
        }

        // 3. Output norm + LM head
        nint outputNorm = UploadNormWeight(cpuWeights.OutputNormWeight, allocs);

        nint outputWeightQuant = UploadQuantized(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
            cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs);

        bool lmHeadHasGemv = HipKernels.HasQuantizedGemv(cpuWeights.OutputQuantType);
        nint outputWeight = (!IsQuantized(cpuWeights.OutputQuantType) || !lmHeadHasGemv)
            ? UploadAndDequant(cpuWeights.OutputWeight, cpuWeights.OutputQuantType,
                cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim, allocs, kernels, stream)
            : 0;

        // 4. Per-layer weights
        int layerCount = config.NumLayers;
        var layers = new HipLayerWeights[layerCount];
        for (int i = 0; i < layerCount; i++)
        {
            ref readonly var lw = ref cpuWeights.Layers[i];

            nint q = SkipFp16(lw.QQuantType) ? 0 : UploadAndDequant(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs, kernels, stream);
            nint k = SkipFp16(lw.KQuantType) ? 0 : UploadAndDequant(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs, kernels, stream);
            nint v = SkipFp16(lw.VQuantType) ? 0 : UploadAndDequant(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs, kernels, stream);
            nint o = SkipFp16(lw.OQuantType) ? 0 : UploadAndDequant(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs, kernels, stream);
            nint gate = SkipFp16(lw.GateQuantType) ? 0 : UploadAndDequant(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs, kernels, stream);
            nint up = SkipFp16(lw.UpQuantType) ? 0 : UploadAndDequant(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs, kernels, stream);
            nint down = SkipFp16(lw.DownQuantType) ? 0 : UploadAndDequant(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs, kernels, stream);

            // Upload raw quantized weights (single copy on device)
            nint qQuant = UploadQuantized(lw.QWeight, lw.QQuantType, lw.QOutputDim, lw.QInputDim, allocs);
            nint kQuant = UploadQuantized(lw.KWeight, lw.KQuantType, lw.KOutputDim, lw.KInputDim, allocs);
            nint vQuant = UploadQuantized(lw.VWeight, lw.VQuantType, lw.VOutputDim, lw.VInputDim, allocs);
            nint oQuant = UploadQuantized(lw.OWeight, lw.OQuantType, lw.OOutputDim, lw.OInputDim, allocs);
            nint gateQuant = UploadQuantized(lw.GateWeight, lw.GateQuantType, lw.GateOutputDim, lw.GateInputDim, allocs);
            nint upQuant = UploadQuantized(lw.UpWeight, lw.UpQuantType, lw.UpOutputDim, lw.UpInputDim, allocs);
            nint downQuant = UploadQuantized(lw.DownWeight, lw.DownQuantType, lw.DownOutputDim, lw.DownInputDim, allocs);

            nint attnNorm = UploadNormWeight(lw.AttnNormWeight, allocs);
            nint ffnNorm = UploadNormWeight(lw.FfnNormWeight, allocs);

            nint qNorm = lw.QNormWeight is not null ? UploadNormWeight(lw.QNormWeight, allocs) : 0;
            nint kNorm = lw.KNormWeight is not null ? UploadNormWeight(lw.KNormWeight, allocs) : 0;

            nint qBias = UploadBias(lw.QBias, allocs);
            nint kBias = UploadBias(lw.KBias, allocs);
            nint vBias = UploadBias(lw.VBias, allocs);
            nint oBias = UploadBias(lw.OBias, allocs);
            nint gateBias = UploadBias(lw.GateBias, allocs);
            nint upBias = UploadBias(lw.UpBias, allocs);
            nint downBias = UploadBias(lw.DownBias, allocs);

            layers[i] = new HipLayerWeights(
                q, lw.QOutputDim, lw.QInputDim, k, lw.KOutputDim, lw.KInputDim,
                v, lw.VOutputDim, lw.VInputDim, o, lw.OOutputDim, lw.OInputDim,
                gate, lw.GateOutputDim, lw.GateInputDim, up, lw.UpOutputDim, lw.UpInputDim,
                down, lw.DownOutputDim, lw.DownInputDim,
                attnNorm, ffnNorm, qNorm, kNorm,
                qBias, kBias, vBias, oBias, gateBias, upBias, downBias,
                qQuant, lw.QQuantType, kQuant, lw.KQuantType,
                vQuant, lw.VQuantType, oQuant, lw.OQuantType,
                gateQuant, lw.GateQuantType, upQuant, lw.UpQuantType,
                downQuant, lw.DownQuantType);
        }

        // Sync stream
        HipApi.hipStreamSynchronize(stream).ThrowOnError();

        // 5. Final VRAM verification after upload
        HipApi.hipMemGetInfo(out nuint freeAfter, out totalVram).ThrowOnError();
        if (totalVram > 0)
        {
            double usedRatio = 1.0 - ((double)freeAfter / (double)totalVram);
            if (usedRatio > 0.95)
            {
                throw new InvalidOperationException(
                    $"VRAM usage exceeded 95% threshold after loading model weights ({usedRatio * 100:F1}% used: " +
                    $"{(totalVram - freeAfter) / (1024 * 1024)} MB used / {totalVram / (1024 * 1024)} MB total). " +
                    $"Aborting to prevent silent driver spill to system memory (W3).");
            }
        }

        return new HipWeights(layers, tokenEmbed, tokenEmbedQt,
            outputNorm, outputWeight, cpuWeights.OutputOutputDim, cpuWeights.OutputInputDim,
            outputWeightQuant, cpuWeights.OutputQuantType, allocs);
    }

    /// <summary>Upload raw quantized weight bytes to GPU (no dequant). For decode quantized GEMV.</summary>
    private static nint UploadQuantized(nint hostPtr, QuantizationType qt,
                                        int outputDim, int inputDim, List<nint> allocs)
    {
        if (qt is QuantizationType.F16 or QuantizationType.F32)
            return 0;

        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        return AllocAndUpload(hostPtr, (nuint)quantBytes, allocs);
    }

    /// <summary>Upload quantized weight to GPU, then dequantize to FP16 on device.</summary>
    private static nint UploadAndDequant(nint hostPtr, QuantizationType qt,
                                         int outputDim, int inputDim,
                                         List<nint> allocs, HipKernels kernels, nint stream)
    {
        int totalElements = outputDim * inputDim;

        if (qt == QuantizationType.F16)
        {
            long bytes = (long)totalElements * sizeof(ushort);
            return AllocAndUpload(hostPtr, (nuint)bytes, allocs);
        }

        if (qt == QuantizationType.F32)
        {
            long f32Bytes = (long)totalElements * sizeof(float);
            nint devF32 = AllocAndUpload(hostPtr, (nuint)f32Bytes, allocs);
            long f16Bytes = (long)totalElements * sizeof(ushort);
            HipApi.hipMalloc(out nint devF16, (nuint)f16Bytes).ThrowOnError();
            allocs.Add(devF16);
            kernels.LaunchConvertF32ToF16(devF32, devF16, totalElements, stream);
            HipApi.hipStreamSynchronize(stream).ThrowOnError();
            allocs.Remove(devF32);
            HipApi.hipFree(devF32);
            return devF16;
        }

        // Upload raw bytes to temp buffer, dequantize, free temp buffer
        long quantBytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        HipApi.hipMalloc(out nint tempQuant, (nuint)quantBytes).ThrowOnError();
        HipApi.hipMemcpy(tempQuant, hostPtr, (nuint)quantBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();

        long fp16Bytes = (long)totalElements * sizeof(ushort);
        HipApi.hipMalloc(out nint devFp16, (nuint)fp16Bytes).ThrowOnError();
        allocs.Add(devFp16);

        kernels.LaunchDequantToF16(tempQuant, qt, devFp16, totalElements, stream);
        HipApi.hipStreamSynchronize(stream).ThrowOnError();
        HipApi.hipFree(tempQuant);

        return devFp16;
    }

    /// <summary>Upload float[] norm weight directly as FP16 on device.</summary>
    private static unsafe nint UploadNormWeight(float[] weight, List<nint> allocs)
    {
        int n = weight.Length;
        nuint f16Bytes = (nuint)(n * sizeof(Half));
        HipApi.hipMalloc(out nint devF16, f16Bytes).ThrowOnError();
        allocs.Add(devF16);

        Half[] halfs = new Half[n];
        for (int i = 0; i < n; i++)
            halfs[i] = (Half)weight[i];

        fixed (Half* ptr = halfs)
        {
            HipApi.hipMemcpy(devF16, (nint)ptr, f16Bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
        }

        return devF16;
    }

    /// <summary>Upload optional float[] bias as FP16 on device. Returns 0 if bias is null.</summary>
    private static nint UploadBias(float[]? bias, List<nint> allocs)
    {
        if (bias is null) return 0;
        return UploadNormWeight(bias, allocs);
    }

    private static bool IsQuantized(QuantizationType qt) =>
        qt is not QuantizationType.F16 and not QuantizationType.F32;

    /// <summary>
    /// For all supported quantized types with GEMV kernels, persistent FP16 copies are skipped.
    /// Prefill uses on-the-fly dequant into a scratch buffer, and decode uses quantized GEMV.
    /// </summary>
    private static bool SkipFp16(QuantizationType qt) => HipKernels.HasQuantizedGemv(qt);

    /// <summary>Allocate device memory and copy host data.</summary>
    private static nint AllocAndUpload(nint hostPtr, nuint bytes, List<nint> allocs)
    {
        HipApi.hipMalloc(out nint devPtr, bytes).ThrowOnError();
        allocs.Add(devPtr);
        HipApi.hipMemcpy(devPtr, hostPtr, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
        return devPtr;
    }

    public void Dispose()
    {
        foreach (nint ptr in _allAllocations)
        {
            if (ptr != 0)
                HipApi.hipFree(ptr);
        }
        _allAllocations.Clear();
    }
}
