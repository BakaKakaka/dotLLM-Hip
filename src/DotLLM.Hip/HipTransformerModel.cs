using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Hip;

/// <summary>
/// GPU-accelerated transformer forward pass using AMD HIP (RDNA3 / RDNA4).
/// All operations execute on a single HIP stream with no host synchronization until the final logits D2H transfer.
/// Implements FP32 residual stream (D5), pinned async H2D/D2H transfers, single-copy weights,
/// and sets the active device upon every Forward invocation (W10).
/// </summary>
public sealed unsafe class HipTransformerModel : IModel, IGpuModelInfo
{
    private readonly HipWeights _weights;
    private readonly HipForwardState _state;
    private readonly HipStream _stream;
    private readonly HipBlasHandle _rocblas;
    private readonly HipKernels _kernels;
    private readonly GgufFile _gguf;
    private readonly int _deviceId;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly int _ropeType;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes;

    /// <inheritdoc/>
    public string? VramWarning { get; }

    /// <summary>Debug: limit the number of transformer layers processed. 0 = all layers (default). -1 = skip all layers.</summary>
    internal int DebugMaxLayers { get; set; }

    /// <summary>Debug: override RoPE type. -1 = use model's type (default).</summary>
    internal int DebugRopeTypeOverride { get; set; } = -1;

    /// <summary>Debug: skip bias add operations.</summary>
    internal bool DebugSkipBias { get; set; }

    private HipTransformerModel(
        ModelConfig config, HipWeights weights, HipForwardState state,
        HipStream stream, HipBlasHandle rocblas,
        HipKernels kernels, GgufFile gguf, int deviceId,
        float ropeTheta, int ropeDim, int ropeType, string? vramWarning)
    {
        Config = config;
        _weights = weights;
        _state = state;
        _stream = stream;
        _rocblas = rocblas;
        _kernels = kernels;
        _gguf = gguf;
        _deviceId = deviceId;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeType = ropeType;
        VramWarning = vramWarning;
    }

    /// <summary>
    /// Loads a transformer model onto the AMD GPU from an opened GGUF file.
    /// </summary>
    /// <param name="gguf">Opened GGUF file (must remain alive for model lifetime).</param>
    /// <param name="config">Model configuration extracted from GGUF metadata.</param>
    /// <param name="deviceSpecifier">Device specifier (e.g. "hip", "hip:gfx1100", "hip:gfx1201", "hip:0").</param>
    public static HipTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? deviceSpecifier = "hip")
    {
        var device = HipDevice.ResolveDevice(deviceSpecifier);
        return LoadFromGguf(gguf, config, device);
    }

    /// <summary>
    /// Loads a transformer model onto the specified AMD GPU device.
    /// </summary>
    public static HipTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, HipDevice device)
    {
        device.MakeCurrent();

        var stream = HipStream.Create();
        var rocblas = HipBlasHandle.Create();
        rocblas.SetStream(stream);

        var kernels = new HipKernels(device.Arch);

        // Load CPU weights (mmap references only)
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);

        // Upload weights to GPU (with VRAM budget checking)
        var weights = HipWeights.LoadFromGguf(cpuWeights, config, kernels, stream.Handle, out string? vramWarning);

        // Create scratch buffers
        var state = new HipForwardState(
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize);

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        int ropeType = (config.RoPEConfig?.Type == RoPEType.NeoX) ? 1 : 0;

        return new HipTransformerModel(config, weights, state, stream, rocblas,
            kernels, gguf, device.Ordinal, ropeTheta, ropeDim, ropeType, vramWarning);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        // Enforce thread-local active device on entry (W10 mandate)
        HipApi.hipSetDevice(_deviceId).ThrowOnError();

        int seqLen = tokenIds.Length;
        int hiddenSize = Config.HiddenSize;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int slidingWindow = Config.SlidingWindowSize ?? 0;

        _state.EnsureCapacity(seqLen);
        nint s = _stream.Handle;

        // 1 & 2. Embedding lookup → FP32 residual stream (D5 mandate)
        if (seqLen == 1)
        {
            // Decode fast path: scalar token ID passed directly as kernel argument,
            // eliminating DMA submission and driver synchronization overhead.
            int tokenId = tokenIds[0];
            switch (_weights.TokenEmbedQuantType)
            {
                case QuantizationType.F32:
                    _kernels.LaunchEmbeddingDecodeF32(_weights.TokenEmbedDevice, tokenId, _state.Residual,
                        hiddenSize, s);
                    break;
                case QuantizationType.F16:
                    _kernels.LaunchEmbeddingDecodeF16(_weights.TokenEmbedDevice, tokenId, _state.Residual,
                        hiddenSize, s);
                    break;
                case QuantizationType.Q8_0:
                    _kernels.LaunchEmbeddingDecodeQ8_0(_weights.TokenEmbedDevice, tokenId, _state.Residual,
                        hiddenSize, s);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported embedding quant type: {_weights.TokenEmbedQuantType}");
            }
        }
        else
        {
            // Prefill: Pinned async H2D copy of token IDs and positions (H2.5 mandate)
            var tokenSpan = _state.TokenIdsHost.AsSpan<int>();
            var posSpan = _state.PositionsHost.AsSpan<int>();
            tokenIds.CopyTo(tokenSpan[..seqLen]);
            positions.CopyTo(posSpan[..seqLen]);

            nuint seqIntBytes = (nuint)(seqLen * sizeof(int));
            HipApi.hipMemcpyAsync(_state.TokenIdsDevice, _state.TokenIdsHost.Pointer, seqIntBytes,
                HipApi.hipMemcpyHostToDevice, s).ThrowOnError();
            HipApi.hipMemcpyAsync(_state.PositionsDevice, _state.PositionsHost.Pointer, seqIntBytes,
                HipApi.hipMemcpyHostToDevice, s).ThrowOnError();

            switch (_weights.TokenEmbedQuantType)
            {
                case QuantizationType.F32:
                    _kernels.LaunchEmbeddingF32(_weights.TokenEmbedDevice, _state.TokenIdsDevice, _state.Residual,
                        seqLen, hiddenSize, s);
                    break;
                case QuantizationType.F16:
                    _kernels.LaunchEmbeddingF16(_weights.TokenEmbedDevice, _state.TokenIdsDevice, _state.Residual,
                        seqLen, hiddenSize, s);
                    break;
                case QuantizationType.Q8_0:
                    _kernels.LaunchEmbeddingQ8_0(_weights.TokenEmbedDevice, _state.TokenIdsDevice, _state.Residual,
                        seqLen, hiddenSize, s);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported embedding quant type: {_weights.TokenEmbedQuantType}");
            }
        }

        // 3. First layer attention norm (FP32 residual in → FP16 NormOutput)
        if (Config.NumLayers > 0)
        {
            _kernels.LaunchRmsNormF32InF16OutW16(_state.Residual, _weights.Layers[0].AttnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);
        }

        // 4. Transformer layers
        int numLayers = DebugMaxLayers switch
        {
            < 0 => 0,   // skip all layers (embedding + LM head only)
            0 => Config.NumLayers,
            _ => Math.Min(DebugMaxLayers, Config.NumLayers)
        };

        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // ── ATTENTION BLOCK ──
            var qQt = lw.QQuant != 0 ? lw.QQuantType : QuantizationType.F16;
            var kQt = lw.KQuant != 0 ? lw.KQuantType : QuantizationType.F16;
            var vQt = lw.VQuant != 0 ? lw.VQuantType : QuantizationType.F16;

            bool canFuseQkv = seqLen == 1
                && qQt == kQt && qQt == vQt
                && HipKernels.HasFusedQkvGemv(qQt)
                && (lw.QQuant != 0 || lw.Q != 0)
                && (lw.KQuant != 0 || lw.K != 0)
                && (lw.VQuant != 0 || lw.V != 0);

            bool canFuseQk = seqLen == 1
                && !canFuseQkv
                && qQt == kQt
                && HipKernels.HasFusedQkvGemv(qQt)
                && (lw.QQuant != 0 || lw.Q != 0)
                && (lw.KQuant != 0 || lw.K != 0);

            nint kDest = _state.K;
            nint vDest = _state.V;
            bool directKv = kvCache is HipKvCache && seqLen == 1;
            if (directKv)
            {
                var hk = (HipKvCache)kvCache!;
                kDest = hk.GetKeysWritePtr(layer, positions[0]);
                vDest = hk.GetValuesWritePtr(layer, positions[0]);
            }

            if (canFuseQkv)
            {
                nint qw = lw.QQuant != 0 ? lw.QQuant : lw.Q;
                nint kw = lw.KQuant != 0 ? lw.KQuant : lw.K;
                nint vw = lw.VQuant != 0 ? lw.VQuant : lw.V;
                nint qb = DebugSkipBias ? 0 : lw.QBias;
                nint kb = DebugSkipBias ? 0 : lw.KBias;
                nint vb = DebugSkipBias ? 0 : lw.VBias;

                if (HipKernels.HasFusedQkvGemvMmvq(qQt) && lw.QQuant != 0)
                {
                    _kernels.LaunchQuantizeQ8_1F16(_state.NormOutput, _state.Q8_1Scratch, lw.QInputDim, s);
                    _kernels.LaunchFusedQkvGemvMmvq(
                        qw, kw, vw, qQt,
                        qb, kb, vb,
                        _state.Q8_1Scratch, _state.Q, kDest, vDest,
                        lw.QOutputDim, lw.KOutputDim, lw.VOutputDim, lw.QInputDim, s);
                }
                else
                {
                    _kernels.LaunchFusedQkvGemv(
                        qw, kw, vw, qQt,
                        qb, kb, vb,
                        _state.NormOutput, _state.Q, kDest, vDest,
                        lw.QOutputDim, lw.KOutputDim, lw.VOutputDim, lw.QInputDim, s);
                }
            }
            else if (canFuseQk)
            {
                nint qw = lw.QQuant != 0 ? lw.QQuant : lw.Q;
                nint kw = lw.KQuant != 0 ? lw.KQuant : lw.K;
                nint qb = DebugSkipBias ? 0 : lw.QBias;
                nint kb = DebugSkipBias ? 0 : lw.KBias;

                if (HipKernels.HasFusedQkvGemvMmvq(qQt) && lw.QQuant != 0)
                {
                    _kernels.LaunchQuantizeQ8_1F16(_state.NormOutput, _state.Q8_1Scratch, lw.QInputDim, s);
                    _kernels.LaunchFusedQkvGemvMmvq(
                        qw, kw, 0, qQt,
                        qb, kb, 0,
                        _state.Q8_1Scratch, _state.Q, kDest, 0,
                        lw.QOutputDim, lw.KOutputDim, 0, lw.QInputDim, s);
                }
                else
                {
                    _kernels.LaunchFusedQkvGemv(
                        qw, kw, 0, qQt,
                        qb, kb, 0,
                        _state.NormOutput, _state.Q, kDest, 0,
                        lw.QOutputDim, lw.KOutputDim, 0, lw.QInputDim, s);
                }

                Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, vDest, lw.VOutputDim, lw.VInputDim, seqLen);
                if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF16(vDest, lw.VBias, lw.VOutputDim, seqLen, s);
            }
            else
            {
                Project(lw.QQuant, lw.QQuantType, lw.Q, _state.NormOutput, _state.Q, lw.QOutputDim, lw.QInputDim, seqLen);
                Project(lw.KQuant, lw.KQuantType, lw.K, _state.NormOutput, kDest, lw.KOutputDim, lw.KInputDim, seqLen);
                Project(lw.VQuant, lw.VQuantType, lw.V, _state.NormOutput, vDest, lw.VOutputDim, lw.VInputDim, seqLen);

                // Optional biases (FP16)
                if (lw.QBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF16(_state.Q, lw.QBias, lw.QOutputDim, seqLen, s);
                if (lw.KBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF16(kDest, lw.KBias, lw.KOutputDim, seqLen, s);
                if (lw.VBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF16(vDest, lw.VBias, lw.VOutputDim, seqLen, s);
            }

            // Optional QK-norms (FP16)
            if (lw.QNormWeight != 0)
                _kernels.LaunchPerHeadRmsNormF16(_state.Q, lw.QNormWeight, eps, numHeads, headDim, seqLen, s);
            if (lw.KNormWeight != 0)
                _kernels.LaunchPerHeadRmsNormF16(kDest, lw.KNormWeight, eps, numKvHeads, headDim, seqLen, s);

            // RoPE (FP16, in-place on Q and K)
            int effectiveRopeType = DebugRopeTypeOverride >= 0 ? DebugRopeTypeOverride : _ropeType;
            if (seqLen == 1)
            {
                _kernels.LaunchRoPEDecodeF16(_state.Q, kDest, positions[0],
                    numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);
            }
            else
            {
                _kernels.LaunchRoPEF16(_state.Q, kDest, _state.PositionsDevice,
                    seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, effectiveRopeType, s);
            }

            // KV-cache update + Attention
            if (directKv)
            {
                var hk = (HipKvCache)kvCache!;
                hk.AdvancePosition(positions[0]);
                int seqKv = hk.CurrentLength;

                _kernels.LaunchAttention(_state.Q, hk.GetKeysPtr(layer),
                    hk.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else if (kvCache is HipKvCache hipKvCache)
            {
                hipKvCache.UpdateDevice(_state.K, _state.V, positions, seqLen, layer, s);
                int seqKv = hipKvCache.CurrentLength;

                _kernels.LaunchAttention(_state.Q, hipKvCache.GetKeysPtr(layer),
                    hipKvCache.GetValuesPtr(layer), _state.AttnOutput,
                    seqLen, seqKv, numHeads, numKvHeads, headDim,
                    positions[0], slidingWindow, s);
            }
            else
            {
                _kernels.LaunchAttention(_state.Q, _state.K, _state.V, _state.AttnOutput,
                    seqLen, seqLen, numHeads, numKvHeads, headDim,
                    0, slidingWindow, s);
            }

            // O projection → NormOutput
            Project(lw.OQuant, lw.OQuantType, lw.O, _state.AttnOutput, _state.NormOutput, lw.OOutputDim, lw.OInputDim, seqLen);
            if (lw.OBias != 0) _kernels.LaunchBiasAddF16(_state.NormOutput, lw.OBias, lw.OOutputDim, seqLen, s);

            // ── FUSED: attention residual + FFN norm (Residual stays FP32) ──
            _kernels.LaunchFusedAddRmsNormF32ResW16(_state.Residual, _state.NormOutput, lw.FfnNormWeight, _state.NormOutput,
                hiddenSize, eps, seqLen, s);

            // ── FFN BLOCK ──
            var gateQt = lw.GateQuant != 0 ? lw.GateQuantType : QuantizationType.F16;
            var upQt = lw.UpQuant != 0 ? lw.UpQuantType : QuantizationType.F16;

            bool canFuseGateUpSwiglu = seqLen == 1
                && gateQt == upQt
                && HipKernels.HasFusedGateUpSwiglu(gateQt)
                && (lw.GateQuant != 0 || lw.Gate != 0)
                && (lw.UpQuant != 0 || lw.Up != 0);

            if (canFuseGateUpSwiglu)
            {
                nint gw = lw.GateQuant != 0 ? lw.GateQuant : lw.Gate;
                nint uw = lw.UpQuant != 0 ? lw.UpQuant : lw.Up;
                nint gb = DebugSkipBias ? 0 : lw.GateBias;
                nint ub = DebugSkipBias ? 0 : lw.UpBias;

                if (HipKernels.HasFusedGateUpSwigluMmvq(gateQt) && lw.GateQuant != 0)
                {
                    _kernels.LaunchQuantizeQ8_1F16(_state.NormOutput, _state.Q8_1Scratch, lw.GateInputDim, s);
                    _kernels.LaunchFusedGateUpSwigluMmvq(
                        gw, uw, gateQt,
                        gb, ub,
                        _state.Q8_1Scratch, _state.SiluOutput,
                        intermediateSize, lw.GateInputDim, s);
                }
                else
                {
                    _kernels.LaunchFusedGateUpSwiglu(
                        gw, uw, gateQt,
                        gb, ub,
                        _state.NormOutput, _state.SiluOutput,
                        intermediateSize, lw.GateInputDim, s);
                }
            }
            else
            {
                Project(lw.GateQuant, lw.GateQuantType, lw.Gate, _state.NormOutput, _state.FfnGate, lw.GateOutputDim, lw.GateInputDim, seqLen);
                Project(lw.UpQuant, lw.UpQuantType, lw.Up, _state.NormOutput, _state.FfnUp, lw.UpOutputDim, lw.UpInputDim, seqLen);

                if (lw.GateBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF16(_state.FfnGate, lw.GateBias, lw.GateOutputDim, seqLen, s);
                if (lw.UpBias != 0 && !DebugSkipBias) _kernels.LaunchBiasAddF16(_state.FfnUp, lw.UpBias, lw.UpOutputDim, seqLen, s);

                // SwiGLU (FP16)
                _kernels.LaunchSwiGLUF16(_state.FfnGate, _state.FfnUp, _state.SiluOutput,
                    intermediateSize, seqLen, s);
            }

            // Down projection → NormOutput
            Project(lw.DownQuant, lw.DownQuantType, lw.Down, _state.SiluOutput, _state.NormOutput, lw.DownOutputDim, lw.DownInputDim, seqLen);
            if (lw.DownBias != 0) _kernels.LaunchBiasAddF16(_state.NormOutput, lw.DownBias, lw.DownOutputDim, seqLen, s);

            // ── FUSED: FFN residual + next layer attention norm ──
            if (layer < numLayers - 1)
            {
                ref readonly var nextLw = ref _weights.Layers[layer + 1];
                _kernels.LaunchFusedAddRmsNormF32ResW16(_state.Residual, _state.NormOutput, nextLw.AttnNormWeight, _state.NormOutput,
                    hiddenSize, eps, seqLen, s);
            }
            else
            {
                // Last layer: Add FP16 NormOutput into FP32 Residual
                _kernels.LaunchAddF32F16(_state.Residual, _state.NormOutput, _state.Residual,
                    seqLen * hiddenSize, s);
            }
        }

        // 5. Final RMSNorm (last token only, FP32 residual → FP16 NormOutput)
        nint lastResidual = _state.Residual + (nint)((seqLen - 1) * hiddenSize * sizeof(float));
        _kernels.LaunchRmsNormF32InF16OutW16(lastResidual, _weights.OutputNormWeight, _state.NormOutput,
            hiddenSize, eps, 1, s);

        // 6. LM head (last token only)
        if (seqLen == 1 && _weights.OutputWeightQuant != 0 && HipKernels.HasQuantizedGemvF32Out(_weights.OutputQuantType))
        {
            // Direct FP32 output: eliminates LaunchConvertF16ToF32 and preserves full dot-product precision
            _kernels.LaunchQuantizedGemvF32Out(_weights.OutputWeightQuant, _weights.OutputQuantType,
                _state.NormOutput, _state.LogitsF32, vocabSize, hiddenSize, s);
        }
        else
        {
            Project(_weights.OutputWeightQuant, _weights.OutputQuantType, _weights.OutputWeight,
                _state.NormOutput, _state.LogitsF16,
                _weights.OutputOutputDim, _weights.OutputInputDim, 1, preferMmvq: seqLen == 1);
            _kernels.LaunchConvertF16ToF32(_state.LogitsF16, _state.LogitsF32, vocabSize, s);
        }

        // 7. Async D2H copy of FP32 logits to pinned host buffer (H2.5 mandate)
        nuint logitsBytes = (nuint)(vocabSize * sizeof(float));
        HipApi.hipMemcpyAsync(_state.LogitsHost.Pointer, _state.LogitsF32, logitsBytes,
            HipApi.hipMemcpyDeviceToHost, s).ThrowOnError();

        // 8. Stream synchronize (single sync point per forward pass)
        _stream.Synchronize();

        // 9. Zero-copy return of pre-allocated non-owning tensor view
        return _state.LogitsTensor;
    }

    /// <summary>
    /// Dispatches projection as rocBLAS HGEMM (prefill) or quantized/rocBLAS GEMV (decode).
    /// </summary>
    private void Project(nint quantWeight, QuantizationType qt, nint fp16Weight,
                         nint input, nint output, int outputDim, int inputDim, int seqLen, bool preferMmvq = true)
    {
        nint s = _stream.Handle;

        if (seqLen > 1) // Prefill: rocBLAS HGEMM
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                // Quantized: dequant on-the-fly into scratch buffer, then GEMM
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            HipGemm.LinearF16(_rocblas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
        else if (quantWeight != 0 && preferMmvq && HipKernels.HasMmvqGemv(qt)) // Decode: vectorized MMVQ GEMV (H3.2)
        {
            _kernels.LaunchQuantizeQ8_1F16(input, _state.Q8_1Scratch, inputDim, s);
            if (qt == QuantizationType.Q8_0)
                _kernels.LaunchMmvqGemvQ8_0(quantWeight, _state.Q8_1Scratch, output, outputDim, inputDim, s);
            else if (qt == QuantizationType.Q4_K)
                _kernels.LaunchMmvqGemvQ4_K(quantWeight, _state.Q8_1Scratch, output, outputDim, inputDim, s);
            else if (qt == QuantizationType.Q6_K)
                _kernels.LaunchMmvqGemvQ6_K(quantWeight, _state.Q8_1Scratch, output, outputDim, inputDim, s);
        }
        else if (quantWeight != 0 && HipKernels.HasQuantizedGemv(qt)) // Decode: fallback quantized GEMV
        {
            _kernels.LaunchQuantizedGemv(quantWeight, qt, input, output, outputDim, inputDim, s);
        }
        else // Decode fallback: rocBLAS GEMV
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                    outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            HipGemm.GemvF16(_rocblas.Handle, w, input, output, outputDim, inputDim, s);
        }
    }

    /// <summary>
    /// Creates a <see cref="HipKvCache"/> for this model.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for the cache.</param>
    public HipKvCache CreateKvCache(int maxSeqLen)
    {
        HipApi.hipSetDevice(_deviceId).ThrowOnError();
        return new HipKvCache(Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _stream.Synchronize();
        _weights.Dispose();
        _state.Dispose();
        _rocblas.Dispose();
        _kernels.Dispose();
        _stream.Dispose();
    }
}
