using DotLLM.Core.Configuration;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Loads embedded AOT-compiled HIP kernel modules for a given GPU architecture
/// and provides typed launch methods for each kernel.
/// Arguments are marshaled as pointer arrays for <see cref="HipApi.hipModuleLaunchKernel"/>.
/// </summary>
public sealed unsafe class HipKernels : IDisposable
{
    private const int BlockSize = 256;

    private readonly HipModule _embeddingModule;
    private readonly HipModule _rmsnormModule;
    private readonly HipModule _fusedAddRmsNormModule;
    private readonly HipModule _perHeadRmsNormModule;
    private readonly HipModule _ropeModule;
    private readonly HipModule _swigluModule;
    private readonly HipModule _addModule;
    private readonly HipModule _biasAddModule;
    private readonly HipModule _convertModule;
    private readonly HipModule _softmaxModule;
    private readonly HipModule _dequantModule;
    private readonly HipModule _quantizedGemvModule;
    private readonly HipModule _attentionModule;
    private readonly HipModule _fusedGateUpSwigluModule;
    private readonly HipModule _fusedQkvGemvModule;
    private readonly HipModule _quantizeQ8_1Module;
    private readonly HipModule _mmvqModule;
    private readonly HipModule _gatedDeltaNetModule;

    // Embedding functions
    private readonly nint _embeddingF32F32OutFunc;
    private readonly nint _embeddingF16F32OutFunc;
    private readonly nint _embeddingQ8_0F32OutFunc;
    private readonly nint _embeddingF32DecodeF32OutFunc;
    private readonly nint _embeddingF16DecodeF32OutFunc;
    private readonly nint _embeddingQ8_0DecodeF32OutFunc;

    // RMSNorm functions
    private readonly nint _rmsnormF32InF16OutFunc;
    private readonly nint _rmsnormF32InF16OutW16Func;
    private readonly nint _rmsnormF16Func;
    private readonly nint _rmsnormF32Func;

    // Fused Add + RMSNorm functions
    private readonly nint _fusedAddRmsNormF32ResFunc;
    private readonly nint _fusedAddRmsNormF32ResW16Func;
    private readonly nint _fusedAddRmsNormF16Func;

    // Per-head RMSNorm (QK-norm)
    private readonly nint _perHeadRmsNormF16Func;
    private readonly nint _perHeadRmsNormF32Func;

    // RoPE
    private readonly nint _ropeF16Func;
    private readonly nint _ropeDecodeF16Func;
    private readonly nint _ropeF32Func;

    // SwiGLU
    private readonly nint _swigluF16Func;
    private readonly nint _swigluF32Func;

    // Add
    private readonly nint _addF16Func;
    private readonly nint _addF32Func;
    private readonly nint _addF32F16Func;

    // Bias Add
    private readonly nint _biasAddF16Func;
    private readonly nint _biasAddF32Func;

    // Convert
    private readonly nint _convertF16ToF32Func;
    private readonly nint _convertF32ToF16Func;

    // Softmax
    private readonly nint _softmaxF16Func;
    private readonly nint _softmaxF32Func;

    // Dequant
    private readonly nint _dequantQ8_0Func;
    private readonly nint _dequantQ4_0Func;
    private readonly nint _dequantQ5_0Func;
    private readonly nint _dequantQ4_KFunc;
    private readonly nint _dequantQ5_KFunc;
    private readonly nint _dequantQ6_KFunc;

    // Quantized GEMV
    private readonly nint _quantizedGemvQ8_0Func;
    private readonly nint _quantizedGemvQ4_0Func;
    private readonly nint _quantizedGemvQ5_0Func;
    private readonly nint _quantizedGemvQ4_KFunc;
    private readonly nint _quantizedGemvQ5_KFunc;
    private readonly nint _quantizedGemvQ6_KFunc;
    private readonly nint _quantizedGemvQ8_0F32OutFunc;
    private readonly nint _quantizedGemvQ4_KF32OutFunc;
    private readonly nint _quantizedGemvQ6_KF32OutFunc;

    // Attention
    private readonly nint _attentionF16Func;
    private readonly nint _attentionDecodeF16Func;
    private readonly nint _attentionF32Func;

    // Fused Gate + Up + SwiGLU
    private readonly nint _fusedGateUpSwigluQ8_0Func;
    private readonly nint _fusedGateUpSwigluQ4_0Func;
    private readonly nint _fusedGateUpSwigluQ5_0Func;
    private readonly nint _fusedGateUpSwigluQ4_KFunc;
    private readonly nint _fusedGateUpSwigluQ5_KFunc;
    private readonly nint _fusedGateUpSwigluQ6_KFunc;
    private readonly nint _fusedGateUpSwigluF16Func;
    private readonly nint _fusedGateUpSwigluMmvqQ8_0Func;
    private readonly nint _fusedGateUpSwigluMmvqQ4_KFunc;

    // Fused QKV GEMV
    private readonly nint _fusedQkvGemvQ8_0Func;
    private readonly nint _fusedQkvGemvQ4_0Func;
    private readonly nint _fusedQkvGemvQ5_0Func;
    private readonly nint _fusedQkvGemvQ4_KFunc;
    private readonly nint _fusedQkvGemvQ5_KFunc;
    private readonly nint _fusedQkvGemvQ6_KFunc;
    private readonly nint _fusedQkvGemvF16Func;
    private readonly nint _fusedQkvGemvMmvqQ8_0Func;
    private readonly nint _fusedQkvGemvMmvqQ4_KFunc;

    // Quantize Q8_1
    private readonly nint _quantizeQ8_1F16Func;
    private readonly nint _quantizeQ8_1F32Func;

    // MMVQ GEMV
    private readonly nint _mmvqGemvQ8_0Func;
    private readonly nint _mmvqGemvQ8_0Fdot2Func;
    private readonly nint _mmvqGemvQ4_KFunc;
    private readonly nint _mmvqGemvQ6_KFunc;

    // Gated DeltaNet & Attention Gating
    private readonly nint _causalConv1dSiLuFunc;
    private readonly nint _l2NormAndGqaRepeatFunc;
    private readonly nint _computeSsmGatesFunc;
    private readonly nint _computeSsmGatesF16Func;
    private readonly nint _gatedDeltaNetRecurrentFunc;
    private readonly nint _gatedRmsNormSiLuFunc;
    private readonly nint _deinterleaveQGateFunc;
    private readonly nint _mulSigmoidGateFunc;

    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of <see cref="HipKernels"/> loading embedded modules for the given GPU architecture.
    /// </summary>
    /// <param name="arch">The GPU architecture (e.g. "gfx1100" or "gfx1201").</param>
    public HipKernels(string arch)
    {
        _embeddingModule = HipKernelLoader.LoadEmbeddedModule(arch, "embedding");
        _rmsnormModule = HipKernelLoader.LoadEmbeddedModule(arch, "rmsnorm");
        _fusedAddRmsNormModule = HipKernelLoader.LoadEmbeddedModule(arch, "fused_add_rmsnorm");
        _perHeadRmsNormModule = HipKernelLoader.LoadEmbeddedModule(arch, "per_head_rmsnorm");
        _ropeModule = HipKernelLoader.LoadEmbeddedModule(arch, "rope");
        _swigluModule = HipKernelLoader.LoadEmbeddedModule(arch, "swiglu");
        _addModule = HipKernelLoader.LoadEmbeddedModule(arch, "add");
        _biasAddModule = HipKernelLoader.LoadEmbeddedModule(arch, "bias_add");
        _convertModule = HipKernelLoader.LoadEmbeddedModule(arch, "convert");
        _softmaxModule = HipKernelLoader.LoadEmbeddedModule(arch, "softmax");
        _dequantModule = HipKernelLoader.LoadEmbeddedModule(arch, "dequant");
        _quantizedGemvModule = HipKernelLoader.LoadEmbeddedModule(arch, "quantized_gemv");
        _attentionModule = HipKernelLoader.LoadEmbeddedModule(arch, "attention");

        _embeddingF32F32OutFunc = _embeddingModule.GetFunction("embedding_lookup_f32_f32out");
        _embeddingF16F32OutFunc = _embeddingModule.GetFunction("embedding_lookup_f16_f32out");
        _embeddingQ8_0F32OutFunc = _embeddingModule.GetFunction("embedding_lookup_q8_0_f32out");
        _embeddingF32DecodeF32OutFunc = _embeddingModule.GetFunction("embedding_lookup_f32_decode_f32out");
        _embeddingF16DecodeF32OutFunc = _embeddingModule.GetFunction("embedding_lookup_f16_decode_f32out");
        _embeddingQ8_0DecodeF32OutFunc = _embeddingModule.GetFunction("embedding_lookup_q8_0_decode_f32out");

        _rmsnormF32InF16OutFunc = _rmsnormModule.GetFunction("rmsnorm_f32in_f16out");
        _rmsnormF32InF16OutW16Func = _rmsnormModule.GetFunction("rmsnorm_f32in_f16out_w16");
        _rmsnormF16Func = _rmsnormModule.GetFunction("rmsnorm_f16");
        _rmsnormF32Func = _rmsnormModule.GetFunction("rmsnorm_f32");

        _fusedAddRmsNormF32ResFunc = _fusedAddRmsNormModule.GetFunction("fused_add_rmsnorm_f32res");
        _fusedAddRmsNormF32ResW16Func = _fusedAddRmsNormModule.GetFunction("fused_add_rmsnorm_f32res_w16");
        _fusedAddRmsNormF16Func = _fusedAddRmsNormModule.GetFunction("fused_add_rmsnorm_f16");

        _perHeadRmsNormF16Func = _perHeadRmsNormModule.GetFunction("per_head_rmsnorm_f16");
        _perHeadRmsNormF32Func = _perHeadRmsNormModule.GetFunction("per_head_rmsnorm_f32");

        _ropeF16Func = _ropeModule.GetFunction("rope_f16");
        _ropeDecodeF16Func = _ropeModule.GetFunction("rope_decode_f16");
        _ropeF32Func = _ropeModule.GetFunction("rope_f32");

        _swigluF16Func = _swigluModule.GetFunction("swiglu_f16");
        _swigluF32Func = _swigluModule.GetFunction("swiglu_f32");

        _addF16Func = _addModule.GetFunction("add_f16");
        _addF32Func = _addModule.GetFunction("add_f32");
        _addF32F16Func = _addModule.GetFunction("add_f32_f16");

        _biasAddF16Func = _biasAddModule.GetFunction("bias_add_f16");
        _biasAddF32Func = _biasAddModule.GetFunction("bias_add_f32");

        _convertF16ToF32Func = _convertModule.GetFunction("convert_f16_to_f32");
        _convertF32ToF16Func = _convertModule.GetFunction("convert_f32_to_f16");

        _softmaxF16Func = _softmaxModule.GetFunction("softmax_f16");
        _softmaxF32Func = _softmaxModule.GetFunction("softmax_f32");

        _dequantQ8_0Func = _dequantModule.GetFunction("dequant_q8_0_f16");
        _dequantQ4_0Func = _dequantModule.GetFunction("dequant_q4_0_f16");
        _dequantQ5_0Func = _dequantModule.GetFunction("dequant_q5_0_f16");
        _dequantQ4_KFunc = _dequantModule.GetFunction("dequant_q4_k_f16");
        _dequantQ5_KFunc = _dequantModule.GetFunction("dequant_q5_k_f16");
        _dequantQ6_KFunc = _dequantModule.GetFunction("dequant_q6_k_f16");

        _quantizedGemvQ8_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q8_0");
        _quantizedGemvQ4_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q4_0");
        _quantizedGemvQ5_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q5_0");
        _quantizedGemvQ4_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q4_k");
        _quantizedGemvQ5_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q5_k");
        _quantizedGemvQ6_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q6_k");
        _quantizedGemvQ8_0F32OutFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q8_0_f32out");
        _quantizedGemvQ4_KF32OutFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q4_k_f32out");
        _quantizedGemvQ6_KF32OutFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q6_k_f32out");

        _attentionF16Func = _attentionModule.GetFunction("attention_f16");
        _attentionDecodeF16Func = _attentionModule.GetFunction("attention_decode_f16");
        _attentionF32Func = _attentionModule.GetFunction("attention_f32");

        _fusedGateUpSwigluModule = HipKernelLoader.LoadEmbeddedModule(arch, "fused_gate_up_swiglu");
        _fusedQkvGemvModule = HipKernelLoader.LoadEmbeddedModule(arch, "fused_qkv_gemv");
        _quantizeQ8_1Module = HipKernelLoader.LoadEmbeddedModule(arch, "quantize_q8_1");
        _mmvqModule = HipKernelLoader.LoadEmbeddedModule(arch, "mmvq");

        _fusedGateUpSwigluQ8_0Func = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_q8_0");
        _fusedGateUpSwigluQ4_0Func = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_q4_0");
        _fusedGateUpSwigluQ5_0Func = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_q5_0");
        _fusedGateUpSwigluQ4_KFunc = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_q4_k");
        _fusedGateUpSwigluQ5_KFunc = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_q5_k");
        _fusedGateUpSwigluQ6_KFunc = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_q6_k");
        _fusedGateUpSwigluF16Func = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_f16");
        _fusedGateUpSwigluMmvqQ8_0Func = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_mmvq_q8_0");
        _fusedGateUpSwigluMmvqQ4_KFunc = _fusedGateUpSwigluModule.GetFunction("fused_gate_up_swiglu_mmvq_q4_k");

        _fusedQkvGemvQ8_0Func = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_q8_0");
        _fusedQkvGemvQ4_0Func = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_q4_0");
        _fusedQkvGemvQ5_0Func = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_q5_0");
        _fusedQkvGemvQ4_KFunc = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_q4_k");
        _fusedQkvGemvQ5_KFunc = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_q5_k");
        _fusedQkvGemvQ6_KFunc = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_q6_k");
        _fusedQkvGemvF16Func = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_f16");
        _fusedQkvGemvMmvqQ8_0Func = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_mmvq_q8_0");
        _fusedQkvGemvMmvqQ4_KFunc = _fusedQkvGemvModule.GetFunction("fused_qkv_gemv_mmvq_q4_k");

        _quantizeQ8_1F16Func = _quantizeQ8_1Module.GetFunction("quantize_q8_1_f16");
        _quantizeQ8_1F32Func = _quantizeQ8_1Module.GetFunction("quantize_q8_1_f32");

        _mmvqGemvQ8_0Func = _mmvqModule.GetFunction("mmvq_gemv_q8_0");
        _mmvqGemvQ8_0Fdot2Func = _mmvqModule.GetFunction("mmvq_gemv_q8_0_fdot2");
        _mmvqGemvQ4_KFunc = _mmvqModule.GetFunction("mmvq_gemv_q4_k");
        _mmvqGemvQ6_KFunc = _mmvqModule.GetFunction("mmvq_gemv_q6_k");

        _gatedDeltaNetModule = HipKernelLoader.LoadEmbeddedModule(arch, "gated_delta_net");
        _causalConv1dSiLuFunc = _gatedDeltaNetModule.GetFunction("causal_conv1d_silu_f16");
        _l2NormAndGqaRepeatFunc = _gatedDeltaNetModule.GetFunction("l2_norm_and_gqa_repeat_f16");
        _computeSsmGatesFunc = _gatedDeltaNetModule.GetFunction("compute_ssm_gates");
        _computeSsmGatesF16Func = _gatedDeltaNetModule.GetFunction("compute_ssm_gates_f16");
        _gatedDeltaNetRecurrentFunc = _gatedDeltaNetModule.GetFunction("gated_delta_net_recurrent_f16");
        _gatedRmsNormSiLuFunc = _gatedDeltaNetModule.GetFunction("gated_rmsnorm_silu_f16");
        _deinterleaveQGateFunc = _gatedDeltaNetModule.GetFunction("deinterleave_q_gate_f16");
        _mulSigmoidGateFunc = _gatedDeltaNetModule.GetFunction("mul_sigmoid_gate_f16");
    }

    // ─────────────────────── Embedding ───────────────────────

    /// <summary>Embedding lookup with FP32 table and FP32 output for the residual stream.</summary>
    public void LaunchEmbeddingF32(nint embedTable, nint tokenIds, nint output,
                                   int seqLen, int hiddenSize, nint stream)
    {
        nint eArg = embedTable, tArg = tokenIds, oArg = output;
        int sArg = seqLen, hArg = hiddenSize;
        void** args = stackalloc void*[] { &eArg, &tArg, &oArg, &sArg, &hArg };
        HipApi.hipModuleLaunchKernel(_embeddingF32F32OutFunc,
            (uint)seqLen, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Embedding lookup with FP16 table and FP32 output for the residual stream.</summary>
    public void LaunchEmbeddingF16(nint embedTable, nint tokenIds, nint output,
                                   int seqLen, int hiddenSize, nint stream)
    {
        nint eArg = embedTable, tArg = tokenIds, oArg = output;
        int sArg = seqLen, hArg = hiddenSize;
        void** args = stackalloc void*[] { &eArg, &tArg, &oArg, &sArg, &hArg };
        HipApi.hipModuleLaunchKernel(_embeddingF16F32OutFunc,
            (uint)seqLen, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Embedding lookup with Q8_0 table and FP32 output for the residual stream.</summary>
    public void LaunchEmbeddingQ8_0(nint embedTable, nint tokenIds, nint output,
                                     int seqLen, int hiddenSize, nint stream)
    {
        nint eArg = embedTable, tArg = tokenIds, oArg = output;
        int sArg = seqLen, hArg = hiddenSize;
        void** args = stackalloc void*[] { &eArg, &tArg, &oArg, &sArg, &hArg };
        HipApi.hipModuleLaunchKernel(_embeddingQ8_0F32OutFunc,
            (uint)seqLen, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Embedding lookup for decode (scalar token_id) with FP32 table and FP32 output.</summary>
    public void LaunchEmbeddingDecodeF32(nint embedTable, int tokenId, nint output,
                                         int hiddenSize, nint stream)
    {
        nint eArg = embedTable, oArg = output;
        int tArg = tokenId, hArg = hiddenSize;
        void** args = stackalloc void*[] { &eArg, &tArg, &oArg, &hArg };
        HipApi.hipModuleLaunchKernel(_embeddingF32DecodeF32OutFunc,
            1, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Embedding lookup for decode (scalar token_id) with FP16 table and FP32 output.</summary>
    public void LaunchEmbeddingDecodeF16(nint embedTable, int tokenId, nint output,
                                         int hiddenSize, nint stream)
    {
        nint eArg = embedTable, oArg = output;
        int tArg = tokenId, hArg = hiddenSize;
        void** args = stackalloc void*[] { &eArg, &tArg, &oArg, &hArg };
        HipApi.hipModuleLaunchKernel(_embeddingF16DecodeF32OutFunc,
            1, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Embedding lookup for decode (scalar token_id) with Q8_0 table and FP32 output.</summary>
    public void LaunchEmbeddingDecodeQ8_0(nint embedTable, int tokenId, nint output,
                                          int hiddenSize, nint stream)
    {
        nint eArg = embedTable, oArg = output;
        int tArg = tokenId, hArg = hiddenSize;
        void** args = stackalloc void*[] { &eArg, &tArg, &oArg, &hArg };
        HipApi.hipModuleLaunchKernel(_embeddingQ8_0DecodeF32OutFunc,
            1, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── RMSNorm ───────────────────────

    /// <summary>RMS normalization with FP32 input, FP32 weight, and FP16 output.</summary>
    public void LaunchRmsNormF32InF16Out(nint input, nint weight, nint output,
                                         int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_rmsnormF32InF16OutFunc,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>RMS normalization with FP32 input, FP16 weight, and FP16 output.</summary>
    public void LaunchRmsNormF32InF16OutW16(nint input, nint weight, nint output,
                                            int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_rmsnormF32InF16OutW16Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>RMS normalization with FP16 input, FP16 weight, and FP16 output.</summary>
    public void LaunchRmsNormF16(nint input, nint weight, nint output,
                                 int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_rmsnormF16Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>RMS normalization with FP32 input, FP32 weight, and FP32 output.</summary>
    public void LaunchRmsNormF32(nint input, nint weight, nint output,
                                 int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_rmsnormF32Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Fused Add + RMSNorm ───────────────────────

    /// <summary>Fused residual add and RMS normalization maintaining FP32 residual stream (FP32 weight).</summary>
    public void LaunchFusedAddRmsNormF32Res(nint residual, nint x, nint weight, nint output,
                                            int hiddenSize, float eps, int rows, nint stream)
    {
        nint rArg = residual, xArg = x, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &rArg, &xArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_fusedAddRmsNormF32ResFunc,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Fused residual add and RMS normalization maintaining FP32 residual stream (FP16 weight).</summary>
    public void LaunchFusedAddRmsNormF32ResW16(nint residual, nint x, nint weight, nint output,
                                               int hiddenSize, float eps, int rows, nint stream)
    {
        nint rArg = residual, xArg = x, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &rArg, &xArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_fusedAddRmsNormF32ResW16Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Fused residual add and RMS normalization with FP16 residual stream.</summary>
    public void LaunchFusedAddRmsNormF16(nint residual, nint x, nint weight, nint output,
                                         int hiddenSize, float eps, int rows, nint stream)
    {
        nint rArg = residual, xArg = x, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;
        void** args = stackalloc void*[] { &rArg, &xArg, &wArg, &oArg, &nArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_fusedAddRmsNormF16Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Per-Head RMSNorm ───────────────────────

    /// <summary>Per-head RMS normalization (QK-norm) in FP16.</summary>
    public void LaunchPerHeadRmsNormF16(nint qk, nint weight, float eps,
                                        int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qArg = qk, wArg = weight;
        float epsArg = eps;
        int hArg = numHeads, dArg = headDim, sArg = seqLen;
        void** args = stackalloc void*[] { &qArg, &wArg, &epsArg, &hArg, &dArg, &sArg };
        uint blocks = (uint)(seqLen * numHeads);
        HipApi.hipModuleLaunchKernel(_perHeadRmsNormF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Per-head RMS normalization (QK-norm) in FP32.</summary>
    public void LaunchPerHeadRmsNormF32(nint qk, nint weight, float eps,
                                        int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qArg = qk, wArg = weight;
        float epsArg = eps;
        int hArg = numHeads, dArg = headDim, sArg = seqLen;
        void** args = stackalloc void*[] { &qArg, &wArg, &epsArg, &hArg, &dArg, &sArg };
        uint blocks = (uint)(seqLen * numHeads);
        HipApi.hipModuleLaunchKernel(_perHeadRmsNormF32Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── RoPE ───────────────────────

    /// <summary>Rotary position embedding (RoPE) in FP16.</summary>
    public void LaunchRoPEF16(nint q, nint k, nint positions, int seqLen,
                              int numHeads, int numKvHeads, int headDim, int ropeDim,
                              float theta, int ropeType, nint stream)
    {
        nint qArg = q, kArg = k, pArg = positions;
        int sArg = seqLen, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim, rdArg = ropeDim;
        float thArg = theta;
        int rtArg = ropeType;
        void** args = stackalloc void*[]
        {
            &qArg, &kArg, &pArg, &sArg, &nhArg, &nkvArg, &hdArg, &rdArg, &thArg, &rtArg
        };

        int maxHeads = Math.Max(numHeads, numKvHeads);
        int halfRope = ropeDim / 2;
        int totalPairs = seqLen * maxHeads * halfRope;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);

        HipApi.hipModuleLaunchKernel(_ropeF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Rotary position embedding (RoPE) in FP16 for decode (scalar pos).</summary>
    public void LaunchRoPEDecodeF16(nint q, nint k, int position,
                                    int numHeads, int numKvHeads, int headDim, int ropeDim,
                                    float theta, int ropeType, nint stream)
    {
        nint qArg = q, kArg = k;
        int pArg = position, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim, rdArg = ropeDim;
        float thArg = theta;
        int rtArg = ropeType;
        void** args = stackalloc void*[]
        {
            &qArg, &kArg, &pArg, &nhArg, &nkvArg, &hdArg, &rdArg, &thArg, &rtArg
        };

        int maxHeads = Math.Max(numHeads, numKvHeads);
        int halfRope = ropeDim / 2;
        int totalPairs = maxHeads * halfRope;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);

        HipApi.hipModuleLaunchKernel(_ropeDecodeF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Rotary position embedding (RoPE) in FP32.</summary>
    public void LaunchRoPEF32(nint q, nint k, nint positions, int seqLen,
                              int numHeads, int numKvHeads, int headDim, int ropeDim,
                              float theta, int ropeType, nint stream)
    {
        nint qArg = q, kArg = k, pArg = positions;
        int sArg = seqLen, nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim, rdArg = ropeDim;
        float thArg = theta;
        int rtArg = ropeType;
        void** args = stackalloc void*[]
        {
            &qArg, &kArg, &pArg, &sArg, &nhArg, &nkvArg, &hdArg, &rdArg, &thArg, &rtArg
        };

        int maxHeads = Math.Max(numHeads, numKvHeads);
        int halfRope = ropeDim / 2;
        int totalPairs = seqLen * maxHeads * halfRope;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);

        HipApi.hipModuleLaunchKernel(_ropeF32Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── SwiGLU ───────────────────────

    /// <summary>SwiGLU activation in FP16.</summary>
    public void LaunchSwiGLUF16(nint gate, nint up, nint output,
                                int intermediateSize, int seqLen, nint stream)
    {
        nint gArg = gate, uArg = up, oArg = output;
        int nArg = intermediateSize, sArg = seqLen;
        void** args = stackalloc void*[] { &gArg, &uArg, &oArg, &nArg, &sArg };

        int totalPairs = (intermediateSize * seqLen + 1) / 2;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_swigluF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>SwiGLU activation in FP32.</summary>
    public void LaunchSwiGLUF32(nint gate, nint up, nint output,
                                int intermediateSize, int seqLen, nint stream)
    {
        nint gArg = gate, uArg = up, oArg = output;
        int nArg = intermediateSize, sArg = seqLen;
        void** args = stackalloc void*[] { &gArg, &uArg, &oArg, &nArg, &sArg };

        int total = intermediateSize * seqLen;
        uint blocks = (uint)((total + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_swigluF32Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Add ───────────────────────

    /// <summary>Element-wise addition in FP16.</summary>
    public void LaunchAddF16(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, oArg = output;
        int nArg = n;
        void** args = stackalloc void*[] { &aArg, &bArg, &oArg, &nArg };

        int totalPairs = (n + 1) / 2;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_addF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Element-wise addition in FP32.</summary>
    public void LaunchAddF32(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, oArg = output;
        int nArg = n;
        void** args = stackalloc void*[] { &aArg, &bArg, &oArg, &nArg };

        uint blocks = (uint)((n + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_addF32Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Element-wise addition adding FP16 into FP32 residual output.</summary>
    public void LaunchAddF32F16(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, oArg = output;
        int nArg = n;
        void** args = stackalloc void*[] { &aArg, &bArg, &oArg, &nArg };

        uint blocks = (uint)((n + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_addF32F16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Bias Add ───────────────────────

    /// <summary>Bias addition in FP16.</summary>
    public void LaunchBiasAddF16(nint output, nint bias, int dim, int seqLen, nint stream)
    {
        nint oArg = output, bArg = bias;
        int dArg = dim, sArg = seqLen;
        void** args = stackalloc void*[] { &oArg, &bArg, &dArg, &sArg };

        int totalPairs = (dim * seqLen + 1) / 2;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_biasAddF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Bias addition in FP32.</summary>
    public void LaunchBiasAddF32(nint output, nint bias, int dim, int seqLen, nint stream)
    {
        nint oArg = output, bArg = bias;
        int dArg = dim, sArg = seqLen;
        void** args = stackalloc void*[] { &oArg, &bArg, &dArg, &sArg };

        int total = dim * seqLen;
        uint blocks = (uint)((total + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_biasAddF32Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Type Convert ───────────────────────

    /// <summary>Converts FP16 array to FP32 array.</summary>
    public void LaunchConvertF16ToF32(nint src, nint dst, int n, nint stream)
    {
        nint sArg = src, dArg = dst;
        int nArg = n;
        void** args = stackalloc void*[] { &sArg, &dArg, &nArg };

        int totalPairs = (n + 1) / 2;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_convertF16ToF32Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Converts FP32 array to FP16 array.</summary>
    public void LaunchConvertF32ToF16(nint src, nint dst, int n, nint stream)
    {
        nint sArg = src, dArg = dst;
        int nArg = n;
        void** args = stackalloc void*[] { &sArg, &dArg, &nArg };

        int totalPairs = (n + 1) / 2;
        uint blocks = (uint)((totalPairs + BlockSize - 1) / BlockSize);
        if (blocks == 0) blocks = 1;

        HipApi.hipModuleLaunchKernel(_convertF32ToF16Func,
            blocks, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Softmax ───────────────────────

    /// <summary>Softmax in FP16.</summary>
    public void LaunchSoftmaxF16(nint input, nint output, int rows, int cols, nint stream)
    {
        nint iArg = input, oArg = output;
        int rArg = rows, cArg = cols;
        void** args = stackalloc void*[] { &iArg, &oArg, &rArg, &cArg };

        HipApi.hipModuleLaunchKernel(_softmaxF16Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Softmax in FP32.</summary>
    public void LaunchSoftmaxF32(nint input, nint output, int rows, int cols, nint stream)
    {
        nint iArg = input, oArg = output;
        int rArg = rows, cArg = cols;
        void** args = stackalloc void*[] { &iArg, &oArg, &rArg, &cArg };

        HipApi.hipModuleLaunchKernel(_softmaxF32Func,
            (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Dequant ───────────────────────

    /// <summary>Dequantizes weight buffer to FP16 on device.</summary>
    public void LaunchDequantToF16(nint src, QuantizationType qt, nint dst, int totalElements, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _dequantQ8_0Func,
            QuantizationType.Q4_0 => _dequantQ4_0Func,
            QuantizationType.Q5_0 => _dequantQ5_0Func,
            QuantizationType.Q4_K => _dequantQ4_KFunc,
            QuantizationType.Q5_K => _dequantQ5_KFunc,
            QuantizationType.Q6_K => _dequantQ6_KFunc,
            _ => throw new NotSupportedException($"GPU dequantization not supported for {qt}")
        };

        bool isKQuant = qt is QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;
        int totalUnits = isKQuant ? totalElements / 256 : totalElements / 32;
        int gridDim = isKQuant ? Math.Clamp(totalUnits, 1, 256) : Math.Clamp((totalUnits + 7) / 8, 1, 256);

        nint sArg = src, dArg = dst;
        int uArg = totalUnits;
        void** args = stackalloc void*[] { &sArg, &dArg, &uArg };

        HipApi.hipModuleLaunchKernel(func, (uint)gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Quantized GEMV ───────────────────────

    /// <summary>Returns true if the quantization type has a dedicated quantized GEMV kernel.</summary>
    public static bool HasQuantizedGemv(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_0 or QuantizationType.Q5_0
           or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K;

    /// <summary>Runs quantized GEMV: y[n] = W[n, k] @ x[k] with 1 wave per row.</summary>
    public void LaunchQuantizedGemv(nint weight, QuantizationType qt, nint x, nint y,
                                    int n, int k, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _quantizedGemvQ8_0Func,
            QuantizationType.Q4_0 => _quantizedGemvQ4_0Func,
            QuantizationType.Q5_0 => _quantizedGemvQ5_0Func,
            QuantizationType.Q4_K => _quantizedGemvQ4_KFunc,
            QuantizationType.Q5_K => _quantizedGemvQ5_KFunc,
            QuantizationType.Q6_K => _quantizedGemvQ6_KFunc,
            _ => throw new NotSupportedException($"GPU quantized GEMV not supported for {qt}")
        };

        nint wArg = weight, xArg = x, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };

        // 8 waves of 32 threads per block -> 8 rows per block
        uint gridDim = (uint)((n + 7) / 8);
        if (gridDim == 0) gridDim = 1;

        HipApi.hipModuleLaunchKernel(func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Returns true if the quantization type has a dedicated quantized GEMV kernel with FP32 output.</summary>
    public static bool HasQuantizedGemvF32Out(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K or QuantizationType.Q6_K;

    /// <summary>Runs quantized GEMV: y[n] = W[n, k] @ x[k] with FP32 output y.</summary>
    public void LaunchQuantizedGemvF32Out(nint weight, QuantizationType qt, nint x, nint y,
                                          int n, int k, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _quantizedGemvQ8_0F32OutFunc,
            QuantizationType.Q4_K => _quantizedGemvQ4_KF32OutFunc,
            QuantizationType.Q6_K => _quantizedGemvQ6_KF32OutFunc,
            _ => throw new NotSupportedException($"GPU quantized GEMV with FP32 out not supported for {qt}")
        };

        nint wArg = weight, xArg = x, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };

        uint gridDim = (uint)((n + 7) / 8);
        if (gridDim == 0) gridDim = 1;

        HipApi.hipModuleLaunchKernel(func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Fused Gate + Up + SwiGLU ───────────────────────

    /// <summary>Returns true if the quantization type is supported by the fused Gate+Up+SwiGLU kernel.</summary>
    public static bool HasFusedGateUpSwiglu(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_0 or QuantizationType.Q5_0
           or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K or QuantizationType.F16;

    /// <summary>Runs fused Gate + Up GEMV with SwiGLU activation for decode.</summary>
    public void LaunchFusedGateUpSwiglu(
        nint gateWeight, nint upWeight, QuantizationType qt,
        nint gateBias, nint upBias,
        nint x, nint output,
        int intermediateSize, int k, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _fusedGateUpSwigluQ8_0Func,
            QuantizationType.Q4_0 => _fusedGateUpSwigluQ4_0Func,
            QuantizationType.Q5_0 => _fusedGateUpSwigluQ5_0Func,
            QuantizationType.Q4_K => _fusedGateUpSwigluQ4_KFunc,
            QuantizationType.Q5_K => _fusedGateUpSwigluQ5_KFunc,
            QuantizationType.Q6_K => _fusedGateUpSwigluQ6_KFunc,
            QuantizationType.F16 => _fusedGateUpSwigluF16Func,
            _ => throw new NotSupportedException($"Fused Gate+Up+SwiGLU not supported for {qt}")
        };

        nint gwArg = gateWeight, uwArg = upWeight;
        nint gbArg = gateBias, ubArg = upBias;
        nint xArg = x, outArg = output;
        int intArg = intermediateSize, kArg = k;

        void** args = stackalloc void*[] {
            &gwArg, &uwArg, &gbArg, &ubArg,
            &xArg, &outArg, &intArg, &kArg
        };

        uint gridDim = (uint)((intermediateSize + 7) / 8);
        if (gridDim == 0) gridDim = 1;

        HipApi.hipModuleLaunchKernel(func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Returns true if the quantization type is supported by the vectorized MMVQ fused Gate+Up+SwiGLU kernel.</summary>
    public static bool HasFusedGateUpSwigluMmvq(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K;

    /// <summary>Runs vectorized MMVQ fused Gate + Up GEMV with SwiGLU activation for decode (H3.2).</summary>
    public void LaunchFusedGateUpSwigluMmvq(
        nint gateWeight, nint upWeight, QuantizationType qt,
        nint gateBias, nint upBias,
        nint xQ8_1, nint output,
        int intermediateSize, int k, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _fusedGateUpSwigluMmvqQ8_0Func,
            QuantizationType.Q4_K => _fusedGateUpSwigluMmvqQ4_KFunc,
            _ => throw new NotSupportedException($"MMVQ Fused Gate+Up+SwiGLU not supported for {qt}")
        };

        nint gwArg = gateWeight, uwArg = upWeight;
        nint gbArg = gateBias, ubArg = upBias;
        nint xArg = xQ8_1, outArg = output;
        int intArg = intermediateSize, kArg = k;

        void** args = stackalloc void*[] {
            &gwArg, &uwArg, &gbArg, &ubArg,
            &xArg, &outArg, &intArg, &kArg
        };

        uint gridDim = (uint)((intermediateSize + 7) / 8);
        if (gridDim == 0) gridDim = 1;

        HipApi.hipModuleLaunchKernel(func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Fused QKV GEMV ───────────────────────

    /// <summary>Returns true if the quantization type is supported by the fused QKV GEMV kernel.</summary>
    public static bool HasFusedQkvGemv(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_0 or QuantizationType.Q5_0
           or QuantizationType.Q4_K or QuantizationType.Q5_K or QuantizationType.Q6_K or QuantizationType.F16;

    /// <summary>Runs fused Q, K, V GEMV projections in a single grid launch for decode.</summary>
    public void LaunchFusedQkvGemv(
        nint qWeight, nint kWeight, nint vWeight, QuantizationType qt,
        nint qBias, nint kBias, nint vBias,
        nint x, nint qOut, nint kOut, nint vOut,
        int dimQ, int dimK, int dimV, int k, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _fusedQkvGemvQ8_0Func,
            QuantizationType.Q4_0 => _fusedQkvGemvQ4_0Func,
            QuantizationType.Q5_0 => _fusedQkvGemvQ5_0Func,
            QuantizationType.Q4_K => _fusedQkvGemvQ4_KFunc,
            QuantizationType.Q5_K => _fusedQkvGemvQ5_KFunc,
            QuantizationType.Q6_K => _fusedQkvGemvQ6_KFunc,
            QuantizationType.F16 => _fusedQkvGemvF16Func,
            _ => throw new NotSupportedException($"Fused QKV GEMV not supported for {qt}")
        };

        nint qwArg = qWeight, kwArg = kWeight, vwArg = vWeight;
        nint qbArg = qBias, kbArg = kBias, vbArg = vBias;
        nint xArg = x, qoArg = qOut, koArg = kOut, voArg = vOut;
        int dqArg = dimQ, dkArg = dimK, dvArg = dimV, kArg = k;

        void** args = stackalloc void*[] {
            &qwArg, &kwArg, &vwArg,
            &qbArg, &kbArg, &vbArg,
            &xArg, &qoArg, &koArg, &voArg,
            &dqArg, &dkArg, &dvArg, &kArg
        };

        int totalRows = dimQ + dimK + dimV;
        uint gridDim = (uint)((totalRows + 7) / 8);
        if (gridDim == 0) gridDim = 1;

        HipApi.hipModuleLaunchKernel(func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Returns true if the quantization type is supported by the vectorized MMVQ fused QKV GEMV kernel.</summary>
    public static bool HasFusedQkvGemvMmvq(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K;

    /// <summary>Runs vectorized MMVQ fused Q, K, V GEMV projections in a single grid launch for decode (H3.2).</summary>
    public void LaunchFusedQkvGemvMmvq(
        nint qWeight, nint kWeight, nint vWeight, QuantizationType qt,
        nint qBias, nint kBias, nint vBias,
        nint xQ8_1, nint qOut, nint kOut, nint vOut,
        int dimQ, int dimK, int dimV, int k, nint stream)
    {
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _fusedQkvGemvMmvqQ8_0Func,
            QuantizationType.Q4_K => _fusedQkvGemvMmvqQ4_KFunc,
            _ => throw new NotSupportedException($"MMVQ Fused QKV GEMV not supported for {qt}")
        };

        nint qwArg = qWeight, kwArg = kWeight, vwArg = vWeight;
        nint qbArg = qBias, kbArg = kBias, vbArg = vBias;
        nint xArg = xQ8_1, qoArg = qOut, koArg = kOut, voArg = vOut;
        int dqArg = dimQ, dkArg = dimK, dvArg = dimV, kArg = k;

        void** args = stackalloc void*[] {
            &qwArg, &kwArg, &vwArg,
            &qbArg, &kbArg, &vbArg,
            &xArg, &qoArg, &koArg, &voArg,
            &dqArg, &dkArg, &dvArg, &kArg
        };

        int totalRows = dimQ + dimK + dimV;
        uint gridDim = (uint)((totalRows + 7) / 8);
        if (gridDim == 0) gridDim = 1;

        HipApi.hipModuleLaunchKernel(func, gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Attention ───────────────────────

    /// <summary>Tiled scaled dot-product attention with causal mask, sliding window and GQA (FP16).</summary>
    public void LaunchAttention(nint q, nint k, nint v, nint output,
                                 int seqQ, int seqKv,
                                 int numHeads, int numKvHeads, int headDim,
                                 int positionOffset, int slidingWindow, nint stream)
    {
        if (seqQ == 1 && (headDim == 64 || headDim == 128))
        {
            nint qArg = q, kArg = k, vArg = v, outArg = output;
            int skvArg = seqKv;
            int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
            int poArg = positionOffset, swArg = slidingWindow;

            void** args = stackalloc void*[] {
                &qArg, &kArg, &vArg, &outArg,
                &skvArg, &nhArg, &nkvArg, &hdArg,
                &poArg, &swArg
            };

            HipApi.hipModuleLaunchKernel(_attentionDecodeF16Func,
                (uint)numHeads, 1, 1, BlockSize, 1, 1,
                0, stream, args, null).ThrowOnError();
            return;
        }

        nint qArg2 = q, kArg2 = k, vArg2 = v, outArg2 = output;
        int sqArg2 = seqQ, skvArg2 = seqKv;
        int nhArg2 = numHeads, nkvArg2 = numKvHeads, hdArg2 = headDim;
        int poArg2 = positionOffset, swArg2 = slidingWindow;

        void** args2 = stackalloc void*[] {
            &qArg2, &kArg2, &vArg2, &outArg2,
            &sqArg2, &skvArg2, &nhArg2, &nkvArg2, &hdArg2,
            &poArg2, &swArg2
        };

        int numBlocks = seqQ * numHeads;
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        HipApi.hipModuleLaunchKernel(_attentionF16Func,
            (uint)numBlocks, 1, 1, BlockSize, 1, 1,
            sharedBytes, stream, args2, null).ThrowOnError();
    }

    /// <summary>Tiled scaled dot-product attention with causal mask, sliding window and GQA (FP32).</summary>
    public void LaunchAttentionF32(nint q, nint k, nint v, nint output,
                                    int seqQ, int seqKv,
                                    int numHeads, int numKvHeads, int headDim,
                                    int positionOffset, int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, outArg = output;
        int sqArg = seqQ, skvArg = seqKv;
        int nhArg = numHeads, nkvArg = numKvHeads, hdArg = headDim;
        int poArg = positionOffset, swArg = slidingWindow;

        void** args = stackalloc void*[] {
            &qArg, &kArg, &vArg, &outArg,
            &sqArg, &skvArg, &nhArg, &nkvArg, &hdArg,
            &poArg, &swArg
        };

        int numBlocks = seqQ * numHeads;
        const int TileKv = 256;
        uint sharedBytes = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));

        HipApi.hipModuleLaunchKernel(_attentionF32Func,
            (uint)numBlocks, 1, 1, BlockSize, 1, 1,
            sharedBytes, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Quantize Q8_1 ───────────────────────

    /// <summary>Quantizes FP16 activations on device into block_q8_1 format (H3.2).</summary>
    public void LaunchQuantizeQ8_1F16(nint x, nint y, int k, nint stream)
    {
        nint xArg = x, yArg = y;
        int kArg = k;
        void** args = stackalloc void*[] { &xArg, &yArg, &kArg };
        int numBlocks = k / 32;
        uint gridDim = (uint)((numBlocks + 7) / 8);
        if (gridDim == 0) gridDim = 1;
        HipApi.hipModuleLaunchKernel(_quantizeQ8_1F16Func,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Quantizes FP32 activations on device into block_q8_1 format (H3.2).</summary>
    public void LaunchQuantizeQ8_1F32(nint x, nint y, int k, nint stream)
    {
        nint xArg = x, yArg = y;
        int kArg = k;
        void** args = stackalloc void*[] { &xArg, &yArg, &kArg };
        int numBlocks = k / 32;
        uint gridDim = (uint)((numBlocks + 7) / 8);
        if (gridDim == 0) gridDim = 1;
        HipApi.hipModuleLaunchKernel(_quantizeQ8_1F32Func,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── MMVQ GEMV ───────────────────────

    /// <summary>Returns true if the quantization type has a dedicated mmvq GEMV implementation.</summary>
    public static bool HasMmvqGemv(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K;

    /// <summary>Executes vectorized mmvq GEMV for Q8_0 weights with Q8_1 activations (H3.2).</summary>
    public void LaunchMmvqGemvQ8_0(nint weight, nint xQ8_1, nint y, int n, int k, nint stream)
    {
        nint wArg = weight, xArg = xQ8_1, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        uint gridDim = (uint)((n + 7) / 8);
        if (gridDim == 0) gridDim = 1;
        HipApi.hipModuleLaunchKernel(_mmvqGemvQ8_0Func,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Executes vectorized mmvq GEMV for Q8_0 weights with FP16 activations via fdot2 (H3.2).</summary>
    public void LaunchMmvqGemvQ8_0Fdot2(nint weight, nint x, nint y, int n, int k, nint stream)
    {
        nint wArg = weight, xArg = x, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        uint gridDim = (uint)((n + 7) / 8);
        if (gridDim == 0) gridDim = 1;
        HipApi.hipModuleLaunchKernel(_mmvqGemvQ8_0Fdot2Func,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Executes vectorized mmvq GEMV for Q4_K weights with Q8_1 activations (H3.2).</summary>
    public void LaunchMmvqGemvQ4_K(nint weight, nint xQ8_1, nint y, int n, int k, nint stream)
    {
        nint wArg = weight, xArg = xQ8_1, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        uint gridDim = (uint)((n + 7) / 8);
        if (gridDim == 0) gridDim = 1;
        HipApi.hipModuleLaunchKernel(_mmvqGemvQ4_KFunc,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Executes vectorized mmvq GEMV for Q6_K weights with Q8_1 activations (H3.2).</summary>
    public void LaunchMmvqGemvQ6_K(nint weight, nint xQ8_1, nint y, int n, int k, nint stream)
    {
        nint wArg = weight, xArg = xQ8_1, yArg = y;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        uint gridDim = (uint)((n + 7) / 8);
        if (gridDim == 0) gridDim = 1;
        HipApi.hipModuleLaunchKernel(_mmvqGemvQ6_KFunc,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    // ─────────────────────── Gated DeltaNet & Attention Gating ───────────────────────

    /// <summary>Depthwise 1D causal conv (kernel_size=4) + state update + SiLU activation.</summary>
    public void LaunchCausalConv1dSiLu(nint input, nint weights, nint state, nint output, int channels, nint stream)
    {
        nint inArg = input, wArg = weights, sArg = state, outArg = output;
        int cArg = channels;
        void** args = stackalloc void*[] { &inArg, &wArg, &sArg, &outArg, &cArg };
        uint gridDim = (uint)((channels + 255) / 256);
        HipApi.hipModuleLaunchKernel(_causalConv1dSiLuFunc,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Per-head L2 norm on Q and K, scale Q by 1/sqrt(headDim), and GQA repeat from numKHeads to numVHeads.</summary>
    public void LaunchL2NormAndGqaRepeat(nint qIn, nint kIn, nint qOut, nint kOut,
                                         int numKHeads, int numVHeads, int headDim, float eps, nint stream)
    {
        nint qInArg = qIn, kInArg = kIn, qOutArg = qOut, kOutArg = kOut;
        int nkArg = numKHeads, nvArg = numVHeads, hdArg = headDim;
        float epsArg = eps;
        void** args = stackalloc void*[] { &qInArg, &kInArg, &qOutArg, &kOutArg, &nkArg, &nvArg, &hdArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_l2NormAndGqaRepeatFunc,
            (uint)numKHeads, 1, 1, 128, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Computes SSM recurrence gates beta = sigmoid(beta_raw) and g = exp(ssm_a * softplus(alpha_raw + dt_bias)).</summary>
    public void LaunchComputeSsmGates(nint betaRaw, nint alphaRaw, nint ssmA, nint ssmDtBias,
                                      nint betaOut, nint gOut, int numVHeads, nint stream)
    {
        nint bRawArg = betaRaw, aRawArg = alphaRaw, aArg = ssmA, dtArg = ssmDtBias;
        nint bOutArg = betaOut, gOutArg = gOut;
        int nvArg = numVHeads;
        void** args = stackalloc void*[] { &bRawArg, &aRawArg, &aArg, &dtArg, &bOutArg, &gOutArg, &nvArg };
        HipApi.hipModuleLaunchKernel(_computeSsmGatesFunc,
            1, 1, 1, (uint)numVHeads, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Computes SSM recurrence gates from FP16 inputs: beta = sigmoid(beta_raw) and g = exp(ssm_a * softplus(alpha_raw + dt_bias)).</summary>
    public void LaunchComputeSsmGatesF16(nint betaRaw, nint alphaRaw, nint ssmA, nint ssmDtBias,
                                         nint betaOut, nint gOut, int numVHeads, nint stream)
    {
        nint bRawArg = betaRaw, aRawArg = alphaRaw, aArg = ssmA, dtArg = ssmDtBias;
        nint bOutArg = betaOut, gOutArg = gOut;
        int nvArg = numVHeads;
        void** args = stackalloc void*[] { &bRawArg, &aRawArg, &aArg, &dtArg, &bOutArg, &gOutArg, &nvArg };
        HipApi.hipModuleLaunchKernel(_computeSsmGatesF16Func,
            1, 1, 1, (uint)numVHeads, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Recurrent Delta Rule single token step with 128 threads per head owning column j of state.</summary>
    public void LaunchGatedDeltaNetRecurrent(nint q, nint k, nint v, nint beta, nint g,
                                            nint state, nint output, int numVHeads, int headKDim, int headVDim, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, bArg = beta, gArg = g, sArg = state, outArg = output;
        int nvArg = numVHeads, hkArg = headKDim, hvArg = headVDim;
        void** args = stackalloc void*[] { &qArg, &kArg, &vArg, &bArg, &gArg, &sArg, &outArg, &nvArg, &hkArg, &hvArg };
        HipApi.hipModuleLaunchKernel(_gatedDeltaNetRecurrentFunc,
            (uint)numVHeads, 1, 1, 128, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Gated RMSNorm + SiLU across head_v_dim: y = RMSNorm(o, norm_weight) * SiLU(z).</summary>
    public void LaunchGatedRmsNormSiLu(nint o, nint z, nint normWeight, nint output,
                                       int numVHeads, int headVDim, float eps, nint stream)
    {
        nint oArg = o, zArg = z, wArg = normWeight, outArg = output;
        int nvArg = numVHeads, hvArg = headVDim;
        float epsArg = eps;
        void** args = stackalloc void*[] { &oArg, &zArg, &wArg, &outArg, &nvArg, &hvArg, &epsArg };
        HipApi.hipModuleLaunchKernel(_gatedRmsNormSiLuFunc,
            (uint)numVHeads, 1, 1, 128, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>De-interleaves 2x Q projection output into Q and Gate.</summary>
    public void LaunchDeinterleaveQGate(nint qg, nint q, nint gate, int numHeads, int headDim, nint stream)
    {
        nint qgArg = qg, qArg = q, gArg = gate;
        int nhArg = numHeads, hdArg = headDim;
        void** args = stackalloc void*[] { &qgArg, &qArg, &gArg, &nhArg, &hdArg };
        HipApi.hipModuleLaunchKernel(_deinterleaveQGateFunc,
            (uint)numHeads, 1, 1, (uint)headDim, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Multiplies attention output by sigmoid(gate) element-wise.</summary>
    public void LaunchMulSigmoidGate(nint attnOut, nint gate, int totalElements, nint stream)
    {
        nint aArg = attnOut, gArg = gate;
        int totArg = totalElements;
        void** args = stackalloc void*[] { &aArg, &gArg, &totArg };
        uint gridDim = (uint)((totalElements + 255) / 256);
        HipApi.hipModuleLaunchKernel(_mulSigmoidGateFunc,
            gridDim, 1, 1, BlockSize, 1, 1, 0, stream, args, null).ThrowOnError();
    }

    /// <summary>Disposes all loaded HIP kernel modules.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        _embeddingModule.Dispose();
        _rmsnormModule.Dispose();
        _fusedAddRmsNormModule.Dispose();
        _perHeadRmsNormModule.Dispose();
        _ropeModule.Dispose();
        _swigluModule.Dispose();
        _addModule.Dispose();
        _biasAddModule.Dispose();
        _convertModule.Dispose();
        _softmaxModule.Dispose();
        _dequantModule.Dispose();
        _quantizedGemvModule.Dispose();
        _attentionModule.Dispose();
        _fusedGateUpSwigluModule.Dispose();
        _fusedQkvGemvModule.Dispose();
        _quantizeQ8_1Module.Dispose();
        _mmvqModule.Dispose();
        _gatedDeltaNetModule.Dispose();
    }
}
