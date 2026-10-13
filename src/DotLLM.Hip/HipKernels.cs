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

    // Embedding functions
    private readonly nint _embeddingF32F32OutFunc;
    private readonly nint _embeddingF16F32OutFunc;
    private readonly nint _embeddingQ8_0F32OutFunc;

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

    // Attention
    private readonly nint _attentionF16Func;
    private readonly nint _attentionF32Func;

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

        _attentionF16Func = _attentionModule.GetFunction("attention_f16");
        _attentionF32Func = _attentionModule.GetFunction("attention_f32");
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

    // ─────────────────────── Attention ───────────────────────

    /// <summary>Tiled scaled dot-product attention with causal mask, sliding window and GQA (FP16).</summary>
    public void LaunchAttention(nint q, nint k, nint v, nint output,
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

        HipApi.hipModuleLaunchKernel(_attentionF16Func,
            (uint)numBlocks, 1, 1, BlockSize, 1, 1,
            sharedBytes, stream, args, null).ThrowOnError();
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
    }
}
