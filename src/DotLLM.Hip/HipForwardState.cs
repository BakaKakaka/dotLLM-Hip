using System.Numerics;
using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Pre-allocated GPU scratch buffers for the HIP forward pass.
/// Residual stream is maintained in FP32 throughout (D5), while intermediate activations
/// are FP16 for optimal memory bandwidth. Includes pinned host memory buffers for asynchronous
/// H2D and D2H transfers.
/// </summary>
internal sealed class HipForwardState : IDisposable
{
    private const int Fp16Bytes = 2;

    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;

    private int _currentSeqLen;

    /// <summary>Total bytes currently allocated on GPU.</summary>
    public long AllocatedBytes { get; private set; }

    // Activation buffers on device
    public nint HiddenState;    // [seqLen, hiddenSize] FP16
    public nint Residual;       // [seqLen, hiddenSize] FP32 (D5 mandate)
    public nint NormOutput;     // [seqLen, hiddenSize] FP16
    public nint Q;              // [seqLen, numHeads * headDim] FP16
    public nint K;              // [seqLen, numKvHeads * headDim] FP16
    public nint V;              // [seqLen, numKvHeads * headDim] FP16
    public nint AttnOutput;     // [seqLen, numHeads * headDim] FP16
    public nint FfnGate;        // [seqLen, intermediateSize] FP16
    public nint FfnUp;          // [seqLen, intermediateSize] FP16
    public nint SiluOutput;     // [seqLen, intermediateSize] FP16

    // Logits on device
    public nint LogitsF16;      // [vocabSize] FP16
    public nint LogitsF32;      // [vocabSize] FP32

    // On-the-fly dequantization scratch (FP16)
    public nint DequantScratch;

    // On-the-fly Q8_1 activation quantization scratch (H3.2)
    public nint Q8_1Scratch;

    // Small device buffers for token IDs and positions
    public nint TokenIdsDevice; // [maxSeqLen] int32
    public nint PositionsDevice;// [maxSeqLen] int32

    // Pinned host buffers for async H2D/D2H transfers (H2.5)
    public HipPinnedBuffer TokenIdsHost { get; }
    public HipPinnedBuffer PositionsHost { get; }
    public HipPinnedBuffer LogitsHost { get; }

    /// <summary>Pre-allocated non-owning tensor view wrapping <see cref="LogitsHost"/> for zero-alloc return.</summary>
    public UnmanagedTensor LogitsTensor { get; }

    public HipForwardState(int hiddenSize, int numHeads, int numKvHeads, int headDim,
                           int intermediateSize, int vocabSize, int maxInitialSeqLen = 2048)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _currentSeqLen = 0;

        // Pinned host memory
        TokenIdsHost = new HipPinnedBuffer((long)maxInitialSeqLen * sizeof(int));
        PositionsHost = new HipPinnedBuffer((long)maxInitialSeqLen * sizeof(int));
        LogitsHost = new HipPinnedBuffer((long)vocabSize * sizeof(float));
        LogitsTensor = new UnmanagedTensor(new TensorShape(1, vocabSize), DType.Float32, -1, LogitsHost.Pointer, ownsMemory: false);

        // Device tokens / positions buffers
        TokenIdsDevice = AllocDevice((long)maxInitialSeqLen * sizeof(int));
        PositionsDevice = AllocDevice((long)maxInitialSeqLen * sizeof(int));

        // Fixed-size device logits
        LogitsF16 = AllocDevice((long)vocabSize * Fp16Bytes);
        LogitsF32 = AllocDevice((long)vocabSize * sizeof(float));

        // Dequant scratch buffer sized for the largest projection
        long maxProjectionElements = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocDevice(maxProjectionElements * Fp16Bytes);

        // Q8_1 activation scratch buffer sized for the largest activation vector
        long maxActivationElements = Math.Max(hiddenSize, intermediateSize);
        long maxQ8_1Bytes = ((maxActivationElements + 31) / 32) * 36;
        Q8_1Scratch = AllocDevice(maxQ8_1Bytes);

        // Initial allocation for decode (seqLen = 1)
        EnsureCapacity(1);
    }

    /// <summary>
    /// Ensures all sequence-dependent scratch buffers are large enough for <paramref name="seqLen"/> tokens.
    /// Uses power-of-2 growth to amortize reallocation cost.
    /// </summary>
    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen)
            return;

        int newCapacity = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        int half = Fp16Bytes;      // FP16 = 2 bytes
        int single = sizeof(float);// FP32 = 4 bytes

        // FP16 activations
        HiddenState = AllocDevice((long)newCapacity * _hiddenSize * half);
        NormOutput = AllocDevice((long)newCapacity * _hiddenSize * half);
        Q = AllocDevice((long)newCapacity * _numHeads * _headDim * half);
        K = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        V = AllocDevice((long)newCapacity * _numKvHeads * _headDim * half);
        AttnOutput = AllocDevice((long)newCapacity * _numHeads * _headDim * half);
        FfnGate = AllocDevice((long)newCapacity * _intermediateSize * half);
        FfnUp = AllocDevice((long)newCapacity * _intermediateSize * half);
        SiluOutput = AllocDevice((long)newCapacity * _intermediateSize * half);

        // FP32 residual stream (D5 mandate)
        Residual = AllocDevice((long)newCapacity * _hiddenSize * single);

        _currentSeqLen = newCapacity;
    }

    private nint AllocDevice(long bytes)
    {
        HipApi.hipMalloc(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private void FreeDevice(ref nint ptr, long bytes)
    {
        if (ptr != 0)
        {
            HipApi.hipFree(ptr);
            ptr = 0;
            AllocatedBytes -= bytes;
        }
    }

    private void FreeSequenceBuffers()
    {
        if (_currentSeqLen == 0) return;

        int half = Fp16Bytes;
        int single = sizeof(float);

        FreeDevice(ref HiddenState, (long)_currentSeqLen * _hiddenSize * half);
        FreeDevice(ref NormOutput, (long)_currentSeqLen * _hiddenSize * half);
        FreeDevice(ref Q, (long)_currentSeqLen * _numHeads * _headDim * half);
        FreeDevice(ref K, (long)_currentSeqLen * _numKvHeads * _headDim * half);
        FreeDevice(ref V, (long)_currentSeqLen * _numKvHeads * _headDim * half);
        FreeDevice(ref AttnOutput, (long)_currentSeqLen * _numHeads * _headDim * half);
        FreeDevice(ref FfnGate, (long)_currentSeqLen * _intermediateSize * half);
        FreeDevice(ref FfnUp, (long)_currentSeqLen * _intermediateSize * half);
        FreeDevice(ref SiluOutput, (long)_currentSeqLen * _intermediateSize * half);

        FreeDevice(ref Residual, (long)_currentSeqLen * _hiddenSize * single);

        _currentSeqLen = 0;
    }

    public void Dispose()
    {
        FreeSequenceBuffers();

        FreeDevice(ref LogitsF16, (long)_vocabSize * Fp16Bytes);
        FreeDevice(ref LogitsF32, (long)_vocabSize * sizeof(float));

        long maxProjectionElements = Math.Max(
            (long)Math.Max(_numHeads * _headDim, _numKvHeads * _headDim) * _hiddenSize,
            (long)_intermediateSize * _hiddenSize);
        FreeDevice(ref DequantScratch, maxProjectionElements * Fp16Bytes);

        long maxActivationElements = Math.Max(_hiddenSize, _intermediateSize);
        long maxQ8_1Bytes = ((maxActivationElements + 31) / 32) * 36;
        FreeDevice(ref Q8_1Scratch, maxQ8_1Bytes);

        FreeDevice(ref TokenIdsDevice, 2048 * sizeof(int));
        FreeDevice(ref PositionsDevice, 2048 * sizeof(int));

        TokenIdsHost.Dispose();
        PositionsHost.Dispose();
        LogitsHost.Dispose();
    }
}
