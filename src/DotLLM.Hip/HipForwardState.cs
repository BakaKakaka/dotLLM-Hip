using System.Numerics;
using DotLLM.Core.Models;
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

    // DeltaNet parameters
    private readonly bool _hasSsm;
    private readonly int _convDim;
    private readonly int _innerSize;
    private readonly int _numVHeads;
    private readonly int _headKDim;
    private readonly int _headVDim;
    private readonly long _totalConvStateBytes;
    private readonly long _totalSsmStateBytes;

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

    // Full Attention Gating (Qwen 3.5)
    public nint QgScratch;      // [seqLen, numHeads * headDim * 2] FP16
    public nint AttentionGate;  // [seqLen, numHeads * headDim] FP16

    // DeltaNet activations on device
    public nint QkvRaw;            // [seqLen, convDim] FP16
    public nint ConvOut;           // [seqLen, convDim] FP16
    public nint SsmGateRaw;        // [seqLen, innerSize] FP16
    public nint BetaRaw;           // [seqLen, numVHeads] FP16
    public nint AlphaRaw;          // [seqLen, numVHeads] FP16
    public nint QRepeated;         // [seqLen, numVHeads * headKDim] FP16
    public nint KRepeated;         // [seqLen, numVHeads * headKDim] FP16
    public nint Beta;              // [seqLen, numVHeads] float (FP32)
    public nint G;                 // [seqLen, numVHeads] float (FP32)
    public nint DeltaNetOut;       // [seqLen, innerSize] FP16
    public nint NormedDeltaNetOut; // [seqLen, innerSize] FP16

    // DeltaNet persistent recurrent states on device (FP32)
    public readonly nint ConvStates;
    public readonly nint SsmStates;

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
                           int intermediateSize, int vocabSize,
                           SsmConfig? ssmConfig = null, int numLayers = 0, int maxInitialSeqLen = 2048)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;
        _currentSeqLen = 0;

        _hasSsm = ssmConfig.HasValue;
        if (_hasSsm)
        {
            var ssm = ssmConfig!.Value;
            int interval = ssm.FullAttentionInterval > 0 ? ssm.FullAttentionInterval : 4;
            int numDeltaNetLayers = 0;
            for (int i = 0; i < numLayers; i++)
            {
                if ((i + 1) % interval != 0)
                    numDeltaNetLayers++;
            }

            _headKDim = ssm.StateSize > 0 ? ssm.StateSize : 128;
            _headVDim = _headKDim;
            int numKHeads = ssm.GroupCount > 0 ? ssm.GroupCount : 16;
            _numVHeads = ssm.TimeStepRank > 0 ? ssm.TimeStepRank : 32;
            _innerSize = ssm.InnerSize > 0 ? ssm.InnerSize : (_numVHeads * _headVDim);
            _convDim = numKHeads * _headKDim * 2 + _innerSize; // e.g. 16*128*2 + 4096 = 8192

            // Conv state: 3 floats per channel (kernel size 4 -> 3 historical elements)
            _totalConvStateBytes = (long)numDeltaNetLayers * _convDim * 3 * sizeof(float);
            ConvStates = AllocDevice(_totalConvStateBytes);

            // SSM recurrent state: numVHeads * headKDim * headVDim floats per layer
            _totalSsmStateBytes = (long)numDeltaNetLayers * _numVHeads * _headKDim * _headVDim * sizeof(float);
            SsmStates = AllocDevice(_totalSsmStateBytes);
        }

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
        long maxDim = Math.Max(
            Math.Max(numHeads * headDim, numKvHeads * headDim),
            intermediateSize);
        if (_hasSsm)
        {
            maxDim = Math.Max(maxDim, Math.Max(_convDim, _innerSize));
            maxDim = Math.Max(maxDim, numHeads * headDim * 2);
        }
        long maxProjectionElements = maxDim * hiddenSize;
        DequantScratch = AllocDevice(maxProjectionElements * Fp16Bytes);

        // Q8_1 activation scratch buffer sized for the largest activation vector
        long maxActivationElements = Math.Max(hiddenSize, intermediateSize);
        if (_hasSsm)
            maxActivationElements = Math.Max(maxActivationElements, Math.Max(_convDim, _innerSize));
        long maxQ8_1Bytes = ((maxActivationElements + 31) / 32) * 36;
        Q8_1Scratch = AllocDevice(maxQ8_1Bytes);

        // Initial allocation for decode (seqLen = 1)
        EnsureCapacity(1);
    }

    public nint GetConvState(int deltaNetIdx) => ConvStates + (nint)(deltaNetIdx * _convDim * 3 * sizeof(float));
    public nint GetSsmState(int deltaNetIdx) => SsmStates + (nint)(deltaNetIdx * _numVHeads * _headKDim * _headVDim * sizeof(float));

    public void ResetRecurrentState(nint stream)
    {
        if (_totalConvStateBytes > 0)
            HipApi.hipMemsetAsync(ConvStates, 0, (nuint)_totalConvStateBytes, stream).ThrowOnError();
        if (_totalSsmStateBytes > 0)
            HipApi.hipMemsetAsync(SsmStates, 0, (nuint)_totalSsmStateBytes, stream).ThrowOnError();
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

        if (_hasSsm)
        {
            QgScratch = AllocDevice((long)newCapacity * _numHeads * _headDim * 2 * half);
            AttentionGate = AllocDevice((long)newCapacity * _numHeads * _headDim * half);

            QkvRaw = AllocDevice((long)newCapacity * _convDim * half);
            ConvOut = AllocDevice((long)newCapacity * _convDim * half);
            SsmGateRaw = AllocDevice((long)newCapacity * _innerSize * half);
            BetaRaw = AllocDevice((long)newCapacity * _numVHeads * half);
            AlphaRaw = AllocDevice((long)newCapacity * _numVHeads * half);
            QRepeated = AllocDevice((long)newCapacity * _numVHeads * _headKDim * half);
            KRepeated = AllocDevice((long)newCapacity * _numVHeads * _headKDim * half);
            Beta = AllocDevice((long)newCapacity * _numVHeads * single);
            G = AllocDevice((long)newCapacity * _numVHeads * single);
            DeltaNetOut = AllocDevice((long)newCapacity * _innerSize * half);
            NormedDeltaNetOut = AllocDevice((long)newCapacity * _innerSize * half);
        }

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

        if (_hasSsm)
        {
            FreeDevice(ref QgScratch, (long)_currentSeqLen * _numHeads * _headDim * 2 * half);
            FreeDevice(ref AttentionGate, (long)_currentSeqLen * _numHeads * _headDim * half);

            FreeDevice(ref QkvRaw, (long)_currentSeqLen * _convDim * half);
            FreeDevice(ref ConvOut, (long)_currentSeqLen * _convDim * half);
            FreeDevice(ref SsmGateRaw, (long)_currentSeqLen * _innerSize * half);
            FreeDevice(ref BetaRaw, (long)_currentSeqLen * _numVHeads * half);
            FreeDevice(ref AlphaRaw, (long)_currentSeqLen * _numVHeads * half);
            FreeDevice(ref QRepeated, (long)_currentSeqLen * _numVHeads * _headKDim * half);
            FreeDevice(ref KRepeated, (long)_currentSeqLen * _numVHeads * _headKDim * half);
            FreeDevice(ref Beta, (long)_currentSeqLen * _numVHeads * single);
            FreeDevice(ref G, (long)_currentSeqLen * _numVHeads * single);
            FreeDevice(ref DeltaNetOut, (long)_currentSeqLen * _innerSize * half);
            FreeDevice(ref NormedDeltaNetOut, (long)_currentSeqLen * _innerSize * half);
        }

        _currentSeqLen = 0;
    }

    public void Dispose()
    {
        FreeSequenceBuffers();

        if (_totalConvStateBytes > 0)
        {
            nint cs = ConvStates;
            FreeDevice(ref cs, _totalConvStateBytes);
        }
        if (_totalSsmStateBytes > 0)
        {
            nint ss = SsmStates;
            FreeDevice(ref ss, _totalSsmStateBytes);
        }

        FreeDevice(ref LogitsF16, (long)_vocabSize * Fp16Bytes);
        FreeDevice(ref LogitsF32, (long)_vocabSize * sizeof(float));

        long maxDim = Math.Max(
            Math.Max(_numHeads * _headDim, _numKvHeads * _headDim),
            _intermediateSize);
        if (_hasSsm)
        {
            maxDim = Math.Max(maxDim, Math.Max(_convDim, _innerSize));
            maxDim = Math.Max(maxDim, _numHeads * _headDim * 2);
        }
        long maxProjectionElements = maxDim * _hiddenSize;
        FreeDevice(ref DequantScratch, maxProjectionElements * Fp16Bytes);

        long maxActivationElements = Math.Max(_hiddenSize, _intermediateSize);
        if (_hasSsm)
            maxActivationElements = Math.Max(maxActivationElements, Math.Max(_convDim, _innerSize));
        long maxQ8_1Bytes = ((maxActivationElements + 31) / 32) * 36;
        FreeDevice(ref Q8_1Scratch, maxQ8_1Bytes);

        FreeDevice(ref TokenIdsDevice, 2048 * sizeof(int));
        FreeDevice(ref PositionsDevice, 2048 * sizeof(int));

        TokenIdsHost.Dispose();
        PositionsHost.Dispose();
        LogitsHost.Dispose();
    }
}
