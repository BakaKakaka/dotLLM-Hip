using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// GPU-resident KV-cache storing FP16 key and value vectors per layer on AMD GPU.
/// Layout: [maxSeqLen, numKvHeads * headDim] per layer, FP16.
/// </summary>
public sealed class HipKvCache : IKvCache
{
    private const int Fp16Bytes = 2;

    private readonly nint[] _keys;
    private readonly nint[] _values;
    private readonly int _numLayers;
    private readonly int _kvStride;    // numKvHeads * headDim
    private readonly int _maxSeqLen;
    private int _currentLength;

    /// <inheritdoc/>
    public int CurrentLength => _currentLength;

    /// <inheritdoc/>
    public int MaxLength => _maxSeqLen;

    /// <summary>
    /// Allocates GPU KV-cache buffers for all layers.
    /// </summary>
    public HipKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _keys = new nint[numLayers];
        _values = new nint[numLayers];

        nuint bytesPerLayer = (nuint)((long)maxSeqLen * _kvStride * Fp16Bytes);
        for (int i = 0; i < numLayers; i++)
        {
            HipApi.hipMalloc(out _keys[i], bytesPerLayer).ThrowOnError();
            HipApi.hipMalloc(out _values[i], bytesPerLayer).ThrowOnError();
        }
    }

    /// <summary>
    /// Updates KV-cache from device pointers asynchronously on the specified stream.
    /// </summary>
    internal void UpdateDevice(nint keysDevice, nint valuesDevice,
                               ReadOnlySpan<int> positions, int seqLen,
                               int layerIndex, nint stream)
    {
        long rowBytes = (long)_kvStride * Fp16Bytes;

        bool contiguous = seqLen > 0;
        for (int i = 0; i < seqLen; i++)
        {
            if ((uint)positions[i] >= (uint)_maxSeqLen)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max KV-cache length {_maxSeqLen}.");
            if (i > 0 && positions[i] != positions[i - 1] + 1)
                contiguous = false;
        }

        if (contiguous && seqLen > 1)
        {
            nuint bulkBytes = (nuint)((long)seqLen * rowBytes);
            nint kDst = _keys[layerIndex] + (nint)(positions[0] * rowBytes);
            nint vDst = _values[layerIndex] + (nint)(positions[0] * rowBytes);
            HipApi.hipMemcpyAsync(kDst, keysDevice, bulkBytes, HipApi.hipMemcpyDeviceToDevice, stream).ThrowOnError();
            HipApi.hipMemcpyAsync(vDst, valuesDevice, bulkBytes, HipApi.hipMemcpyDeviceToDevice, stream).ThrowOnError();
        }
        else
        {
            nuint rowBytesNuint = (nuint)rowBytes;
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                nint kDst = _keys[layerIndex] + (nint)(pos * rowBytes);
                nint vDst = _values[layerIndex] + (nint)(pos * rowBytes);
                nint kSrc = keysDevice + (nint)(i * rowBytes);
                nint vSrc = valuesDevice + (nint)(i * rowBytes);
                HipApi.hipMemcpyAsync(kDst, kSrc, rowBytesNuint, HipApi.hipMemcpyDeviceToDevice, stream).ThrowOnError();
                HipApi.hipMemcpyAsync(vDst, vSrc, rowBytesNuint, HipApi.hipMemcpyDeviceToDevice, stream).ThrowOnError();
            }
        }

        int maxPos = positions[seqLen - 1];
        for (int i = 0; i < seqLen; i++)
        {
            if (positions[i] > maxPos) maxPos = positions[i];
        }
        int newLength = maxPos + 1;
        if (newLength > _currentLength)
            _currentLength = newLength;
    }

    /// <summary>Returns device pointer to cached keys for the given layer.</summary>
    internal nint GetKeysPtr(int layerIndex) => _keys[layerIndex];

    /// <summary>Returns device pointer to cached values for the given layer.</summary>
    internal nint GetValuesPtr(int layerIndex) => _values[layerIndex];

    // ── IKvCache interface implementation ─────────────────────────────

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        throw new NotSupportedException("HipKvCache.Update(ITensor) not supported. Use UpdateDevice().");
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        throw new NotSupportedException("HipKvCache.Update(TensorRef) not supported. Use UpdateDevice().");
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        throw new NotSupportedException("HipKvCache.GetKeys(ITensor) not supported. Use GetKeysPtr().");
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        throw new NotSupportedException("HipKvCache.GetValues(ITensor) not supported. Use GetValuesPtr().");
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex) =>
        new(_currentLength, _kvStride, DType.Float16, 0, _keys[layerIndex]);

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex) =>
        new(_currentLength, _kvStride, DType.Float16, 0, _values[layerIndex]);

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
            throw new ArgumentOutOfRangeException(nameof(length));
        _currentLength = length;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keys[i] != 0) { HipApi.hipFree(_keys[i]); _keys[i] = 0; }
            if (_values[i] != 0) { HipApi.hipFree(_values[i]); _values[i] = 0; }
        }
    }
}
