using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// GPU-resident tensor backed by <c>hipMalloc</c>. Owns unmanaged device memory
/// and frees it deterministically on disposal. <see cref="DataPointer"/> is an opaque
/// device pointer and must not be dereferenced from host code.
/// </summary>
public sealed class HipTensor : ITensor
{
    private nint _ptr;
    private readonly bool _ownsMemory;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <inheritdoc/>
    public nint DataPointer => _ptr;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, _ptr);

    /// <inheritdoc/>
    public long ElementCount => Shape.ElementCount;

    /// <inheritdoc/>
    public long ByteCount { get; }

    private HipTensor(TensorShape shape, DType dtype, int deviceId, nint ptr, long byteCount, bool ownsMemory = true)
    {
        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        _ptr = ptr;
        ByteCount = byteCount;
        _ownsMemory = ownsMemory;
    }

    /// <summary>
    /// Allocates a GPU tensor of the given shape and data type on the specified device.
    /// </summary>
    /// <param name="shape">Tensor shape.</param>
    /// <param name="dtype">Element data type.</param>
    /// <param name="deviceId">HIP device ordinal (0-based).</param>
    public static HipTensor Allocate(TensorShape shape, DType dtype, int deviceId)
    {
        long bytes = dtype.ComputeByteCount(shape.ElementCount);
        if (bytes <= 0)
            throw new ArgumentException($"Cannot allocate tensor with {shape.ElementCount} elements of {dtype} (computed byte count={bytes}).");

        HipApi.hipSetDevice(deviceId).ThrowOnError();
        HipApi.hipMalloc(out nint ptr, (nuint)bytes).ThrowOnError();
        return new HipTensor(shape, dtype, deviceId, ptr, bytes);
    }

    /// <summary>
    /// Allocates a GPU tensor with an explicit byte count (e.g. For block-quantized types).
    /// </summary>
    public static HipTensor AllocateBytes(TensorShape shape, DType dtype, int deviceId, long byteCount)
    {
        if (byteCount <= 0)
            throw new ArgumentException($"Byte count must be positive, got {byteCount}.");

        HipApi.hipSetDevice(deviceId).ThrowOnError();
        HipApi.hipMalloc(out nint ptr, (nuint)byteCount).ThrowOnError();
        return new HipTensor(shape, dtype, deviceId, ptr, byteCount);
    }

    /// <summary>
    /// Wraps an existing device pointer as a non-owning tensor view.
    /// The caller remains responsible for lifetime and memory cleanup.
    /// </summary>
    public static HipTensor WrapExisting(TensorShape shape, DType dtype, int deviceId, nint devicePtr, long byteCount)
    {
        return new HipTensor(shape, dtype, deviceId, devicePtr, byteCount, ownsMemory: false);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _ptr, 0);
        if (ptr != 0 && _ownsMemory)
            HipApi.hipFree(ptr);
    }
}
