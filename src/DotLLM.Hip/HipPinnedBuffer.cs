using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// RAII wrapper around pinned (page-locked) host memory allocated via <c>hipHostMalloc</c>.
/// Enables high-bandwidth, asynchronous host-to-device and device-to-host transfers.
/// </summary>
public sealed class HipPinnedBuffer : IDisposable
{
    private nint _ptr;

    /// <summary>Total byte capacity of the pinned buffer.</summary>
    public long ByteCount { get; }

    /// <summary>Raw host pointer to the pinned memory region.</summary>
    public nint Pointer => _ptr;

    /// <summary>
    /// Allocates pinned host memory of the requested size.
    /// </summary>
    /// <param name="byteCount">Size in bytes to allocate.</param>
    /// <param name="flags">Allocation flags (<c>hipHostMallocDefault</c> by default).</param>
    public HipPinnedBuffer(long byteCount, uint flags = HipApi.hipHostMallocDefault)
    {
        if (byteCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Byte count must be positive.");

        ByteCount = byteCount;
        HipApi.hipHostMalloc(out nint ptr, (nuint)byteCount, flags).ThrowOnError();
        _ptr = ptr;
    }

    /// <summary>
    /// Obtains a <see cref="Span{T}"/> over the pinned memory.
    /// </summary>
    public unsafe Span<T> AsSpan<T>() where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_ptr == 0, this);
        int elementCount = checked((int)(ByteCount / sizeof(T)));
        return new Span<T>((void*)_ptr, elementCount);
    }

    /// <summary>
    /// Obtains a <see cref="ReadOnlySpan{T}"/> over the pinned memory.
    /// </summary>
    public unsafe ReadOnlySpan<T> AsReadOnlySpan<T>() where T : unmanaged
    {
        ObjectDisposedException.ThrowIf(_ptr == 0, this);
        int elementCount = checked((int)(ByteCount / sizeof(T)));
        return new ReadOnlySpan<T>((void*)_ptr, elementCount);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _ptr, 0);
        if (ptr != 0)
            HipApi.hipHostFree(ptr);
    }
}
