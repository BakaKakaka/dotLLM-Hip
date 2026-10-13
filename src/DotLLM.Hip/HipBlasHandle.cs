using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// RAII wrapper around an AMD rocBLAS context handle (<c>rocblas_handle</c>).
/// </summary>
public sealed class HipBlasHandle : IDisposable
{
    private nint _handle;

    /// <summary>Native rocBLAS context handle.</summary>
    public nint Handle => _handle;

    private HipBlasHandle(nint handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Creates a new rocBLAS handle on the active HIP device.
    /// </summary>
    public static HipBlasHandle Create()
    {
        HipLibraryResolver.Register();
        RocblasApi.rocblas_create_handle(out nint handle).ThrowOnRocblasError();
        return new HipBlasHandle(handle);
    }

    /// <summary>
    /// Binds this rocBLAS handle to a <see cref="HipStream"/>.
    /// </summary>
    public void SetStream(HipStream stream)
    {
        ObjectDisposedException.ThrowIf(_handle == 0, this);
        RocblasApi.rocblas_set_stream(_handle, stream.Handle).ThrowOnRocblasError();
    }

    /// <summary>
    /// Binds this rocBLAS handle to a raw HIP stream handle.
    /// </summary>
    public void SetStream(nint streamHandle)
    {
        ObjectDisposedException.ThrowIf(_handle == 0, this);
        RocblasApi.rocblas_set_stream(_handle, streamHandle).ThrowOnRocblasError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint handle = Interlocked.Exchange(ref _handle, 0);
        if (handle != 0)
            RocblasApi.rocblas_destroy_handle(handle);
    }
}
