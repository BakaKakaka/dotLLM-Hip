using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// RAII wrapper around a HIP execution stream (<c>hipStream_t</c>).
/// </summary>
public sealed class HipStream : IDisposable
{
    private nint _stream;

    /// <summary>The native HIP stream handle.</summary>
    public nint Handle => _stream;

    private HipStream(nint stream)
    {
        _stream = stream;
    }

    /// <summary>
    /// Creates a new HIP stream on the active device.
    /// Defaults to non-blocking stream (<c>hipStreamNonBlocking</c>).
    /// </summary>
    /// <param name="nonBlocking">If true, does not serialize with the NULL stream.</param>
    public static HipStream Create(bool nonBlocking = true)
    {
        uint flags = nonBlocking ? HipApi.hipStreamNonBlocking : HipApi.hipStreamDefault;
        HipApi.hipStreamCreateWithFlags(out nint stream, flags).ThrowOnError();
        return new HipStream(stream);
    }

    /// <summary>
    /// Blocks the calling thread until all operations queued on this stream have completed.
    /// </summary>
    public void Synchronize()
    {
        ObjectDisposedException.ThrowIf(_stream == 0, this);
        HipApi.hipStreamSynchronize(_stream).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint stream = Interlocked.Exchange(ref _stream, 0);
        if (stream != 0)
            HipApi.hipStreamDestroy(stream);
    }
}
