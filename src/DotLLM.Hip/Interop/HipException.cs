using System.Runtime.CompilerServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Exception thrown when a HIP or rocBLAS operation fails.
/// </summary>
public class HipException : Exception
{
    /// <summary>The raw HIP or rocBLAS error status code, or -1 if unspecified.</summary>
    public int ErrorCode { get; }

    /// <summary>Initializes a new instance of <see cref="HipException"/> with a message.</summary>
    public HipException(string message) : base(message)
    {
        ErrorCode = -1;
    }

    /// <summary>Initializes a new instance of <see cref="HipException"/> with an error code and message.</summary>
    public HipException(int errorCode, string message) : base(message)
    {
        ErrorCode = errorCode;
    }

    /// <summary>Initializes a new instance of <see cref="HipException"/> with an error code, message, and inner exception.</summary>
    public HipException(int errorCode, string message, Exception? innerException) : base(message, innerException)
    {
        ErrorCode = errorCode;
    }

    /// <summary>
    /// Throws a <see cref="HipException"/> if <paramref name="error"/> is non-zero (<c>hipSuccess = 0</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowIfError(int error, [CallerArgumentExpression(nameof(error))] string? call = null)
    {
        if (error != HipApi.hipSuccess)
            Throw(error, call);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Throw(int error, string? call)
    {
        string name = HipApi.GetErrorNameString(error);
        string desc = HipApi.GetErrorStringText(error);
        string msg = string.IsNullOrEmpty(call)
            ? $"HIP error {error} ({name}): {desc}"
            : $"{call} failed with HIP error {error} ({name}): {desc}";
        throw new HipException(error, msg);
    }

    /// <summary>
    /// Throws a <see cref="HipException"/> if <paramref name="status"/> is non-zero (<c>rocblas_status_success = 0</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ThrowIfRocblasError(int status, [CallerArgumentExpression(nameof(status))] string? call = null)
    {
        if (status != RocblasApi.rocblas_status_success)
            ThrowRocblas(status, call);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowRocblas(int status, string? call)
    {
        string desc = RocblasApi.GetStatusString(status);
        string msg = string.IsNullOrEmpty(call)
            ? $"rocBLAS error {status}: {desc}"
            : $"{call} failed with rocBLAS error {status}: {desc}";
        throw new HipException(status, msg);
    }
}

/// <summary>
/// Extension methods for checking HIP and rocBLAS return codes.
/// </summary>
internal static class HipErrorExtensions
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ThrowOnError(this int error, [CallerArgumentExpression(nameof(error))] string? call = null)
        => HipException.ThrowIfError(error, call);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ThrowOnRocblasError(this int status, [CallerArgumentExpression(nameof(status))] string? call = null)
        => HipException.ThrowIfRocblasError(status, call);
}
