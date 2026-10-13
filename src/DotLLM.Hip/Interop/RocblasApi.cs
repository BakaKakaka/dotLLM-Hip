using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Direct P/Invoke declarations against AMD rocBLAS (<c>rocblas.dll</c>).
/// Loaded only after <c>amdhip64_7.dll</c> via <see cref="HipLibraryResolver"/> (W6).
/// </summary>
internal static unsafe partial class RocblasApi
{
    private const string LibName = "rocblas";

    // rocblas_status
    public const int rocblas_status_success = 0;
    public const int rocblas_status_invalid_handle = 1;
    public const int rocblas_status_not_implemented = 2;
    public const int rocblas_status_invalid_pointer = 3;
    public const int rocblas_status_invalid_size = 4;
    public const int rocblas_status_memory_error = 5;
    public const int rocblas_status_internal_error = 6;
    public const int rocblas_status_perf_degraded = 7;
    public const int rocblas_status_size_query_mismatch = 8;

    // rocblas_operation
    public const int rocblas_operation_none = 111;
    public const int rocblas_operation_transpose = 112;

    // rocblas_datatype
    public const int rocblas_datatype_f16_r = 150;
    public const int rocblas_datatype_f32_r = 151;

    // rocblas_gemm_algo
    public const int rocblas_gemm_algo_standard = 0;

    static RocblasApi() => HipLibraryResolver.Register();

    [LibraryImport(LibName)]
    public static partial int rocblas_create_handle(out nint handle);

    [LibraryImport(LibName)]
    public static partial int rocblas_destroy_handle(nint handle);

    [LibraryImport(LibName)]
    public static partial int rocblas_set_stream(nint handle, nint stream);

    [LibraryImport(LibName)]
    public static partial int rocblas_get_stream(nint handle, out nint stream);

    [LibraryImport(LibName)]
    public static partial byte* rocblas_status_to_string(int status);

    [LibraryImport(LibName)]
    public static partial int rocblas_gemm_ex(
        nint handle,
        int transA,
        int transB,
        int m,
        int n,
        int k,
        nint alpha,
        nint a,
        int aType,
        int lda,
        nint b,
        int bType,
        int ldb,
        nint beta,
        nint c,
        int cType,
        int ldc,
        nint d,
        int dType,
        int ldd,
        int computeType,
        int algo,
        int solutionIndex,
        uint flags);

    /// <summary>Returns the symbolic status string or message for a rocBLAS status code.</summary>
    public static string GetStatusString(int status)
    {
        try
        {
            byte* p = rocblas_status_to_string(status);
            return p != null ? Marshal.PtrToStringAnsi((nint)p) ?? $"Status {status}" : $"Status {status}";
        }
        catch
        {
            return status switch
            {
                rocblas_status_invalid_handle => "rocblas_status_invalid_handle",
                rocblas_status_not_implemented => "rocblas_status_not_implemented",
                rocblas_status_invalid_pointer => "rocblas_status_invalid_pointer",
                rocblas_status_invalid_size => "rocblas_status_invalid_size",
                rocblas_status_memory_error => "rocblas_status_memory_error",
                rocblas_status_internal_error => "rocblas_status_internal_error",
                rocblas_status_perf_degraded => "rocblas_status_perf_degraded",
                rocblas_status_size_query_mismatch => "rocblas_status_size_query_mismatch",
                _ => $"Status {status}"
            };
        }
    }
}
