using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// rocBLAS GEMM and GEMV wrappers for linear projections.
/// Input activations and weights are FP16; accumulation is performed in FP32 (D5).
/// Matrices are stored in row-major format, mapped to column-major rocBLAS using the
/// standard algebraic transpose identity: <c>Y = X × Wᵀ &lt;=&gt; Yᵀ = W × Xᵀ</c>.
/// </summary>
public static class HipGemm
{
    private static readonly float s_alphaOne = 1.0f;
    private static readonly float s_betaZero = 0.0f;

    /// <summary>
    /// Linear projection: <c>Y_f16[m, n] = X_f16[m, k] × W_f16ᵀ[n, k]</c>.
    /// Computes with FP16 inputs/outputs and FP32 accumulation.
    /// </summary>
    /// <param name="handle">Active rocBLAS handle.</param>
    /// <param name="xF16">Device pointer to input tensor [m, k] (FP16).</param>
    /// <param name="wF16">Device pointer to weight tensor [n, k] (FP16).</param>
    /// <param name="yF16">Device pointer to output tensor [m, n] (FP16).</param>
    /// <param name="m">Batch / sequence length (rows of X).</param>
    /// <param name="k">Input dimension (cols of X, cols of W).</param>
    /// <param name="n">Output dimension (rows of W, cols of Y).</param>
    /// <param name="stream">HIP stream on which to enqueue the GEMM operation.</param>
    public static unsafe void LinearF16(
        nint handle,
        nint xF16,
        nint wF16,
        nint yF16,
        int m,
        int k,
        int n,
        nint stream)
    {
        RocblasApi.rocblas_set_stream(handle, stream).ThrowOnRocblasError();

        float alpha = s_alphaOne;
        float beta = s_betaZero;

        RocblasApi.rocblas_gemm_ex(
            handle,
            RocblasApi.rocblas_operation_transpose,
            RocblasApi.rocblas_operation_none,
            n, m, k,
            (nint)(&alpha),
            wF16, RocblasApi.rocblas_datatype_f16_r, k,
            xF16, RocblasApi.rocblas_datatype_f16_r, k,
            (nint)(&beta),
            yF16, RocblasApi.rocblas_datatype_f16_r, n,
            yF16, RocblasApi.rocblas_datatype_f16_r, n,
            RocblasApi.rocblas_datatype_f32_r,
            RocblasApi.rocblas_gemm_algo_standard,
            solutionIndex: 0,
            flags: 0
        ).ThrowOnRocblasError();
    }

    /// <summary>
    /// Matrix-vector multiplication for single-token decode: <c>y_f16[n] = W_f16[n, k] × x_f16[k]</c>.
    /// </summary>
    /// <param name="handle">Active rocBLAS handle.</param>
    /// <param name="wF16">Device pointer to weight matrix [n, k] (FP16).</param>
    /// <param name="xF16">Device pointer to activation vector [k] (FP16).</param>
    /// <param name="yF16">Device pointer to output vector [n] (FP16).</param>
    /// <param name="n">Output dimension.</param>
    /// <param name="k">Input dimension.</param>
    /// <param name="stream">HIP stream handle.</param>
    public static void GemvF16(
        nint handle,
        nint wF16,
        nint xF16,
        nint yF16,
        int n,
        int k,
        nint stream)
    {
        LinearF16(handle, xF16, wF16, yF16, 1, k, n, stream);
    }
}
