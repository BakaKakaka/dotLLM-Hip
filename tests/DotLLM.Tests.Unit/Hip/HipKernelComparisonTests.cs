using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Unit tests for all elementwise and norm kernels against CPU reference implementations (H2.2).
/// Runs across both physical GPUs (RX 7900 XTX gfx1100 and RX 9070 XT gfx1201).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipKernelComparisonTests
{
    private const int HiddenSize = 576; // SmolLM-135M dimensions
    private const int NumHeads = 9;
    private const int NumKvHeads = 3;
    private const int HeadDim = 64;
    private const float RmsEps = 1e-5f;
    private const float RopeTheta = 10000.0f;

    private static float[] RandomF32(Random rng, int count, float scale = 1.0f)
    {
        float[] arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = ((float)rng.NextDouble() * 2.0f - 1.0f) * scale;
        return arr;
    }

    private static Half[] RandomF16(Random rng, int count, float scale = 1.0f)
    {
        Half[] arr = new Half[count];
        for (int i = 0; i < count; i++)
            arr[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * scale);
        return arr;
    }

    private static void CompareFloats(string name, float[] expected, float[] actual, float tolerance, string arch)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            Assert.True(diff <= tolerance,
                $"{arch} {name} at index {i}: expected {expected[i]}, got {actual[i]}, diff {diff} > tol {tolerance}");
        }
    }

    private static void CompareHalves(string name, Half[] expected, Half[] actual, float tolerance, string arch)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float expF = (float)expected[i];
            float actF = (float)actual[i];
            float diff = MathF.Abs(expF - actF);
            Assert.True(diff <= tolerance,
                $"{arch} {name} at index {i}: expected {expF}, got {actF}, diff {diff} > tol {tolerance}");
        }
    }

    // ─────────────────────── RMSNorm ───────────────────────

    [SkippableFact]
    public unsafe void RmsNormF32InF16Out_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = HiddenSize;
        int rows = 4;
        var rng = new Random(42);
        float[] input = RandomF32(rng, n * rows);
        float[] weight = RandomF32(rng, n, scale: 1.0f);

        // CPU reference: compute RMSNorm in FP32, convert to FP16
        Half[] cpuResult = new Half[n * rows];
        float[] tempRow = new float[n];
        for (int r = 0; r < rows; r++)
        {
            RmsNorm.Execute(input.AsSpan(r * n, n), weight, RmsEps, tempRow);
            for (int i = 0; i < n; i++)
                cpuResult[r * n + i] = (Half)tempRow[i];
        }

        nuint inBytes = (nuint)(input.Length * sizeof(float));
        nuint weightBytes = (nuint)(weight.Length * sizeof(float));
        nuint outBytes = (nuint)(cpuResult.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dIn, inBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dWeight, weightBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (float* pIn = input, pW = weight)
                {
                    HipApi.hipMemcpy(dIn, (nint)pIn, inBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dWeight, (nint)pW, weightBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchRmsNormF32InF16Out(dIn, dWeight, dOut, n, RmsEps, rows, stream.Handle);
                stream.Synchronize();

                Half[] gpuResult = new Half[cpuResult.Length];
                fixed (Half* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("RmsNormF32InF16Out", cpuResult, gpuResult, tolerance: 0.005f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dIn);
                HipApi.hipFree(dWeight);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void RmsNormF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = HiddenSize;
        int rows = 4;
        var rng = new Random(42);
        float[] input = RandomF32(rng, n * rows);
        float[] weight = RandomF32(rng, n, scale: 1.0f);

        float[] cpuResult = new float[n * rows];
        for (int r = 0; r < rows; r++)
            RmsNorm.Execute(input.AsSpan(r * n, n), weight, RmsEps, cpuResult.AsSpan(r * n, n));

        nuint inBytes = (nuint)(input.Length * sizeof(float));
        nuint weightBytes = (nuint)(weight.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dIn, inBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dWeight, weightBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, inBytes).ThrowOnError();

            try
            {
                fixed (float* pIn = input, pW = weight)
                {
                    HipApi.hipMemcpy(dIn, (nint)pIn, inBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dWeight, (nint)pW, weightBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchRmsNormF32(dIn, dWeight, dOut, n, RmsEps, rows, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[cpuResult.Length];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, inBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("RmsNormF32", cpuResult, gpuResult, tolerance: 0.001f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dIn);
                HipApi.hipFree(dWeight);
                HipApi.hipFree(dOut);
            }
        }
    }

    // ─────────────────────── Fused Add + RMSNorm ───────────────────────

    [SkippableFact]
    public unsafe void FusedAddRmsNormF32Res_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = HiddenSize;
        int rows = 4;
        var rng = new Random(42);
        float[] residual = RandomF32(rng, n * rows);
        Half[] x = RandomF16(rng, n * rows);
        float[] weight = RandomF32(rng, n, scale: 1.0f);

        // CPU reference:
        // 1. updated_residual = residual + (float)x
        // 2. norm_output = rmsnorm(updated_residual, weight) as Half
        float[] expectedResidual = new float[n * rows];
        Half[] expectedNormOut = new Half[n * rows];
        float[] tempRow = new float[n];

        for (int r = 0; r < rows; r++)
        {
            for (int i = 0; i < n; i++)
            {
                expectedResidual[r * n + i] = residual[r * n + i] + (float)x[r * n + i];
            }
            RmsNorm.Execute(expectedResidual.AsSpan(r * n, n), weight, RmsEps, tempRow);
            for (int i = 0; i < n; i++)
                expectedNormOut[r * n + i] = (Half)tempRow[i];
        }

        nuint resBytes = (nuint)(residual.Length * sizeof(float));
        nuint xBytes = (nuint)(x.Length * sizeof(Half));
        nuint weightBytes = (nuint)(weight.Length * sizeof(float));
        nuint outBytes = (nuint)(expectedNormOut.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dRes, resBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, xBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dWeight, weightBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (float* pRes = residual, pW = weight)
                fixed (Half* pX = x)
                {
                    HipApi.hipMemcpy(dRes, (nint)pRes, resBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dWeight, (nint)pW, weightBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchFusedAddRmsNormF32Res(dRes, dX, dWeight, dOut, n, RmsEps, rows, stream.Handle);
                stream.Synchronize();

                float[] gpuResidual = new float[residual.Length];
                Half[] gpuNormOut = new Half[expectedNormOut.Length];

                fixed (float* pRes = gpuResidual)
                fixed (Half* pOut = gpuNormOut)
                {
                    HipApi.hipMemcpy((nint)pRes, dRes, resBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("FusedAddRmsNorm Residual", expectedResidual, gpuResidual, tolerance: 0.001f, device.Arch);
                CompareHalves("FusedAddRmsNorm Output", expectedNormOut, gpuNormOut, tolerance: 0.005f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dRes);
                HipApi.hipFree(dX);
                HipApi.hipFree(dWeight);
                HipApi.hipFree(dOut);
            }
        }
    }

    // ─────────────────────── RoPE ───────────────────────

    [SkippableFact]
    public unsafe void RoPEF32_NormAndNeoX_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqLen = 4;
        int ropeDim = HeadDim;
        int halfRope = ropeDim / 2;
        var rng = new Random(42);

        float[] qOrig = RandomF32(rng, seqLen * NumHeads * HeadDim);
        float[] kOrig = RandomF32(rng, seqLen * NumKvHeads * HeadDim);
        int[] positions = [0, 1, 2, 3];

        int maxPos = seqLen + 1;
        float[] cosTable = new float[maxPos * halfRope];
        float[] sinTable = new float[maxPos * halfRope];
        RoPE.PrecomputeFrequencyTable(maxPos, ropeDim, RopeTheta, cosTable, sinTable);

        // Test both Norm (type 0) and NeoX (type 1)
        foreach (var ropeType in new[] { RoPEType.Norm, RoPEType.NeoX })
        {
            int kernelRopeType = (ropeType == RoPEType.NeoX) ? 1 : 0;

            float[] cpuQ = (float[])qOrig.Clone();
            float[] cpuK = (float[])kOrig.Clone();
            RoPE.Execute(cpuQ, cpuK, positions, NumHeads, NumKvHeads, HeadDim, ropeDim, cosTable, sinTable, ropeType);

            nuint qBytes = (nuint)(qOrig.Length * sizeof(float));
            nuint kBytes = (nuint)(kOrig.Length * sizeof(float));
            nuint posBytes = (nuint)(positions.Length * sizeof(int));

            foreach (var device in devices)
            {
                device.MakeCurrent();
                using var kernels = new HipKernels(device.Arch);
                using var stream = HipStream.Create();

                HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
                HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
                HipApi.hipMalloc(out nint dPos, posBytes).ThrowOnError();

                try
                {
                    fixed (float* pQ = qOrig, pK = kOrig)
                    fixed (int* pPos = positions)
                    {
                        HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                        HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                        HipApi.hipMemcpy(dPos, (nint)pPos, posBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    }

                    kernels.LaunchRoPEF32(dQ, dK, dPos, seqLen, NumHeads, NumKvHeads,
                        HeadDim, ropeDim, RopeTheta, kernelRopeType, stream.Handle);
                    stream.Synchronize();

                    float[] gpuQ = new float[cpuQ.Length];
                    float[] gpuK = new float[cpuK.Length];

                    fixed (float* pQ = gpuQ, pK = gpuK)
                    {
                        HipApi.hipMemcpy((nint)pQ, dQ, qBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)pK, dK, kBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    }

                    CompareFloats($"RoPE-{ropeType} Q", cpuQ, gpuQ, tolerance: 0.001f, device.Arch);
                    CompareFloats($"RoPE-{ropeType} K", cpuK, gpuK, tolerance: 0.001f, device.Arch);
                }
                finally
                {
                    HipApi.hipFree(dQ);
                    HipApi.hipFree(dK);
                    HipApi.hipFree(dPos);
                }
            }
        }
    }

    // ─────────────────────── SwiGLU ───────────────────────

    [SkippableFact]
    public unsafe void SwiGLUF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int intermediateSize = 1024;
        int seqLen = 2;
        int total = intermediateSize * seqLen;
        var rng = new Random(42);
        float[] gate = RandomF32(rng, total);
        float[] up = RandomF32(rng, total);

        float[] cpuResult = new float[total];
        for (int i = 0; i < total; i++)
        {
            float g = gate[i];
            cpuResult[i] = (g / (1.0f + MathF.Exp(-g))) * up[i];
        }

        nuint bytes = (nuint)(total * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dGate, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dUp, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, bytes).ThrowOnError();

            try
            {
                fixed (float* pG = gate, pU = up)
                {
                    HipApi.hipMemcpy(dGate, (nint)pG, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dUp, (nint)pU, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchSwiGLUF32(dGate, dUp, dOut, intermediateSize, seqLen, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[total];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("SwiGLUF32", cpuResult, gpuResult, tolerance: 0.001f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dGate);
                HipApi.hipFree(dUp);
                HipApi.hipFree(dOut);
            }
        }
    }

    // ─────────────────────── Add ───────────────────────

    [SkippableFact]
    public unsafe void AddF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 2048;
        var rng = new Random(42);
        float[] a = RandomF32(rng, n);
        float[] b = RandomF32(rng, n);

        float[] cpuResult = new float[n];
        for (int i = 0; i < n; i++) cpuResult[i] = a[i] + b[i];

        nuint bytes = (nuint)(n * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dA, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dB, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, bytes).ThrowOnError();

            try
            {
                fixed (float* pa = a, pb = b)
                {
                    HipApi.hipMemcpy(dA, (nint)pa, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dB, (nint)pb, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAddF32(dA, dB, dOut, n, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[n];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("AddF32", cpuResult, gpuResult, tolerance: 1e-5f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dA);
                HipApi.hipFree(dB);
                HipApi.hipFree(dOut);
            }
        }
    }

    // ─────────────────────── Bias Add ───────────────────────

    [SkippableFact]
    public unsafe void BiasAddF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int dim = 512;
        int seqLen = 3;
        var rng = new Random(42);
        float[] output = RandomF32(rng, dim * seqLen);
        Half[] bias = RandomF16(rng, dim);

        float[] cpuResult = (float[])output.Clone();
        for (int t = 0; t < seqLen; t++)
            for (int d = 0; d < dim; d++)
                cpuResult[t * dim + d] += (float)bias[d];

        nuint outBytes = (nuint)(output.Length * sizeof(float));
        nuint biasBytes = (nuint)(bias.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dBias, biasBytes).ThrowOnError();

            try
            {
                fixed (float* pOut = output)
                fixed (Half* pB = bias)
                {
                    HipApi.hipMemcpy(dOut, (nint)pOut, outBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dBias, (nint)pB, biasBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchBiasAddF32(dOut, dBias, dim, seqLen, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[output.Length];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("BiasAddF32", cpuResult, gpuResult, tolerance: 0.001f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dOut);
                HipApi.hipFree(dBias);
            }
        }
    }

    // ─────────────────────── Convert ───────────────────────

    [SkippableFact]
    public unsafe void ConvertF16AndF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 1024;
        var rng = new Random(42);
        float[] originalF32 = RandomF32(rng, n);

        Half[] cpuF16 = new Half[n];
        for (int i = 0; i < n; i++) cpuF16[i] = (Half)originalF32[i];

        float[] cpuF32Roundtrip = new float[n];
        for (int i = 0; i < n; i++) cpuF32Roundtrip[i] = (float)cpuF16[i];

        nuint f32Bytes = (nuint)(n * sizeof(float));
        nuint f16Bytes = (nuint)(n * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dF32In, f32Bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dF16, f16Bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dF32Out, f32Bytes).ThrowOnError();

            try
            {
                fixed (float* pF32 = originalF32)
                {
                    HipApi.hipMemcpy(dF32In, (nint)pF32, f32Bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchConvertF32ToF16(dF32In, dF16, n, stream.Handle);
                kernels.LaunchConvertF16ToF32(dF16, dF32Out, n, stream.Handle);
                stream.Synchronize();

                Half[] gpuF16 = new Half[n];
                float[] gpuF32 = new float[n];

                fixed (Half* p16 = gpuF16)
                fixed (float* p32 = gpuF32)
                {
                    HipApi.hipMemcpy((nint)p16, dF16, f16Bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)p32, dF32Out, f32Bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("ConvertF32ToF16", cpuF16, gpuF16, tolerance: 0.0001f, device.Arch);
                CompareFloats("ConvertF16ToF32", cpuF32Roundtrip, gpuF32, tolerance: 0.0001f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dF32In);
                HipApi.hipFree(dF16);
                HipApi.hipFree(dF32Out);
            }
        }
    }

    // ─────────────────────── Softmax ───────────────────────

    [SkippableFact]
    public unsafe void SoftmaxF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int rows = 4;
        int cols = 512;
        var rng = new Random(42);
        float[] input = RandomF32(rng, rows * cols, scale: 5.0f);

        float[] cpuResult = new float[rows * cols];
        for (int r = 0; r < rows; r++)
            Softmax.Execute(input.AsSpan(r * cols, cols), cpuResult.AsSpan(r * cols, cols));

        nuint bytes = (nuint)(input.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dIn, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, bytes).ThrowOnError();

            try
            {
                fixed (float* pIn = input)
                {
                    HipApi.hipMemcpy(dIn, (nint)pIn, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchSoftmaxF32(dIn, dOut, rows, cols, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[input.Length];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("SoftmaxF32", cpuResult, gpuResult, tolerance: 0.001f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dIn);
                HipApi.hipFree(dOut);
            }
        }
    }

    // ─────────────────────── Embedding ───────────────────────

    [SkippableFact]
    public unsafe void EmbeddingF32_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int vocabSize = 100;
        int hiddenSize = 256;
        int seqLen = 4;
        var rng = new Random(42);

        float[] table = RandomF32(rng, vocabSize * hiddenSize);
        int[] tokenIds = [5, 17, 42, 99];

        float[] expected = new float[seqLen * hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            int tid = tokenIds[t];
            Array.Copy(table, tid * hiddenSize, expected, t * hiddenSize, hiddenSize);
        }

        nuint tableBytes = (nuint)(table.Length * sizeof(float));
        nuint tokenBytes = (nuint)(tokenIds.Length * sizeof(int));
        nuint outBytes = (nuint)(expected.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dTable, tableBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dTokens, tokenBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (float* pT = table)
                fixed (int* pTok = tokenIds)
                {
                    HipApi.hipMemcpy(dTable, (nint)pT, tableBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dTokens, (nint)pTok, tokenBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchEmbeddingF32(dTable, dTokens, dOut, seqLen, hiddenSize, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[expected.Length];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("EmbeddingF32", expected, gpuResult, tolerance: 0.0f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dTable);
                HipApi.hipFree(dTokens);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void EmbeddingF16_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int vocabSize = 100;
        int hiddenSize = 256;
        int seqLen = 4;
        var rng = new Random(42);

        Half[] table = RandomF16(rng, vocabSize * hiddenSize);
        int[] tokenIds = [5, 17, 42, 99];

        float[] expected = new float[seqLen * hiddenSize];
        for (int t = 0; t < seqLen; t++)
        {
            int tid = tokenIds[t];
            for (int i = 0; i < hiddenSize; i++)
                expected[t * hiddenSize + i] = (float)table[tid * hiddenSize + i];
        }

        nuint tableBytes = (nuint)(table.Length * sizeof(Half));
        nuint tokenBytes = (nuint)(tokenIds.Length * sizeof(int));
        nuint outBytes = (nuint)(expected.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dTable, tableBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dTokens, tokenBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pT = table)
                fixed (int* pTok = tokenIds)
                {
                    HipApi.hipMemcpy(dTable, (nint)pT, tableBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dTokens, (nint)pTok, tokenBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchEmbeddingF16(dTable, dTokens, dOut, seqLen, hiddenSize, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[expected.Length];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("EmbeddingF16", expected, gpuResult, tolerance: 0.0f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dTable);
                HipApi.hipFree(dTokens);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void PerHeadRmsNormF16_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqLen = 3;
        int numHeads = 8;
        int headDim = 64;
        int total = seqLen * numHeads * headDim;
        var rng = new Random(42);

        Half[] qk = RandomF16(rng, total);
        Half[] weight = RandomF16(rng, headDim, scale: 1.0f);

        Half[] expected = (Half[])qk.Clone();
        float[] tempHead = new float[headDim];
        float[] weightF32 = new float[headDim];
        for (int i = 0; i < headDim; i++) weightF32[i] = (float)weight[i];

        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int baseIdx = t * numHeads * headDim + h * headDim;
                for (int i = 0; i < headDim; i++)
                    tempHead[i] = (float)qk[baseIdx + i];

                float[] normOut = new float[headDim];
                RmsNorm.Execute(tempHead, weightF32, RmsEps, normOut);

                for (int i = 0; i < headDim; i++)
                    expected[baseIdx + i] = (Half)normOut[i];
            }
        }

        nuint qkBytes = (nuint)(qk.Length * sizeof(Half));
        nuint weightBytes = (nuint)(weight.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQk, qkBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dWeight, weightBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = qk, pW = weight)
                {
                    HipApi.hipMemcpy(dQk, (nint)pQ, qkBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dWeight, (nint)pW, weightBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchPerHeadRmsNormF16(dQk, dWeight, RmsEps, numHeads, headDim, seqLen, stream.Handle);
                stream.Synchronize();

                Half[] gpuResult = new Half[qk.Length];
                fixed (Half* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dQk, qkBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("PerHeadRmsNormF16", expected, gpuResult, tolerance: 0.005f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQk);
                HipApi.hipFree(dWeight);
            }
        }
    }

    [SkippableFact]
    public unsafe void AddF32F16_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 2048;
        var rng = new Random(42);
        float[] a = RandomF32(rng, n);
        Half[] b = RandomF16(rng, n);

        float[] expected = new float[n];
        for (int i = 0; i < n; i++) expected[i] = a[i] + (float)b[i];

        nuint aBytes = (nuint)(n * sizeof(float));
        nuint bBytes = (nuint)(n * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dA, aBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dB, bBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, aBytes).ThrowOnError();

            try
            {
                fixed (float* pa = a)
                fixed (Half* pb = b)
                {
                    HipApi.hipMemcpy(dA, (nint)pa, aBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dB, (nint)pb, bBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAddF32F16(dA, dB, dOut, n, stream.Handle);
                stream.Synchronize();

                float[] gpuResult = new float[n];
                fixed (float* pOut = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, aBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("AddF32F16", expected, gpuResult, tolerance: 0.0001f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dA);
                HipApi.hipFree(dB);
                HipApi.hipFree(dOut);
            }
        }
    }
}
