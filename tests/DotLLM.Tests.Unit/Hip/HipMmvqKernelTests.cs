using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Unit tests for H3.2 mmvq kernels:
/// - LaunchQuantizeQ8_1 (F16 and F32)
/// - LaunchMmvqGemvQ8_0
/// - LaunchMmvqGemvQ8_0Fdot2
/// - LaunchMmvqGemvQ4_K
/// Evaluated against CPU references across both physical GPUs (RX 7900 XTX gfx1100 and RX 9070 XT gfx1201).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipMmvqKernelTests
{
    [Fact]
    public unsafe void QuantizeQ8_1F16_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int k = 512;
        var rng = new Random(42);
        Half[] xF16 = new Half[k];
        float[] xF32 = new float[k];
        for (int i = 0; i < k; i++)
        {
            float v = ((float)rng.NextDouble() * 2.0f - 1.0f) * 2.5f;
            xF16[i] = (Half)v;
            xF32[i] = (float)xF16[i]; // match exact Half representation
        }

        int numBlocks = k / 32;
        int q8_1Bytes = numBlocks * 36;
        byte[] expected = new byte[q8_1Bytes];

        fixed (float* pSrc = xF32)
        fixed (byte* pDst = expected)
        {
            MatMul.QuantizeF32ToQ8_1(pSrc, pDst, k);
        }

        foreach (var dev in devices)
        {
            dev.MakeCurrent();
            using var kernels = new HipKernels(dev.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dX, (nuint)(k * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dY, (nuint)q8_1Bytes).ThrowOnError();

            try
            {
                fixed (Half* pX = xF16)
                    HipApi.hipMemcpy(dX, (nint)pX, (nuint)(k * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();

                kernels.LaunchQuantizeQ8_1F16(dX, dY, k, stream.Handle);
                stream.Synchronize();

                byte[] actual = new byte[q8_1Bytes];
                fixed (byte* pAct = actual)
                    HipApi.hipMemcpy((nint)pAct, dY, (nuint)q8_1Bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int b = 0; b < numBlocks; b++)
                {
                    int off = b * 36;
                    float expD = (float)BitConverter.ToHalf(expected, off);
                    float actD = (float)BitConverter.ToHalf(actual, off);
                    float expS = (float)BitConverter.ToHalf(expected, off + 2);
                    float actS = (float)BitConverter.ToHalf(actual, off + 2);

                    Assert.True(MathF.Abs(expD - actD) <= 0.001f,
                        $"{dev.Arch} Block {b} scale d: exp {expD}, act {actD}");
                    Assert.True(MathF.Abs(expS - actS) <= MathF.Max(0.05f, MathF.Abs(expS) * 0.05f),
                        $"{dev.Arch} Block {b} sum s: exp {expS}, act {actS}");

                    for (int i = 0; i < 32; i++)
                    {
                        sbyte expQ = (sbyte)expected[off + 4 + i];
                        sbyte actQ = (sbyte)actual[off + 4 + i];
                        // allow +/- 1 rounding difference
                        Assert.True(Math.Abs(expQ - actQ) <= 1,
                            $"{dev.Arch} Block {b} element {i}: exp {expQ}, act {actQ}");
                    }
                }
            }
            finally
            {
                HipApi.hipFree(dX);
                HipApi.hipFree(dY);
            }
        }
    }

    [Fact]
    public unsafe void MmvqGemvQ8_0_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 64;   // 64 rows = 8 waves
        int k = 512;  // 512 cols = 16 blocks
        var rng = new Random(789);

        long wBytes = Dequantize.RowByteSize(k, QuantizationType.Q8_0) * n;
        byte[] weights = new byte[wBytes];
        rng.NextBytes(weights);

        // Valid float16 scales
        for (int r = 0; r < n; r++)
        {
            long rowOff = r * 34 * (k / 32);
            for (int b = 0; b < k / 32; b++)
            {
                ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                weights[rowOff + b * 34] = (byte)(d & 0xFF);
                weights[rowOff + b * 34 + 1] = (byte)(d >> 8);
            }
        }

        Half[] xF16 = new Half[k];
        float[] xF32 = new float[k];
        for (int i = 0; i < k; i++)
        {
            float v = ((float)rng.NextDouble() * 2.0f - 1.0f) * 0.25f;
            xF16[i] = (Half)v;
            xF32[i] = (float)xF16[i];
        }

        // CPU reference: dequantize row, dot with xF32
        float[] rowDeq = new float[k];
        Half[] expected = new Half[n];
        fixed (byte* pW = weights)
        {
            for (int r = 0; r < n; r++)
            {
                Dequantize.ToFloat32((nint)(pW + r * (k / 32) * 34), k, QuantizationType.Q8_0, rowDeq);
                float acc = 0.0f;
                for (int i = 0; i < k; i++) acc += rowDeq[i] * xF32[i];
                expected[r] = (Half)acc;
            }
        }

        foreach (var dev in devices)
        {
            dev.MakeCurrent();
            using var kernels = new HipKernels(dev.Arch);
            using var stream = HipStream.Create();

            int q8_1Bytes = (k / 32) * 36;
            HipApi.hipMalloc(out nint dW, (nuint)wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, (nuint)(k * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dXQ8_1, (nuint)q8_1Bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dY, (nuint)(n * sizeof(Half))).ThrowOnError();

            try
            {
                fixed (byte* pW = weights) fixed (Half* pX = xF16)
                {
                    HipApi.hipMemcpy(dW, (nint)pW, (nuint)wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, (nuint)(k * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                // 1. Quantize x to Q8_1
                kernels.LaunchQuantizeQ8_1F16(dX, dXQ8_1, k, stream.Handle);

                // 2. Run MMVQ GEMV
                kernels.LaunchMmvqGemvQ8_0(dW, dXQ8_1, dY, n, k, stream.Handle);
                stream.Synchronize();

                Half[] actual = new Half[n];
                fixed (Half* pY = actual)
                    HipApi.hipMemcpy((nint)pY, dY, (nuint)(n * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < n; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)actual[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.05f, MathF.Abs(expF) * 0.05f);
                    Assert.True(diff <= tol, $"{dev.Arch} MMVQ Q8_0 row {r}: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dW); HipApi.hipFree(dX); HipApi.hipFree(dXQ8_1); HipApi.hipFree(dY);
            }
        }
    }

    [Fact]
    public unsafe void MmvqGemvQ4_K_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 64;   // 64 rows = 8 waves
        int k = 512;  // 512 cols = 2 superblocks
        var rng = new Random(101112);

        int numSb = k / 256;
        long wBytes = n * numSb * 144;
        byte[] weights = new byte[wBytes];
        rng.NextBytes(weights);

        // Valid float16 scales and mins
        for (int r = 0; r < n; r++)
        {
            long rowOff = r * numSb * 144;
            for (int sb = 0; sb < numSb; sb++)
            {
                ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                ushort dmin = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.02f + 0.005f));
                weights[rowOff + sb * 144] = (byte)(d & 0xFF);
                weights[rowOff + sb * 144 + 1] = (byte)(d >> 8);
                weights[rowOff + sb * 144 + 2] = (byte)(dmin & 0xFF);
                weights[rowOff + sb * 144 + 3] = (byte)(dmin >> 8);
            }
        }

        Half[] xF16 = new Half[k];
        float[] xF32 = new float[k];
        for (int i = 0; i < k; i++)
        {
            float v = ((float)rng.NextDouble() * 2.0f - 1.0f) * 0.25f;
            xF16[i] = (Half)v;
            xF32[i] = (float)xF16[i];
        }

        // CPU reference: dequantize row, dot with xF32
        float[] rowDeq = new float[k];
        Half[] expected = new Half[n];
        fixed (byte* pW = weights)
        {
            for (int r = 0; r < n; r++)
            {
                Dequantize.ToFloat32((nint)(pW + r * numSb * 144), k, QuantizationType.Q4_K, rowDeq);
                float acc = 0.0f;
                for (int i = 0; i < k; i++) acc += rowDeq[i] * xF32[i];
                expected[r] = (Half)acc;
            }
        }

        foreach (var dev in devices)
        {
            dev.MakeCurrent();
            using var kernels = new HipKernels(dev.Arch);
            using var stream = HipStream.Create();

            int q8_1Bytes = (k / 32) * 36;
            HipApi.hipMalloc(out nint dW, (nuint)wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, (nuint)(k * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dXQ8_1, (nuint)q8_1Bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dY, (nuint)(n * sizeof(Half))).ThrowOnError();

            try
            {
                fixed (byte* pW = weights) fixed (Half* pX = xF16)
                {
                    HipApi.hipMemcpy(dW, (nint)pW, (nuint)wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, (nuint)(k * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                // 1. Quantize x to Q8_1
                kernels.LaunchQuantizeQ8_1F16(dX, dXQ8_1, k, stream.Handle);

                // 2. Run MMVQ GEMV
                kernels.LaunchMmvqGemvQ4_K(dW, dXQ8_1, dY, n, k, stream.Handle);
                stream.Synchronize();

                Half[] actual = new Half[n];
                fixed (Half* pY = actual)
                    HipApi.hipMemcpy((nint)pY, dY, (nuint)(n * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < n; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)actual[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.2f, MathF.Abs(expF) * 0.05f);
                    Assert.True(diff <= tol, $"{dev.Arch} MMVQ Q4_K row {r}: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dW); HipApi.hipFree(dX); HipApi.hipFree(dXQ8_1); HipApi.hipFree(dY);
            }
        }
    }

    [Fact]
    public unsafe void MmvqGemvQ8_0Fdot2_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 64;
        int k = 512;
        var rng = new Random(3344);

        long wBytes = Dequantize.RowByteSize(k, QuantizationType.Q8_0) * n;
        byte[] weights = new byte[wBytes];
        rng.NextBytes(weights);

        for (int r = 0; r < n; r++)
        {
            long rowOff = r * 34 * (k / 32);
            for (int b = 0; b < k / 32; b++)
            {
                ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                weights[rowOff + b * 34] = (byte)(d & 0xFF);
                weights[rowOff + b * 34 + 1] = (byte)(d >> 8);
            }
        }

        Half[] xF16 = new Half[k];
        for (int i = 0; i < k; i++)
            xF16[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * 0.25f);

        float[] rowDeq = new float[k];
        Half[] expected = new Half[n];
        fixed (byte* pW = weights)
        {
            for (int r = 0; r < n; r++)
            {
                Dequantize.ToFloat32((nint)(pW + r * (k / 32) * 34), k, QuantizationType.Q8_0, rowDeq);
                float acc = 0.0f;
                for (int i = 0; i < k; i++) acc += rowDeq[i] * (float)xF16[i];
                expected[r] = (Half)acc;
            }
        }

        foreach (var dev in devices)
        {
            dev.MakeCurrent();
            using var kernels = new HipKernels(dev.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dW, (nuint)wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, (nuint)(k * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dY, (nuint)(n * sizeof(Half))).ThrowOnError();

            try
            {
                fixed (byte* pW = weights) fixed (Half* pX = xF16)
                {
                    HipApi.hipMemcpy(dW, (nint)pW, (nuint)wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, (nuint)(k * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchMmvqGemvQ8_0Fdot2(dW, dX, dY, n, k, stream.Handle);
                stream.Synchronize();

                Half[] actual = new Half[n];
                fixed (Half* pY = actual)
                    HipApi.hipMemcpy((nint)pY, dY, (nuint)(n * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < n; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)actual[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.05f, MathF.Abs(expF) * 0.05f);
                    Assert.True(diff <= tol, $"{dev.Arch} MMVQ Q8_0 fdot2 row {r}: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dW); HipApi.hipFree(dX); HipApi.hipFree(dY);
            }
        }
    }

    [Fact]
    public unsafe void MmvqGemvQ6_K_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 64;
        int k = 512;
        var rng = new Random(789);

        long wBytes = Dequantize.RowByteSize(k, QuantizationType.Q6_K) * n;
        byte[] weights = new byte[wBytes];
        rng.NextBytes(weights);

        int numSb = k / 256;
        for (int r = 0; r < n; r++)
        {
            long rowOff = (long)r * numSb * 210;
            for (int sb = 0; sb < numSb; sb++)
            {
                long sbOff = rowOff + sb * 210;
                ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                weights[sbOff + 208] = (byte)(d & 0xFF);
                weights[sbOff + 209] = (byte)(d >> 8);
            }
        }

        Half[] xF16 = new Half[k];
        float[] xF32 = new float[k];
        for (int i = 0; i < k; i++)
        {
            float v = ((float)rng.NextDouble() * 2.0f - 1.0f) * 0.1f;
            xF16[i] = (Half)v;
            xF32[i] = (float)xF16[i];
        }

        float[] rowDeq = new float[k];
        Half[] expected = new Half[n];
        fixed (byte* pW = weights)
        {
            for (int r = 0; r < n; r++)
            {
                Dequantize.ToFloat32((nint)(pW + r * numSb * 210), k, QuantizationType.Q6_K, rowDeq);
                float acc = 0.0f;
                for (int i = 0; i < k; i++) acc += rowDeq[i] * xF32[i];
                expected[r] = (Half)acc;
            }
        }

        foreach (var dev in devices)
        {
            dev.MakeCurrent();
            using var kernels = new HipKernels(dev.Arch);
            using var stream = HipStream.Create();

            int q8_1Bytes = (k / 32) * 36;
            HipApi.hipMalloc(out nint dW, (nuint)wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, (nuint)(k * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dXQ8_1, (nuint)q8_1Bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dY, (nuint)(n * sizeof(Half))).ThrowOnError();

            try
            {
                fixed (byte* pW = weights) fixed (Half* pX = xF16)
                {
                    HipApi.hipMemcpy(dW, (nint)pW, (nuint)wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, (nuint)(k * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchQuantizeQ8_1F16(dX, dXQ8_1, k, stream.Handle);
                kernels.LaunchMmvqGemvQ6_K(dW, dXQ8_1, dY, n, k, stream.Handle);
                stream.Synchronize();

                Half[] actual = new Half[n];
                fixed (Half* pY = actual)
                    HipApi.hipMemcpy((nint)pY, dY, (nuint)(n * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < n; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)actual[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.35f, MathF.Abs(expF) * 0.10f);
                    Assert.True(diff <= tol, $"{dev.Arch} MMVQ Q6_K row {r}: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dW); HipApi.hipFree(dX); HipApi.hipFree(dXQ8_1); HipApi.hipFree(dY);
            }
        }
    }
}


