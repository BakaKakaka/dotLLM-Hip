using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Unit tests for dequantization and quantized GEMV kernels against CPU reference (H2.3).
/// Tests Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, and Q6_K across both physical GPUs
/// (RX 7900 XTX gfx1100 and RX 9070 XT gfx1201).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipDequantAndGemvTests
{
    private static byte[] CreateRandomQuantizedBytes(Random rng, int count, QuantizationType qt)
    {
        long byteSize = Dequantize.RowByteSize(count, qt);
        byte[] data = new byte[byteSize];
        rng.NextBytes(data);

        // Ensure valid float16 scales to prevent Inf/NaN in tests
        switch (qt)
        {
            case QuantizationType.Q8_0:
                for (int b = 0; b < count / 32; b++)
                {
                    ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                    data[b * 34] = (byte)(d & 0xFF);
                    data[b * 34 + 1] = (byte)(d >> 8);
                }
                break;

            case QuantizationType.Q4_0:
                for (int b = 0; b < count / 32; b++)
                {
                    ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                    data[b * 18] = (byte)(d & 0xFF);
                    data[b * 18 + 1] = (byte)(d >> 8);
                }
                break;

            case QuantizationType.Q5_0:
                for (int b = 0; b < count / 32; b++)
                {
                    ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                    data[b * 22] = (byte)(d & 0xFF);
                    data[b * 22 + 1] = (byte)(d >> 8);
                }
                break;

            case QuantizationType.Q4_K:
                for (int sb = 0; sb < count / 256; sb++)
                {
                    ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                    ushort dmin = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.02f + 0.005f));
                    data[sb * 144] = (byte)(d & 0xFF);
                    data[sb * 144 + 1] = (byte)(d >> 8);
                    data[sb * 144 + 2] = (byte)(dmin & 0xFF);
                    data[sb * 144 + 3] = (byte)(dmin >> 8);
                }
                break;

            case QuantizationType.Q5_K:
                for (int sb = 0; sb < count / 256; sb++)
                {
                    ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                    ushort dmin = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.02f + 0.005f));
                    data[sb * 176] = (byte)(d & 0xFF);
                    data[sb * 176 + 1] = (byte)(d >> 8);
                    data[sb * 176 + 2] = (byte)(dmin & 0xFF);
                    data[sb * 176 + 3] = (byte)(dmin >> 8);
                }
                break;

            case QuantizationType.Q6_K:
                for (int sb = 0; sb < count / 256; sb++)
                {
                    ushort d = BitConverter.HalfToUInt16Bits((Half)((float)rng.NextDouble() * 0.05f + 0.01f));
                    data[sb * 210 + 208] = (byte)(d & 0xFF);
                    data[sb * 210 + 209] = (byte)(d >> 8);
                }
                break;
        }

        return data;
    }

    [Theory]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q5_0)]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    public unsafe void Dequantize_MatchesCpuReference_AllFormats(QuantizationType qt)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int totalElements = 1024; // 32 blocks (Q8/Q4/Q5) or 4 superblocks (K-quants)
        var rng = new Random(42);
        byte[] quantBytes = CreateRandomQuantizedBytes(rng, totalElements, qt);

        // CPU reference: decode to float, convert to Half
        float[] cpuFloats = new float[totalElements];
        fixed (byte* pQ = quantBytes)
        {
            Dequantize.ToFloat32((nint)pQ, totalElements, qt, cpuFloats);
        }
        Half[] expected = new Half[totalElements];
        for (int i = 0; i < totalElements; i++) expected[i] = (Half)cpuFloats[i];

        nuint srcBytes = (nuint)quantBytes.Length;
        nuint dstBytes = (nuint)(totalElements * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dSrc, srcBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dDst, dstBytes).ThrowOnError();

            try
            {
                fixed (byte* pSrc = quantBytes)
                {
                    HipApi.hipMemcpy(dSrc, (nint)pSrc, srcBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchDequantToF16(dSrc, qt, dDst, totalElements, stream.Handle);
                stream.Synchronize();

                Half[] gpuResult = new Half[totalElements];
                fixed (Half* pDst = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pDst, dDst, dstBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int i = 0; i < totalElements; i++)
                {
                    float expF = (float)expected[i];
                    float actF = (float)gpuResult[i];
                    float diff = MathF.Abs(expF - actF);
                    Assert.True(diff <= 0.005f,
                        $"{device.Arch} Dequant({qt}) at index {i}: expected {expF}, got {actF}, diff {diff}");
                }
            }
            finally
            {
                HipApi.hipFree(dSrc);
                HipApi.hipFree(dDst);
            }
        }
    }

    [Theory]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q5_0)]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    public unsafe void QuantizedGemv_MatchesCpuReference_AllFormats(QuantizationType qt)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int n = 32;   // 32 output rows = 4 waves in 1 block
        int k = 512;  // 512 input columns = 16 blocks (Q8/Q4/Q5) or 2 superblocks (K-quants)
        var rng = new Random(42);

        int totalElements = n * k;
        byte[] quantWeights = CreateRandomQuantizedBytes(rng, totalElements, qt);

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++) x[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * 0.5f);

        // CPU reference: dequantize each row to float32, compute dot product with x as float, output as Half
        float[] rowDeq = new float[k];
        Half[] expected = new Half[n];
        long bytesPerRow = Dequantize.RowByteSize(k, qt);

        fixed (byte* pW = quantWeights)
        {
            for (int r = 0; r < n; r++)
            {
                nint rowPtr = (nint)(pW + r * bytesPerRow);
                Dequantize.ToFloat32(rowPtr, k, qt, rowDeq);

                float acc = 0.0f;
                for (int j = 0; j < k; j++)
                    acc += rowDeq[j] * (float)x[j];

                expected[r] = (Half)acc;
            }
        }

        nuint wBytes = (nuint)quantWeights.Length;
        nuint xBytes = (nuint)(k * sizeof(Half));
        nuint yBytes = (nuint)(n * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dW, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, xBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dY, yBytes).ThrowOnError();

            try
            {
                fixed (byte* pW = quantWeights)
                fixed (Half* pX = x)
                {
                    HipApi.hipMemcpy(dW, (nint)pW, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchQuantizedGemv(dW, qt, dX, dY, n, k, stream.Handle);
                stream.Synchronize();

                Half[] gpuResult = new Half[n];
                fixed (Half* pY = gpuResult)
                {
                    HipApi.hipMemcpy((nint)pY, dY, yBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int r = 0; r < n; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)gpuResult[r];
                    float diff = MathF.Abs(expF - actF);
                    // GEMV tolerance: accumulation of 512 products in float, cast to Half
                    Assert.True(diff <= 0.05f,
                        $"{device.Arch} QuantizedGemv({qt}) row {r}: expected {expF}, got {actF}, diff {diff}");
                }
            }
            finally
            {
                HipApi.hipFree(dW);
                HipApi.hipFree(dX);
                HipApi.hipFree(dY);
            }
        }
    }
}
