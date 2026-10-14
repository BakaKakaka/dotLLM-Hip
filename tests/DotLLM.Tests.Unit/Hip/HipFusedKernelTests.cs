using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Unit tests for fused decode kernels (H3.1):
/// - LaunchFusedGateUpSwiglu
/// - LaunchFusedQkvGemv
/// Compared against CPU reference across both physical GPUs (RX 7900 XTX gfx1100 and RX 9070 XT gfx1201).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipFusedKernelTests
{
    private static byte[] CreateRandomQuantizedBytes(Random rng, int count, QuantizationType qt)
    {
        long byteSize = Dequantize.RowByteSize(count, qt);
        byte[] data = new byte[byteSize];
        rng.NextBytes(data);

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
    public unsafe void FusedGateUpSwiglu_MatchesCpuReference_Quantized(QuantizationType qt)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int intermediateSize = 64; // 64 rows = 8 waves
        int k = 512;              // 512 cols
        var rng = new Random(123);

        byte[] gateBytes = CreateRandomQuantizedBytes(rng, intermediateSize * k, qt);
        byte[] upBytes = CreateRandomQuantizedBytes(rng, intermediateSize * k, qt);

        Half[] gateBias = new Half[intermediateSize];
        Half[] upBias = new Half[intermediateSize];
        for (int i = 0; i < intermediateSize; i++)
        {
            gateBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
            upBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
        }

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * 0.05f);

        // CPU reference computation:
        long bytesPerRow = Dequantize.RowByteSize(k, qt);
        float[] gRowDeq = new float[k];
        float[] uRowDeq = new float[k];
        Half[] expected = new Half[intermediateSize];

        fixed (byte* pGate = gateBytes)
        fixed (byte* pUp = upBytes)
        {
            for (int r = 0; r < intermediateSize; r++)
            {
                Dequantize.ToFloat32((nint)(pGate + r * bytesPerRow), k, qt, gRowDeq);
                Dequantize.ToFloat32((nint)(pUp + r * bytesPerRow), k, qt, uRowDeq);

                float gSum = (float)gateBias[r];
                float uSum = (float)upBias[r];
                for (int j = 0; j < k; j++)
                {
                    float xVal = (float)x[j];
                    gSum += gRowDeq[j] * xVal;
                    uSum += uRowDeq[j] * xVal;
                }

                float silu = gSum / (1.0f + MathF.Exp(-gSum));
                expected[r] = (Half)(silu * uSum);
            }
        }

        nuint wBytes = (nuint)gateBytes.Length;
        nuint bBytes = (nuint)(intermediateSize * sizeof(Half));
        nuint xBytes = (nuint)(k * sizeof(Half));
        nuint outBytes = (nuint)(intermediateSize * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dGateW, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dUpW, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dGateB, bBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dUpB, bBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, xBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (byte* pG = gateBytes)
                fixed (byte* pU = upBytes)
                fixed (Half* pGb = gateBias)
                fixed (Half* pUb = upBias)
                fixed (Half* pX = x)
                {
                    HipApi.hipMemcpy(dGateW, (nint)pG, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dUpW, (nint)pU, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dGateB, (nint)pGb, bBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dUpB, (nint)pUb, bBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchFusedGateUpSwiglu(
                    dGateW, dUpW, qt,
                    dGateB, dUpB,
                    dX, dOut,
                    intermediateSize, k, stream.Handle);

                stream.Synchronize();

                Half[] actual = new Half[intermediateSize];
                fixed (Half* pAct = actual)
                {
                    HipApi.hipMemcpy((nint)pAct, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int r = 0; r < intermediateSize; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)actual[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.05f, MathF.Abs(expF) * 0.02f);
                    Assert.True(diff <= tol,
                        $"{device.Arch} FusedGateUpSwiglu({qt}) row {r}: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dGateW);
                HipApi.hipFree(dUpW);
                HipApi.hipFree(dGateB);
                HipApi.hipFree(dUpB);
                HipApi.hipFree(dX);
                HipApi.hipFree(dOut);
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
    public unsafe void FusedQkvGemv_MatchesCpuReference_Quantized(QuantizationType qt)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int dimQ = 64;
        int dimK = 32;
        int dimV = 32;
        int k = 512;
        var rng = new Random(456);

        byte[] qBytes = CreateRandomQuantizedBytes(rng, dimQ * k, qt);
        byte[] kBytes = CreateRandomQuantizedBytes(rng, dimK * k, qt);
        byte[] vBytes = CreateRandomQuantizedBytes(rng, dimV * k, qt);

        Half[] qBias = new Half[dimQ];
        Half[] kBias = new Half[dimK];
        Half[] vBias = new Half[dimV];
        for (int i = 0; i < dimQ; i++) qBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
        for (int i = 0; i < dimK; i++) kBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
        for (int i = 0; i < dimV; i++) vBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * 0.05f);

        // CPU reference computation:
        long bytesPerRow = Dequantize.RowByteSize(k, qt);
        float[] rowDeq = new float[k];

        Half[] expQ = new Half[dimQ];
        Half[] expK = new Half[dimK];
        Half[] expV = new Half[dimV];

        fixed (byte* pQ = qBytes)
        {
            for (int r = 0; r < dimQ; r++)
            {
                Dequantize.ToFloat32((nint)(pQ + r * bytesPerRow), k, qt, rowDeq);
                float sum = (float)qBias[r];
                for (int j = 0; j < k; j++) sum += rowDeq[j] * (float)x[j];
                expQ[r] = (Half)sum;
            }
        }
        fixed (byte* pK = kBytes)
        {
            for (int r = 0; r < dimK; r++)
            {
                Dequantize.ToFloat32((nint)(pK + r * bytesPerRow), k, qt, rowDeq);
                float sum = (float)kBias[r];
                for (int j = 0; j < k; j++) sum += rowDeq[j] * (float)x[j];
                expK[r] = (Half)sum;
            }
        }
        fixed (byte* pV = vBytes)
        {
            for (int r = 0; r < dimV; r++)
            {
                Dequantize.ToFloat32((nint)(pV + r * bytesPerRow), k, qt, rowDeq);
                float sum = (float)vBias[r];
                for (int j = 0; j < k; j++) sum += rowDeq[j] * (float)x[j];
                expV[r] = (Half)sum;
            }
        }

        nuint qWBytes = (nuint)qBytes.Length;
        nuint kWBytes = (nuint)kBytes.Length;
        nuint vWBytes = (nuint)vBytes.Length;
        nuint xBytes = (nuint)(k * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQW, qWBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dKW, kWBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dVW, vWBytes).ThrowOnError();

            HipApi.hipMalloc(out nint dQB, (nuint)(dimQ * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dKB, (nuint)(dimK * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dVB, (nuint)(dimV * sizeof(Half))).ThrowOnError();

            HipApi.hipMalloc(out nint dX, xBytes).ThrowOnError();

            HipApi.hipMalloc(out nint dQOut, (nuint)(dimQ * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dKOut, (nuint)(dimK * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dVOut, (nuint)(dimV * sizeof(Half))).ThrowOnError();

            try
            {
                fixed (byte* pQ = qBytes) fixed (byte* pK = kBytes) fixed (byte* pV = vBytes)
                fixed (Half* pQb = qBias) fixed (Half* pKb = kBias) fixed (Half* pVb = vBias)
                fixed (Half* pX = x)
                {
                    HipApi.hipMemcpy(dQW, (nint)pQ, qWBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dKW, (nint)pK, kWBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dVW, (nint)pV, vWBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();

                    HipApi.hipMemcpy(dQB, (nint)pQb, (nuint)(dimQ * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dKB, (nint)pKb, (nuint)(dimK * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dVB, (nint)pVb, (nuint)(dimV * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();

                    HipApi.hipMemcpy(dX, (nint)pX, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchFusedQkvGemv(
                    dQW, dKW, dVW, qt,
                    dQB, dKB, dVB,
                    dX, dQOut, dKOut, dVOut,
                    dimQ, dimK, dimV, k, stream.Handle);

                stream.Synchronize();

                Half[] actQ = new Half[dimQ];
                Half[] actK = new Half[dimK];
                Half[] actV = new Half[dimV];

                fixed (Half* pQ = actQ) HipApi.hipMemcpy((nint)pQ, dQOut, (nuint)(dimQ * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                fixed (Half* pK = actK) HipApi.hipMemcpy((nint)pK, dKOut, (nuint)(dimK * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                fixed (Half* pV = actV) HipApi.hipMemcpy((nint)pV, dVOut, (nuint)(dimV * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < dimQ; r++)
                {
                    float expF = (float)expQ[r];
                    float actF = (float)actQ[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.05f, MathF.Abs(expF) * 0.02f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedQkvGemv({qt}) Q[{r}]: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
                for (int r = 0; r < dimK; r++)
                {
                    float expF = (float)expK[r];
                    float actF = (float)actK[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.05f, MathF.Abs(expF) * 0.02f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedQkvGemv({qt}) K[{r}]: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
                for (int r = 0; r < dimV; r++)
                {
                    float expF = (float)expV[r];
                    float actF = (float)actV[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.05f, MathF.Abs(expF) * 0.02f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedQkvGemv({qt}) V[{r}]: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dQW); HipApi.hipFree(dKW); HipApi.hipFree(dVW);
                HipApi.hipFree(dQB); HipApi.hipFree(dKB); HipApi.hipFree(dVB);
                HipApi.hipFree(dX);
                HipApi.hipFree(dQOut); HipApi.hipFree(dKOut); HipApi.hipFree(dVOut);
            }
        }
    }

    [Theory]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q4_K)]
    public unsafe void FusedGateUpSwigluMmvq_MatchesCpuReference(QuantizationType qt)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int intermediateSize = 64;
        int k = 512;
        var rng = new Random(123);

        byte[] gateBytes = CreateRandomQuantizedBytes(rng, intermediateSize * k, qt);
        byte[] upBytes = CreateRandomQuantizedBytes(rng, intermediateSize * k, qt);

        Half[] gateBias = new Half[intermediateSize];
        Half[] upBias = new Half[intermediateSize];
        for (int i = 0; i < intermediateSize; i++)
        {
            gateBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
            upBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
        }

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * 0.05f);

        // CPU reference computation
        long bytesPerRow = Dequantize.RowByteSize(k, qt);
        float[] gRowDeq = new float[k];
        float[] uRowDeq = new float[k];
        Half[] expected = new Half[intermediateSize];

        fixed (byte* pGate = gateBytes)
        fixed (byte* pUp = upBytes)
        {
            for (int r = 0; r < intermediateSize; r++)
            {
                Dequantize.ToFloat32((nint)(pGate + r * bytesPerRow), k, qt, gRowDeq);
                Dequantize.ToFloat32((nint)(pUp + r * bytesPerRow), k, qt, uRowDeq);

                float gSum = (float)gateBias[r];
                float uSum = (float)upBias[r];
                for (int j = 0; j < k; j++)
                {
                    float xVal = (float)x[j];
                    gSum += gRowDeq[j] * xVal;
                    uSum += uRowDeq[j] * xVal;
                }

                float silu = gSum / (1.0f + MathF.Exp(-gSum));
                expected[r] = (Half)(silu * uSum);
            }
        }

        nuint wBytes = (nuint)gateBytes.Length;
        nuint bBytes = (nuint)(intermediateSize * sizeof(Half));
        nuint xBytes = (nuint)(k * sizeof(Half));
        nuint xQ8_1Bytes = (nuint)((k / 32) * 36);
        nuint outBytes = (nuint)(intermediateSize * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dGateW, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dUpW, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dGateB, bBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dUpB, bBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dX, xBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dXQ8_1, xQ8_1Bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (byte* pG = gateBytes)
                fixed (byte* pU = upBytes)
                fixed (Half* pGb = gateBias)
                fixed (Half* pUb = upBias)
                fixed (Half* pX = x)
                {
                    HipApi.hipMemcpy(dGateW, (nint)pG, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dUpW, (nint)pU, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dGateB, (nint)pGb, bBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dUpB, (nint)pUb, bBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dX, (nint)pX, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                // Quantize x -> x_q8_1
                kernels.LaunchQuantizeQ8_1F16(dX, dXQ8_1, k, stream.Handle);

                // Run MMVQ Fused Gate+Up
                kernels.LaunchFusedGateUpSwigluMmvq(
                    dGateW, dUpW, qt,
                    dGateB, dUpB,
                    dXQ8_1, dOut,
                    intermediateSize, k, stream.Handle);

                stream.Synchronize();

                Half[] actual = new Half[intermediateSize];
                fixed (Half* pAct = actual)
                    HipApi.hipMemcpy((nint)pAct, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < intermediateSize; r++)
                {
                    float expF = (float)expected[r];
                    float actF = (float)actual[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.2f, MathF.Abs(expF) * 0.15f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedGateUpSwigluMmvq({qt}) row {r}: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dGateW);
                HipApi.hipFree(dUpW);
                HipApi.hipFree(dGateB);
                HipApi.hipFree(dUpB);
                HipApi.hipFree(dX);
                HipApi.hipFree(dXQ8_1);
                HipApi.hipFree(dOut);
            }
        }
    }

    [Theory]
    [InlineData(QuantizationType.Q8_0)]
    [InlineData(QuantizationType.Q4_K)]
    public unsafe void FusedQkvGemvMmvq_MatchesCpuReference(QuantizationType qt)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int dimQ = 64;
        int dimK = 32;
        int dimV = 32;
        int k = 512;
        var rng = new Random(456);

        byte[] qBytes = CreateRandomQuantizedBytes(rng, dimQ * k, qt);
        byte[] kBytes = CreateRandomQuantizedBytes(rng, dimK * k, qt);
        byte[] vBytes = CreateRandomQuantizedBytes(rng, dimV * k, qt);

        Half[] qBias = new Half[dimQ];
        Half[] kBias = new Half[dimK];
        Half[] vBias = new Half[dimV];
        for (int i = 0; i < dimQ; i++) qBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
        for (int i = 0; i < dimK; i++) kBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));
        for (int i = 0; i < dimV; i++) vBias[i] = (Half)(((float)rng.NextDouble() * 0.2f - 0.1f));

        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
            x[i] = (Half)(((float)rng.NextDouble() * 2.0f - 1.0f) * 0.05f);

        // CPU reference
        long bytesPerRow = Dequantize.RowByteSize(k, qt);
        float[] rowDeq = new float[k];

        Half[] expQ = new Half[dimQ];
        fixed (byte* pQ = qBytes)
        {
            for (int r = 0; r < dimQ; r++)
            {
                Dequantize.ToFloat32((nint)(pQ + r * bytesPerRow), k, qt, rowDeq);
                float sum = (float)qBias[r];
                for (int j = 0; j < k; j++) sum += rowDeq[j] * (float)x[j];
                expQ[r] = (Half)sum;
            }
        }

        Half[] expK = new Half[dimK];
        fixed (byte* pK = kBytes)
        {
            for (int r = 0; r < dimK; r++)
            {
                Dequantize.ToFloat32((nint)(pK + r * bytesPerRow), k, qt, rowDeq);
                float sum = (float)kBias[r];
                for (int j = 0; j < k; j++) sum += rowDeq[j] * (float)x[j];
                expK[r] = (Half)sum;
            }
        }

        Half[] expV = new Half[dimV];
        fixed (byte* pV = vBytes)
        {
            for (int r = 0; r < dimV; r++)
            {
                Dequantize.ToFloat32((nint)(pV + r * bytesPerRow), k, qt, rowDeq);
                float sum = (float)vBias[r];
                for (int j = 0; j < k; j++) sum += rowDeq[j] * (float)x[j];
                expV[r] = (Half)sum;
            }
        }

        nuint qWBytes = (nuint)qBytes.Length;
        nuint kWBytes = (nuint)kBytes.Length;
        nuint vWBytes = (nuint)vBytes.Length;
        nuint xBytes = (nuint)(k * sizeof(Half));
        nuint xQ8_1Bytes = (nuint)((k / 32) * 36);

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQW, qWBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dKW, kWBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dVW, vWBytes).ThrowOnError();

            HipApi.hipMalloc(out nint dQB, (nuint)(dimQ * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dKB, (nuint)(dimK * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dVB, (nuint)(dimV * sizeof(Half))).ThrowOnError();

            HipApi.hipMalloc(out nint dX, xBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dXQ8_1, xQ8_1Bytes).ThrowOnError();

            HipApi.hipMalloc(out nint dQOut, (nuint)(dimQ * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dKOut, (nuint)(dimK * sizeof(Half))).ThrowOnError();
            HipApi.hipMalloc(out nint dVOut, (nuint)(dimV * sizeof(Half))).ThrowOnError();

            try
            {
                fixed (byte* pQ = qBytes) fixed (byte* pK = kBytes) fixed (byte* pV = vBytes)
                fixed (Half* pQb = qBias) fixed (Half* pKb = kBias) fixed (Half* pVb = vBias)
                fixed (Half* pX = x)
                {
                    HipApi.hipMemcpy(dQW, (nint)pQ, qWBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dKW, (nint)pK, kWBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dVW, (nint)pV, vWBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();

                    HipApi.hipMemcpy(dQB, (nint)pQb, (nuint)(dimQ * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dKB, (nint)pKb, (nuint)(dimK * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dVB, (nint)pVb, (nuint)(dimV * sizeof(Half)), HipApi.hipMemcpyHostToDevice).ThrowOnError();

                    HipApi.hipMemcpy(dX, (nint)pX, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchQuantizeQ8_1F16(dX, dXQ8_1, k, stream.Handle);

                kernels.LaunchFusedQkvGemvMmvq(
                    dQW, dKW, dVW, qt,
                    dQB, dKB, dVB,
                    dXQ8_1, dQOut, dKOut, dVOut,
                    dimQ, dimK, dimV, k, stream.Handle);

                stream.Synchronize();

                Half[] actQ = new Half[dimQ];
                Half[] actK = new Half[dimK];
                Half[] actV = new Half[dimV];

                fixed (Half* pQ = actQ) HipApi.hipMemcpy((nint)pQ, dQOut, (nuint)(dimQ * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                fixed (Half* pK = actK) HipApi.hipMemcpy((nint)pK, dKOut, (nuint)(dimK * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                fixed (Half* pV = actV) HipApi.hipMemcpy((nint)pV, dVOut, (nuint)(dimV * sizeof(Half)), HipApi.hipMemcpyDeviceToHost).ThrowOnError();

                for (int r = 0; r < dimQ; r++)
                {
                    float expF = (float)expQ[r];
                    float actF = (float)actQ[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.08f, MathF.Abs(expF) * 0.05f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedQkvGemvMmvq({qt}) Q[{r}]: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
                for (int r = 0; r < dimK; r++)
                {
                    float expF = (float)expK[r];
                    float actF = (float)actK[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.08f, MathF.Abs(expF) * 0.05f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedQkvGemvMmvq({qt}) K[{r}]: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
                for (int r = 0; r < dimV; r++)
                {
                    float expF = (float)expV[r];
                    float actF = (float)actV[r];
                    float diff = MathF.Abs(expF - actF);
                    float tol = MathF.Max(0.08f, MathF.Abs(expF) * 0.05f);
                    Assert.True(diff <= tol, $"{device.Arch} FusedQkvGemvMmvq({qt}) V[{r}]: exp {expF}, act {actF}, diff {diff} > tol {tol}");
                }
            }
            finally
            {
                HipApi.hipFree(dQW); HipApi.hipFree(dKW); HipApi.hipFree(dVW);
                HipApi.hipFree(dQB); HipApi.hipFree(dKB); HipApi.hipFree(dVB);
                HipApi.hipFree(dX); HipApi.hipFree(dXQ8_1);
                HipApi.hipFree(dQOut); HipApi.hipFree(dKOut); HipApi.hipFree(dVOut);
            }
        }
    }
}
