using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Unit tests for tiled attention kernel against CPU reference (H2.4).
/// Tests FP16 and FP32 implementations across both GPUs (gfx1100 and gfx1201),
/// verifying causal masking, GQA head grouping, sliding window, and multi-tile online softmax.
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipAttentionTests
{
    private const int NumHeads = 9;
    private const int NumKvHeads = 3;
    private const int HeadDim = 64;

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

    [SkippableFact]
    public unsafe void AttentionF32_Decode_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqQ = 1;
        int seqKv = 4;
        int posOffset = 3;
        var rng = new Random(42);

        float[] q = RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        float[] k = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        float[] v = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] cpuOut = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q, k, v, cpuOut, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        nuint qBytes = (nuint)(q.Length * sizeof(float));
        nuint kBytes = (nuint)(k.Length * sizeof(float));
        nuint vBytes = (nuint)(v.Length * sizeof(float));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (float* pQ = q, pK = k, pV = v)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttentionF32(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, stream.Handle);
                stream.Synchronize();

                float[] gpuOut = new float[cpuOut.Length];
                fixed (float* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("AttentionF32_Decode", cpuOut, gpuOut, 1e-4f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void AttentionF32_Prefill_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqQ = 4;
        int seqKv = 4;
        int posOffset = 0;
        var rng = new Random(123);

        float[] q = RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        float[] k = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        float[] v = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] cpuOut = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q, k, v, cpuOut, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);

        nuint qBytes = (nuint)(q.Length * sizeof(float));
        nuint kBytes = (nuint)(k.Length * sizeof(float));
        nuint vBytes = (nuint)(v.Length * sizeof(float));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (float* pQ = q, pK = k, pV = v)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttentionF32(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, stream.Handle);
                stream.Synchronize();

                float[] gpuOut = new float[cpuOut.Length];
                fixed (float* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("AttentionF32_Prefill", cpuOut, gpuOut, 1e-4f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void AttentionF32_SlidingWindow_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqQ = 8;
        int seqKv = 8;
        int posOffset = 0;
        int slidingWindow = 3;
        var rng = new Random(77);

        float[] q = RandomF32(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        float[] k = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        float[] v = RandomF32(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] cpuOut = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q, k, v, cpuOut, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, slidingWindow);

        nuint qBytes = (nuint)(q.Length * sizeof(float));
        nuint kBytes = (nuint)(k.Length * sizeof(float));
        nuint vBytes = (nuint)(v.Length * sizeof(float));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (float* pQ = q, pK = k, pV = v)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttentionF32(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, slidingWindow, stream.Handle);
                stream.Synchronize();

                float[] gpuOut = new float[cpuOut.Length];
                fixed (float* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareFloats("AttentionF32_SlidingWindow", cpuOut, gpuOut, 5e-4f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void AttentionF16_Decode_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqQ = 1;
        int seqKv = 4;
        int posOffset = 3;
        var rng = new Random(42);

        Half[] q16 = RandomF16(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        Half[] k16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        Half[] v16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] q32 = Array.ConvertAll(q16, h => (float)h);
        float[] k32 = Array.ConvertAll(k16, h => (float)h);
        float[] v32 = Array.ConvertAll(v16, h => (float)h);

        float[] cpuOutF32 = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q32, k32, v32, cpuOutF32, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);
        Half[] cpuOut = Array.ConvertAll(cpuOutF32, f => (Half)f);

        nuint qBytes = (nuint)(q16.Length * sizeof(Half));
        nuint kBytes = (nuint)(k16.Length * sizeof(Half));
        nuint vBytes = (nuint)(v16.Length * sizeof(Half));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = q16, pK = k16, pV = v16)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttention(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, stream.Handle);
                stream.Synchronize();

                Half[] gpuOut = new Half[cpuOut.Length];
                fixed (Half* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("AttentionF16_Decode", cpuOut, gpuOut, 0.01f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void AttentionF16_Prefill_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqQ = 4;
        int seqKv = 4;
        int posOffset = 0;
        var rng = new Random(123);

        Half[] q16 = RandomF16(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        Half[] k16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        Half[] v16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] q32 = Array.ConvertAll(q16, h => (float)h);
        float[] k32 = Array.ConvertAll(k16, h => (float)h);
        float[] v32 = Array.ConvertAll(v16, h => (float)h);

        float[] cpuOutF32 = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q32, k32, v32, cpuOutF32, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);
        Half[] cpuOut = Array.ConvertAll(cpuOutF32, f => (Half)f);

        nuint qBytes = (nuint)(q16.Length * sizeof(Half));
        nuint kBytes = (nuint)(k16.Length * sizeof(Half));
        nuint vBytes = (nuint)(v16.Length * sizeof(Half));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = q16, pK = k16, pV = v16)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttention(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, stream.Handle);
                stream.Synchronize();

                Half[] gpuOut = new Half[cpuOut.Length];
                fixed (Half* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("AttentionF16_Prefill", cpuOut, gpuOut, 0.01f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void AttentionF16_MultiTile_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        // 300 KV tokens crosses the TILE_KV = 256 boundary into 2 tiles!
        int seqQ = 1;
        int seqKv = 300;
        int posOffset = 299;
        var rng = new Random(999);

        Half[] q16 = RandomF16(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        Half[] k16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        Half[] v16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] q32 = Array.ConvertAll(q16, h => (float)h);
        float[] k32 = Array.ConvertAll(k16, h => (float)h);
        float[] v32 = Array.ConvertAll(v16, h => (float)h);

        float[] cpuOutF32 = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q32, k32, v32, cpuOutF32, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset);
        Half[] cpuOut = Array.ConvertAll(cpuOutF32, f => (Half)f);

        nuint qBytes = (nuint)(q16.Length * sizeof(Half));
        nuint kBytes = (nuint)(k16.Length * sizeof(Half));
        nuint vBytes = (nuint)(v16.Length * sizeof(Half));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = q16, pK = k16, pV = v16)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttention(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, 0, stream.Handle);
                stream.Synchronize();

                Half[] gpuOut = new Half[cpuOut.Length];
                fixed (Half* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("AttentionF16_MultiTile", cpuOut, gpuOut, 0.015f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public unsafe void AttentionF16_SlidingWindow_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        int seqQ = 8;
        int seqKv = 8;
        int posOffset = 0;
        int slidingWindow = 3;
        var rng = new Random(77);

        Half[] q16 = RandomF16(rng, seqQ * NumHeads * HeadDim, scale: 0.1f);
        Half[] k16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);
        Half[] v16 = RandomF16(rng, seqKv * NumKvHeads * HeadDim, scale: 0.1f);

        float[] q32 = Array.ConvertAll(q16, h => (float)h);
        float[] k32 = Array.ConvertAll(k16, h => (float)h);
        float[] v32 = Array.ConvertAll(v16, h => (float)h);

        float[] cpuOutF32 = new float[seqQ * NumHeads * HeadDim];
        Attention.Execute(q32, k32, v32, cpuOutF32, seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, slidingWindow);
        Half[] cpuOut = Array.ConvertAll(cpuOutF32, f => (Half)f);

        nuint qBytes = (nuint)(q16.Length * sizeof(Half));
        nuint kBytes = (nuint)(k16.Length * sizeof(Half));
        nuint vBytes = (nuint)(v16.Length * sizeof(Half));
        nuint outBytes = (nuint)(cpuOut.Length * sizeof(Half));

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var kernels = new HipKernels(device.Arch);
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = q16, pK = k16, pV = v16)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchAttention(dQ, dK, dV, dOut,
                    seqQ, seqKv, NumHeads, NumKvHeads, HeadDim, posOffset, slidingWindow, stream.Handle);
                stream.Synchronize();

                Half[] gpuOut = new Half[cpuOut.Length];
                fixed (Half* pOut = gpuOut)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                CompareHalves("AttentionF16_SlidingWindow", cpuOut, gpuOut, 0.01f, device.Arch);
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dOut);
            }
        }
    }
}
