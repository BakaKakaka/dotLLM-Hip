using System.Runtime.InteropServices;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Tests RDNA common primitives and wave probe kernel (H2.1).
/// Verifies warpSize == 32 on both physical GPUs (RX 7900 XTX gfx1100 and RX 9070 XT gfx1201)
/// and tests wave/block reductions.
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class WaveProbeTests
{
    private const int Wave = 32;

    [SkippableFact]
    public void WarpSize_IsThirtyTwo_OnAllSupportedGpus()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        foreach (var device in devices)
        {
            device.MakeCurrent();

            using var module = HipKernelLoader.LoadEmbeddedModule(device.Arch, "wave_probe");
            nint fWarpSize = module.GetFunction("probe_warp_size");

            using var stream = HipStream.Create();

            nuint bytes = 2 * sizeof(int);
            HipApi.hipMalloc(out nint dOut, bytes).ThrowOnError();

            try
            {
                unsafe
                {
                    void* pOut = &dOut;
                    void*[] kernelParams = [pOut];
                    fixed (void** pParams = kernelParams)
                    {
                        HipApi.hipModuleLaunchKernel(fWarpSize, 1, 1, 1, Wave, 1, 1, 0, stream.Handle, pParams, null).ThrowOnError();
                    }
                    stream.Synchronize();

                    int[] hostOut = new int[2];
                    fixed (int* pHost = hostOut)
                    {
                        HipApi.hipMemcpy((nint)pHost, dOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    }

                    Assert.True(hostOut[0] == 32, $"{device.Arch}: warpSize must be 32, got {hostOut[0]}");
                    Assert.True(hostOut[1] == 32, $"{device.Arch}: WAVE constant must be 32, got {hostOut[1]}");
                }
            }
            finally
            {
                HipApi.hipFree(dOut);
            }
        }
    }

    [SkippableFact]
    public void WaveReductions_CalculateCorrectly_OnAllSupportedGpus()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        float[] input = new float[Wave];
        float expectedSum = 0.0f;
        float expectedMax = float.MinValue;
        for (int i = 0; i < Wave; i++)
        {
            input[i] = (i + 1) * 1.5f;
            expectedSum += input[i];
            if (input[i] > expectedMax) expectedMax = input[i];
        }

        nuint bytes = (nuint)(Wave * sizeof(float));

        foreach (var device in devices)
        {
            device.MakeCurrent();

            using var module = HipKernelLoader.LoadEmbeddedModule(device.Arch, "wave_probe");
            nint fWaveReductions = module.GetFunction("probe_wave_reductions");
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dIn, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dSum, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dMax, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dAllSum, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dAllMax, bytes).ThrowOnError();

            try
            {
                unsafe
                {
                    fixed (float* pIn = input)
                    {
                        HipApi.hipMemcpy(dIn, (nint)pIn, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    }

                    void* pInArg = &dIn, pSumArg = &dSum, pMaxArg = &dMax, pAllSumArg = &dAllSum, pAllMaxArg = &dAllMax;
                    void*[] kernelParams = [pInArg, pSumArg, pMaxArg, pAllSumArg, pAllMaxArg];
                    fixed (void** pParams = kernelParams)
                    {
                        HipApi.hipModuleLaunchKernel(fWaveReductions, 1, 1, 1, Wave, 1, 1, 0, stream.Handle, pParams, null).ThrowOnError();
                    }
                    stream.Synchronize();

                    float[] hSum = new float[Wave], hMax = new float[Wave], hAllSum = new float[Wave], hAllMax = new float[Wave];
                    fixed (float* ps = hSum, pm = hMax, pas = hAllSum, pam = hAllMax)
                    {
                        HipApi.hipMemcpy((nint)ps, dSum, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)pm, dMax, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)pas, dAllSum, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)pam, dAllMax, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    }

                    Assert.True(MathF.Abs(hSum[0] - expectedSum) < 1e-3f, $"{device.Arch}: wave_sum lane 0 expected {expectedSum}, got {hSum[0]}");
                    Assert.True(MathF.Abs(hMax[0] - expectedMax) < 1e-3f, $"{device.Arch}: wave_max lane 0 expected {expectedMax}, got {hMax[0]}");

                    for (int i = 0; i < Wave; i++)
                    {
                        Assert.True(MathF.Abs(hAllSum[i] - expectedSum) < 1e-3f, $"{device.Arch}: wave_all_sum lane {i} expected {expectedSum}, got {hAllSum[i]}");
                        Assert.True(MathF.Abs(hAllMax[i] - expectedMax) < 1e-3f, $"{device.Arch}: wave_all_max lane {i} expected {expectedMax}, got {hAllMax[i]}");
                    }
                }
            }
            finally
            {
                HipApi.hipFree(dIn);
                HipApi.hipFree(dSum);
                HipApi.hipFree(dMax);
                HipApi.hipFree(dAllSum);
                HipApi.hipFree(dAllMax);
            }
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void BlockReductions_CalculateCorrectly_OnAllSupportedGpus(int blockSize)
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        float[] input = new float[blockSize];
        float expectedSum = 0.0f;
        float expectedMax = float.MinValue;
        for (int i = 0; i < blockSize; i++)
        {
            input[i] = (i + 1) * 0.25f;
            expectedSum += input[i];
            if (input[i] > expectedMax) expectedMax = input[i];
        }

        nuint inBytes = (nuint)(blockSize * sizeof(float));
        nuint scalarBytes = sizeof(float);

        foreach (var device in devices)
        {
            device.MakeCurrent();

            using var module = HipKernelLoader.LoadEmbeddedModule(device.Arch, "wave_probe");
            nint fBlockReductions = module.GetFunction("probe_block_reductions");
            using var stream = HipStream.Create();

            HipApi.hipMalloc(out nint dIn, inBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dReduceSum, scalarBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dReduceMax, scalarBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dAllSum, inBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dAllMax, inBytes).ThrowOnError();

            try
            {
                unsafe
                {
                    fixed (float* pIn = input)
                    {
                        HipApi.hipMemcpy(dIn, (nint)pIn, inBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    }

                    int nArg = blockSize;
                    void* pInArg = &dIn, pRSumArg = &dReduceSum, pRMaxArg = &dReduceMax, pASumArg = &dAllSum, pAMaxArg = &dAllMax, pNArg = &nArg;
                    void*[] kernelParams = [pInArg, pRSumArg, pRMaxArg, pASumArg, pAMaxArg, pNArg];
                    fixed (void** pParams = kernelParams)
                    {
                        HipApi.hipModuleLaunchKernel(fBlockReductions, 1, 1, 1, (uint)blockSize, 1, 1, 0, stream.Handle, pParams, null).ThrowOnError();
                    }
                    stream.Synchronize();

                    float[] hRSum = new float[1], hRMax = new float[1];
                    float[] hAllSum = new float[blockSize], hAllMax = new float[blockSize];

                    fixed (float* prs = hRSum, prm = hRMax, pas = hAllSum, pam = hAllMax)
                    {
                        HipApi.hipMemcpy((nint)prs, dReduceSum, scalarBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)prm, dReduceMax, scalarBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)pas, dAllSum, inBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                        HipApi.hipMemcpy((nint)pam, dAllMax, inBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    }

                    Assert.True(MathF.Abs(hRSum[0] - expectedSum) < 1e-2f, $"{device.Arch} block={blockSize}: reduce_sum expected {expectedSum}, got {hRSum[0]}");
                    Assert.True(MathF.Abs(hRMax[0] - expectedMax) < 1e-3f, $"{device.Arch} block={blockSize}: reduce_max expected {expectedMax}, got {hRMax[0]}");

                    for (int i = 0; i < blockSize; i++)
                    {
                        Assert.True(MathF.Abs(hAllSum[i] - expectedSum) < 1e-2f, $"{device.Arch} block={blockSize}: all_sum thread {i} expected {expectedSum}, got {hAllSum[i]}");
                        Assert.True(MathF.Abs(hAllMax[i] - expectedMax) < 1e-3f, $"{device.Arch} block={blockSize}: all_max thread {i} expected {expectedMax}, got {hAllMax[i]}");
                    }
                }
            }
            finally
            {
                HipApi.hipFree(dIn);
                HipApi.hipFree(dReduceSum);
                HipApi.hipFree(dReduceMax);
                HipApi.hipFree(dAllSum);
                HipApi.hipFree(dAllMax);
            }
        }
    }
}
