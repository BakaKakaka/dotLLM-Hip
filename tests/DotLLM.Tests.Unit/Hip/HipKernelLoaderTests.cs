using System.Runtime.InteropServices;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Tests loading embedded AOT code objects (<c>hsaco/{arch}/*.hsaco</c>) via <see cref="HipKernelLoader"/>
/// and executing smoke kernels (<c>smoke_dot4</c>, <c>smoke_wave</c>, <c>smoke_wmma</c>) on physical GPUs (H1.3).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipKernelLoaderTests
{
    private const int Wave = 32;

    [SkippableFact]
    public void EmbeddedSmokeKernel_RunsSuccessfully_OnAllSupportedGpus()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        foreach (var device in devices)
        {
            device.MakeCurrent();

            using var module = HipKernelLoader.LoadEmbeddedModule(device.Arch, "toolchain_smoke");
            Assert.NotEqual(0, module.Handle);

            nint fDot4 = module.GetFunction("smoke_dot4");
            nint fWave = module.GetFunction("smoke_wave");
            nint fWmma = module.GetFunction("smoke_wmma");

            using var stream = HipStream.Create();

            TestSmokeDot4(device, fDot4, stream);
            TestSmokeWave(device, fWave, stream);
            TestSmokeWmma(device, fWmma, stream);
        }
    }

    private static unsafe void TestSmokeDot4(HipDevice device, nint fDot4, HipStream stream)
    {
        int[] a = new int[Wave], b = new int[Wave], expected = new int[Wave], got = new int[Wave];
        for (int i = 0; i < Wave; i++)
        {
            int a0 = i * 7 - 100, a1 = -i * 3 + 50, a2 = i - 16, a3 = 127 - i * 8;
            int b0 = -128 + i * 5, b1 = i * 2 - 31, b2 = 99 - i * 6, b3 = (i % 5) - 2;
            a[i] = (a0 & 0xFF) | ((a1 & 0xFF) << 8) | ((a2 & 0xFF) << 16) | ((a3 & 0xFF) << 24);
            b[i] = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
            expected[i] = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }

        nuint bytes = (nuint)(Wave * sizeof(int));
        HipApi.hipMalloc(out nint dA, bytes).ThrowOnError();
        HipApi.hipMalloc(out nint dB, bytes).ThrowOnError();
        HipApi.hipMalloc(out nint dOut, bytes).ThrowOnError();

        try
        {
            fixed (int* pa = a, pb = b)
            {
                HipApi.hipMemcpy(dA, (nint)pa, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                HipApi.hipMemcpy(dB, (nint)pb, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
            }

            void* pA = &dA, pB = &dB, pOut = &dOut;
            void** kernelParams = stackalloc void*[3] { pA, pB, pOut };
            HipApi.hipModuleLaunchKernel(fDot4, 1, 1, 1, Wave, 1, 1, 0, stream.Handle, kernelParams, null).ThrowOnError();
            stream.Synchronize();

            fixed (int* pgot = got)
            {
                HipApi.hipMemcpy((nint)pgot, dOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
            }

            for (int i = 0; i < Wave; i++)
            {
                Assert.True(expected[i] == got[i],
                    $"{device.Arch} smoke_dot4 lane {i}: expected {expected[i]}, got {got[i]}");
            }
        }
        finally
        {
            HipApi.hipFree(dA);
            HipApi.hipFree(dB);
            HipApi.hipFree(dOut);
        }
    }

    private static unsafe void TestSmokeWave(HipDevice device, nint fWave, HipStream stream)
    {
        int[] wave = new int[2 * Wave];
        nuint bytes = (nuint)(wave.Length * sizeof(int));
        HipApi.hipMalloc(out nint dOut, bytes).ThrowOnError();

        try
        {
            void* pOut = &dOut;
            void** kernelParams = stackalloc void*[1] { pOut };
            HipApi.hipModuleLaunchKernel(fWave, 1, 1, 1, Wave, 1, 1, 0, stream.Handle, kernelParams, null).ThrowOnError();
            stream.Synchronize();

            fixed (int* pwave = wave)
            {
                HipApi.hipMemcpy((nint)pwave, dOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
            }

            for (int i = 0; i < Wave; i++)
            {
                Assert.Equal(496, wave[i]);        // sum of 0..31
                Assert.Equal(32, wave[Wave + i]);   // warpSize constant
            }
        }
        finally
        {
            HipApi.hipFree(dOut);
        }
    }

    private static unsafe void TestSmokeWmma(HipDevice device, nint fWmma, HipStream stream)
    {
        // 16x16x16 f16 -> f32, A = 1.0 (0x3C00), B = 2.0 (0x4000) => C = 32.0f
        short[] ha = new short[256], hb = new short[256];
        Array.Fill(ha, (short)0x3C00);
        Array.Fill(hb, (short)0x4000);
        float[] c = new float[256];

        nuint inBytes = 256 * sizeof(short);
        nuint outBytes = 256 * sizeof(float);
        HipApi.hipMalloc(out nint dA, inBytes).ThrowOnError();
        HipApi.hipMalloc(out nint dB, inBytes).ThrowOnError();
        HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

        try
        {
            fixed (short* pa = ha, pb = hb)
            {
                HipApi.hipMemcpy(dA, (nint)pa, inBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                HipApi.hipMemcpy(dB, (nint)pb, inBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
            }

            void* pA = &dA, pB = &dB, pOut = &dOut;
            void** kernelParams = stackalloc void*[3] { pA, pB, pOut };
            HipApi.hipModuleLaunchKernel(fWmma, 1, 1, 1, Wave, 1, 1, 0, stream.Handle, kernelParams, null).ThrowOnError();
            stream.Synchronize();

            fixed (float* pc = c)
            {
                HipApi.hipMemcpy((nint)pc, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
            }

            for (int i = 0; i < 256; i++)
            {
                Assert.True(MathF.Abs(c[i] - 32.0f) < 1e-4f,
                    $"{device.Arch} smoke_wmma index {i}: expected 32.0, got {c[i]}");
            }
        }
        finally
        {
            HipApi.hipFree(dA);
            HipApi.hipFree(dB);
            HipApi.hipFree(dOut);
        }
    }
}
