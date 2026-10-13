using System.Numerics.Tensors;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Tests rocBLAS GEMM and GEMV correctness against CPU references for required shapes (H1.4):
/// 1×4096×4096, 512×4096×14336, and 512×14336×4096.
/// Tests on all available supported discrete GPUs (gfx1100 and gfx1201).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipGemmTests
{
    private readonly ITestOutputHelper _output;

    public HipGemmTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableFact]
    public void LinearF16_Shape_1x4096x4096_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        foreach (var device in devices)
        {
            _output.WriteLine($"Testing 1x4096x4096 on {device}...");
            RunGemmTest(device, m: 1, k: 4096, n: 4096, checkAllRows: true);
        }
    }

    [SkippableFact]
    public void LinearF16_Shape_512x4096x14336_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        foreach (var device in devices)
        {
            _output.WriteLine($"Testing 512x4096x14336 on {device}...");
            // Run full GEMM on GPU, verify representative rows across the batch against CPU reference
            RunGemmTest(device, m: 512, k: 4096, n: 14336, checkAllRows: false);
        }
    }

    [SkippableFact]
    public void LinearF16_Shape_512x14336x4096_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var devices = HipDevice.GetSupportedDevices();
        foreach (var device in devices)
        {
            _output.WriteLine($"Testing 512x14336x4096 on {device}...");
            // Run full GEMM on GPU, verify representative rows across the batch against CPU reference
            RunGemmTest(device, m: 512, k: 14336, n: 4096, checkAllRows: false);
        }
    }

    [SkippableFact]
    public void GemvF16_SingleToken_MatchesLinearF16()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var device = HipDevice.ResolveDevice("hip");
        device.MakeCurrent();

        using var blas = HipBlasHandle.Create();
        using var stream = HipStream.Create();
        blas.SetStream(stream);

        int k = 1024, n = 512;
        var rng = new Random(1234);

        Half[] xHost = new Half[k];
        Half[] wHost = new Half[n * k];
        for (int i = 0; i < xHost.Length; i++) xHost[i] = (Half)(rng.NextSingle() * 0.2f - 0.1f);
        for (int i = 0; i < wHost.Length; i++) wHost[i] = (Half)(rng.NextSingle() * 0.2f - 0.1f);

        nuint xBytes = (nuint)(k * sizeof(short));
        nuint wBytes = (nuint)(n * k * sizeof(short));
        nuint yBytes = (nuint)(n * sizeof(short));

        HipApi.hipMalloc(out nint devX, xBytes).ThrowOnError();
        HipApi.hipMalloc(out nint devW, wBytes).ThrowOnError();
        HipApi.hipMalloc(out nint devY, yBytes).ThrowOnError();

        try
        {
            unsafe
            {
                fixed (Half* px = xHost, pw = wHost)
                {
                    HipApi.hipMemcpy(devX, (nint)px, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(devW, (nint)pw, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                HipGemm.GemvF16(blas.Handle, devW, devX, devY, n, k, stream.Handle);
                stream.Synchronize();

                Half[] yGpu = new Half[n];
                fixed (Half* py = yGpu)
                {
                    HipApi.hipMemcpy((nint)py, devY, yBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                // Verify against CPU dot product
                for (int j = 0; j < n; j++)
                {
                    float expected = 0;
                    for (int p = 0; p < k; p++)
                        expected += (float)xHost[p] * (float)wHost[j * k + p];

                    float diff = MathF.Abs(expected - (float)yGpu[j]);
                    Assert.True(diff < 0.25f, $"GEMV element {j}: expected {expected}, got {(float)yGpu[j]}, diff {diff}");
                }
            }
        }
        finally
        {
            HipApi.hipFree(devX);
            HipApi.hipFree(devW);
            HipApi.hipFree(devY);
        }
    }

    private unsafe void RunGemmTest(HipDevice device, int m, int k, int n, bool checkAllRows)
    {
        device.MakeCurrent();

        using var blas = HipBlasHandle.Create();
        using var stream = HipStream.Create();
        blas.SetStream(stream);

        var rng = new Random(42 + m + k + n);

        // Scale inputs reasonably to keep values well within FP16 dynamic range
        float scale = 0.5f / MathF.Sqrt(k);
        Half[] xHost = new Half[m * k];
        Half[] wHost = new Half[n * k];

        for (int i = 0; i < xHost.Length; i++)
            xHost[i] = (Half)(rng.NextSingle() * (2 * scale) - scale);
        for (int i = 0; i < wHost.Length; i++)
            wHost[i] = (Half)(rng.NextSingle() * (2 * scale) - scale);

        nuint xBytes = (nuint)((long)m * k * sizeof(short));
        nuint wBytes = (nuint)((long)n * k * sizeof(short));
        nuint yBytes = (nuint)((long)m * n * sizeof(short));

        HipApi.hipMalloc(out nint devX, xBytes).ThrowOnError();
        HipApi.hipMalloc(out nint devW, wBytes).ThrowOnError();
        HipApi.hipMalloc(out nint devY, yBytes).ThrowOnError();

        try
        {
            fixed (Half* px = xHost, pw = wHost)
            {
                HipApi.hipMemcpy(devX, (nint)px, xBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                HipApi.hipMemcpy(devW, (nint)pw, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
            }

            // Execute full rocBLAS GEMM on device
            HipGemm.LinearF16(blas.Handle, devX, devW, devY, m, k, n, stream.Handle);
            stream.Synchronize();

            // Download complete result from device
            Half[] yGpu = new Half[(long)m * n];
            fixed (Half* py = yGpu)
            {
                HipApi.hipMemcpy((nint)py, devY, yBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
            }

            // Determine rows to verify against CPU reference
            int[] rowsToCheck = checkAllRows
                ? Enumerable.Range(0, m).ToArray()
                : [0, 1, m / 4, m / 2, (3 * m) / 4, m - 1];

            float maxDiff = 0f;
            float maxRelDiff = 0f;

            // Pre-convert W to float rows on demand or cache
            float[] wRowFloat = new float[k];
            float[] xRowFloat = new float[k];

            foreach (int i in rowsToCheck)
            {
                for (int p = 0; p < k; p++)
                    xRowFloat[p] = (float)xHost[(long)i * k + p];

                for (int j = 0; j < n; j++)
                {
                    for (int p = 0; p < k; p++)
                        wRowFloat[p] = (float)wHost[(long)j * k + p];

                    float cpuSum = TensorPrimitives.Dot(
                        new ReadOnlySpan<float>(xRowFloat),
                        new ReadOnlySpan<float>(wRowFloat));

                    float gpuVal = (float)yGpu[(long)i * n + j];
                    float diff = MathF.Abs(cpuSum - gpuVal);
                    float relDiff = diff / (MathF.Abs(cpuSum) + 1e-4f);

                    if (diff > maxDiff) maxDiff = diff;
                    if (relDiff > maxRelDiff) maxRelDiff = relDiff;
                }
            }

            _output.WriteLine($"  {device.Arch} shape ({m}x{k}x{n}): maxDiff={maxDiff:F5}, maxRelDiff={maxRelDiff:P2} (checked {rowsToCheck.Length} rows)");

            // For FP16 output with FP32 accumulate, verify numerical proximity
            Assert.True(maxDiff < 0.25f,
                $"Max absolute difference {maxDiff:F5} on {device.Arch} shape ({m}x{k}x{n}) exceeded threshold 0.25f");
        }
        finally
        {
            HipApi.hipFree(devX);
            HipApi.hipFree(devW);
            HipApi.hipFree(devY);
        }
    }
}
