using System.Globalization;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// H3.2 mmvq-GEMV bandwidth benchmark.
/// Measures 4096×14336 and 14336×4096 shapes for Q8_0 and Q4_K against H0.3 reference bandwidth (≥ 676 GB/s, 80% of 845 GB/s on RX 7900 XTX).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipMmvqBandwidthTests
{
    private readonly ITestOutputHelper _output;

    public HipMmvqBandwidthTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public unsafe void Benchmark_Mmvq_Bandwidth_4096x14336()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        (int n, int k)[] shapes = [
            (4096, 14336),
            (14336, 4096)
        ];

        foreach (var dev in devices)
        {
            dev.MakeCurrent();
            using var kernels = new HipKernels(dev.Arch);
            using var stream = HipStream.Create();

            _output.WriteLine($"### Device: {dev.Name} ({dev.Arch})");
            _output.WriteLine("| Shape | Format | Kernel | Time (µs) | Effective BW (GB/s) | % of 845 GB/s |");
            _output.WriteLine("|:---|:---|:---|---:|---:|---:|");

            foreach (var (n, k) in shapes)
            {
                // ── Q8_0 Benchmarks ──
                long q8_0Bytes = Dequantize.RowByteSize(k, QuantizationType.Q8_0) * n;
                int q8_1Bytes = (k / 32) * 36;

                HipApi.hipMalloc(out nint dWQ8, (nuint)q8_0Bytes).ThrowOnError();
                HipApi.hipMalloc(out nint dXF16, (nuint)(k * sizeof(Half))).ThrowOnError();
                HipApi.hipMalloc(out nint dXQ8_1, (nuint)q8_1Bytes).ThrowOnError();
                HipApi.hipMalloc(out nint dY, (nuint)(n * sizeof(Half))).ThrowOnError();

                try
                {
                    kernels.LaunchQuantizeQ8_1F16(dXF16, dXQ8_1, k, stream.Handle);
                    stream.Synchronize();

                    // Baseline GEMV Q8_0
                    double timeBaseQ8 = MeasureTimeUs(stream, () =>
                        kernels.LaunchQuantizedGemv(dWQ8, QuantizationType.Q8_0, dXF16, dY, n, k, stream.Handle));
                    double bwBaseQ8 = (q8_0Bytes / (timeBaseQ8 * 1e-6)) / 1e9;
                    double pctBaseQ8 = (bwBaseQ8 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q8_0 | Baseline GEMV | {timeBaseQ8:F1} | {bwBaseQ8:F1} | {pctBaseQ8:F1}% |"));

                    // MMVQ GEMV Q8_0 (sudot4, 16 B/lane)
                    double timeMmvqQ8 = MeasureTimeUs(stream, () =>
                        kernels.LaunchMmvqGemvQ8_0(dWQ8, dXQ8_1, dY, n, k, stream.Handle));
                    double bwMmvqQ8 = (q8_0Bytes / (timeMmvqQ8 * 1e-6)) / 1e9;
                    double pctMmvqQ8 = (bwMmvqQ8 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q8_0 | **MMVQ (sudot4)** | {timeMmvqQ8:F1} | **{bwMmvqQ8:F1}** | **{pctMmvqQ8:F1}%** |"));

                    // MMVQ GEMV Q8_0 (fdot2 variant)
                    double timeFdot2Q8 = MeasureTimeUs(stream, () =>
                        kernels.LaunchMmvqGemvQ8_0Fdot2(dWQ8, dXF16, dY, n, k, stream.Handle));
                    double bwFdot2Q8 = (q8_0Bytes / (timeFdot2Q8 * 1e-6)) / 1e9;
                    double pctFdot2Q8 = (bwFdot2Q8 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q8_0 | MMVQ (fdot2) | {timeFdot2Q8:F1} | {bwFdot2Q8:F1} | {pctFdot2Q8:F1}% |"));
                }
                finally
                {
                    HipApi.hipFree(dWQ8);
                    HipApi.hipFree(dXF16);
                    HipApi.hipFree(dXQ8_1);
                    HipApi.hipFree(dY);
                }

                // ── Q4_K Benchmarks ──
                int numSb = k / 256;
                long q4_KBytes = (long)n * numSb * 144;

                HipApi.hipMalloc(out nint dWQ4, (nuint)q4_KBytes).ThrowOnError();
                HipApi.hipMalloc(out nint dXF16K, (nuint)(k * sizeof(Half))).ThrowOnError();
                HipApi.hipMalloc(out nint dXQ8_1K, (nuint)q8_1Bytes).ThrowOnError();
                HipApi.hipMalloc(out nint dYK, (nuint)(n * sizeof(Half))).ThrowOnError();

                try
                {
                    kernels.LaunchQuantizeQ8_1F16(dXF16K, dXQ8_1K, k, stream.Handle);
                    stream.Synchronize();

                    // Baseline GEMV Q4_K
                    double timeBaseQ4 = MeasureTimeUs(stream, () =>
                        kernels.LaunchQuantizedGemv(dWQ4, QuantizationType.Q4_K, dXF16K, dYK, n, k, stream.Handle));
                    double bwBaseQ4 = (q4_KBytes / (timeBaseQ4 * 1e-6)) / 1e9;
                    double pctBaseQ4 = (bwBaseQ4 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q4_K | Baseline GEMV | {timeBaseQ4:F1} | {bwBaseQ4:F1} | {pctBaseQ4:F1}% |"));

                    // MMVQ GEMV Q4_K (sudot4, 16 B/lane)
                    double timeMmvqQ4 = MeasureTimeUs(stream, () =>
                        kernels.LaunchMmvqGemvQ4_K(dWQ4, dXQ8_1K, dYK, n, k, stream.Handle));
                    double bwMmvqQ4 = (q4_KBytes / (timeMmvqQ4 * 1e-6)) / 1e9;
                    double pctMmvqQ4 = (bwMmvqQ4 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q4_K | **MMVQ (sudot4)** | {timeMmvqQ4:F1} | **{bwMmvqQ4:F1}** | **{pctMmvqQ4:F1}%** |"));
                }
                finally
                {
                    HipApi.hipFree(dWQ4);
                    HipApi.hipFree(dXF16K);
                    HipApi.hipFree(dXQ8_1K);
                    HipApi.hipFree(dYK);
                }

                // ── Q6_K Benchmarks ──
                long q6_KBytes = (long)n * numSb * 210;

                HipApi.hipMalloc(out nint dWQ6, (nuint)q6_KBytes).ThrowOnError();
                HipApi.hipMalloc(out nint dXF16_6, (nuint)(k * sizeof(Half))).ThrowOnError();
                HipApi.hipMalloc(out nint dXQ8_1_6, (nuint)q8_1Bytes).ThrowOnError();
                HipApi.hipMalloc(out nint dY6, (nuint)(n * sizeof(Half))).ThrowOnError();

                try
                {
                    kernels.LaunchQuantizeQ8_1F16(dXF16_6, dXQ8_1_6, k, stream.Handle);
                    stream.Synchronize();

                    // Baseline GEMV Q6_K
                    double timeBaseQ6 = MeasureTimeUs(stream, () =>
                        kernels.LaunchQuantizedGemv(dWQ6, QuantizationType.Q6_K, dXF16_6, dY6, n, k, stream.Handle));
                    double bwBaseQ6 = (q6_KBytes / (timeBaseQ6 * 1e-6)) / 1e9;
                    double pctBaseQ6 = (bwBaseQ6 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q6_K | Baseline GEMV | {timeBaseQ6:F1} | {bwBaseQ6:F1} | {pctBaseQ6:F1}% |"));

                    // MMVQ GEMV Q6_K
                    double timeMmvqQ6 = MeasureTimeUs(stream, () =>
                        kernels.LaunchMmvqGemvQ6_K(dWQ6, dXQ8_1_6, dY6, n, k, stream.Handle));
                    double bwMmvqQ6 = (q6_KBytes / (timeMmvqQ6 * 1e-6)) / 1e9;
                    double pctMmvqQ6 = (bwMmvqQ6 / 845.0) * 100.0;
                    _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"| {n}×{k} | Q6_K | **MMVQ** | {timeMmvqQ6:F1} | **{bwMmvqQ6:F1}** | **{pctMmvqQ6:F1}%** |"));
                }
                finally
                {
                    HipApi.hipFree(dWQ6);
                    HipApi.hipFree(dXF16_6);
                    HipApi.hipFree(dXQ8_1_6);
                    HipApi.hipFree(dY6);
                }
            }
        }
    }

    private static double MeasureTimeUs(HipStream stream, Action launch)
    {
        const int Warmup = 20;
        const int Runs = 100;

        for (int i = 0; i < Warmup; i++)
            launch();
        stream.Synchronize();

        HipApi.hipEventCreate(out nint start).ThrowOnError();
        HipApi.hipEventCreate(out nint stop).ThrowOnError();

        try
        {
            HipApi.hipEventRecord(start, stream.Handle).ThrowOnError();
            for (int i = 0; i < Runs; i++)
                launch();
            HipApi.hipEventRecord(stop, stream.Handle).ThrowOnError();
            HipApi.hipEventSynchronize(stop).ThrowOnError();

            HipApi.hipEventElapsedTime(out float ms, start, stop).ThrowOnError();
            return (ms * 1000.0) / Runs;
        }
        finally
        {
            HipApi.hipEventDestroy(start);
            HipApi.hipEventDestroy(stop);
        }
    }
}
