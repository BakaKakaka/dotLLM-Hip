using System.Globalization;

namespace DotLLM.Hip.Spikes;

/// <summary>
/// H0.3 - read bandwidth vs. working-set size (1 MB … 2 GB), normal vs. non-temporal loads.
/// Small working sets are read several passes per launch (≥ 256 MB per launch) so that L2 and
/// Infinity Cache (MALL) bandwidth is visible instead of launch overhead.
/// </summary>
internal static unsafe class BandwidthSpike
{
    private const long MB = 1L << 20;
    private const long MaxBytes = 2048 * MB;
    private const long MinBytesPerLaunch = 256 * MB;
    private const long MinBytesPerSample = 4096 * MB;
    private const uint Block = 256;

    public readonly record struct Point(long Bytes, double NormalGBs, double NonTemporalGBs);

    public static IReadOnlyList<Point> Run(SpikeDevice dev, Report r)
    {
        dev.MakeCurrent();
        nint fRead = dev.Function("spike_stream_read");
        nint fReadNt = dev.Function("spike_stream_read_nt");
        using var buffer = new DeviceBuffer((nuint)MaxBytes, 0);
        using var sink = new DeviceBuffer(64, 0);
        using var timer = new GpuTimer();
        uint maxGrid = (uint)dev.ComputeUnits * 16;

        // Pseudo-random contents (see spike_fill).
        nint dst = buffer.Ptr;
        ulong n16All = (ulong)MaxBytes / 16;
        void** fillArgs = stackalloc void*[2];
        fillArgs[0] = &dst;
        fillArgs[1] = &n16All;
        dev.Launch(dev.Function("spike_fill"), maxGrid, Block, fillArgs);
        dev.Synchronize();

        var points = new List<Point>();
        for (long bytes = MB; bytes <= MaxBytes; bytes *= 2)
        {
            double normal = Measure(dev, fRead, buffer.Ptr, bytes, sink.Ptr, maxGrid, timer);
            double nt = Measure(dev, fReadNt, buffer.Ptr, bytes, sink.Ptr, maxGrid, timer);
            points.Add(new Point(bytes, normal, nt));
        }

        r.Line($"#### {dev}");
        r.Line();
        r.Line("| Working set | Normal (GB/s) | Non-temporal (GB/s) | NT / normal |");
        r.Line("|---:|---:|---:|---:|");
        foreach (Point p in points)
        {
            string size = p.Bytes >= 1024 * MB ? $"{p.Bytes / (1024 * MB)} GB" : $"{p.Bytes / MB} MB";
            r.Line(string.Create(CultureInfo.InvariantCulture,
                $"| {size} | {p.NormalGBs:F0} | {p.NonTemporalGBs:F0} | {p.NonTemporalGBs / p.NormalGBs:P0} |"));
        }
        Point dram = points[^1];
        double peak = Math.Max(dram.NormalGBs, dram.NonTemporalGBs);
        r.Line();
        r.Line(string.Create(CultureInfo.InvariantCulture,
            $"DRAM reference (2 GB working set, best variant): **{peak:F0} GB/s**. Grid ≤ {maxGrid} x {Block}, 4 x 16 B loads in flight per lane."));
        r.Line();
        return points;
    }

    private static double Measure(SpikeDevice dev, nint function, nint src, long bytes, nint sinkPtr, uint maxGrid, GpuTimer timer)
    {
        ulong n16 = (ulong)bytes / 16;
        uint passes = (uint)Math.Max(1, MinBytesPerLaunch / bytes);
        long bytesPerLaunch = bytes * passes;
        int reps = (int)Math.Max(3, MinBytesPerSample / bytesPerLaunch);
        uint grid = (uint)Math.Min((n16 + Block - 1) / Block, maxGrid);

        void** args = stackalloc void*[4];
        args[0] = &src;
        args[1] = &n16;
        args[2] = &passes;
        args[3] = &sinkPtr;

        for (int i = 0; i < 2; i++)
            dev.Launch(function, grid, Block, args);
        dev.Synchronize();

        double best = 0;
        for (int sample = 0; sample < 3; sample++)
        {
            timer.Start(dev.Stream);
            for (int i = 0; i < reps; i++)
                dev.Launch(function, grid, Block, args);
            timer.Stop(dev.Stream);
            double ms = timer.ElapsedMs();
            best = Math.Max(best, bytesPerLaunch * (double)reps / (ms * 1e6));
        }
        return best;
    }
}
