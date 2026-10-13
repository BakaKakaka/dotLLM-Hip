using System.Diagnostics;

namespace DotLLM.Hip.Spikes;

/// <summary>
/// H0.2 - launch overhead under WDDM (W2). Measures:
///   A. CPU time per hipModuleLaunchKernel (empty kernel, short batches - no queue back-pressure);
///   B. sustained launch throughput including back-pressure;
///   C. GPU back-to-back time per dependent kernel (pre-enqueued behind a spin kernel, events);
///   D. hipGraph capture + replay of the same chain (CPU cost of hipGraphLaunch and GPU time).
/// </summary>
internal static unsafe class LaunchOverheadSpike
{
    private const int Chain = 256;          // kernels per measured chain (≈ half a token at ~450 launches/token)
    private const int LaunchesPerToken = 450; // W2 estimate: ~14 launches/layer x 32 layers
    private const double SpinTargetMs = 30;  // < 100 ms kernel budget (W1)

    public static void Run(SpikeDevice dev, Report r)
    {
        dev.MakeCurrent();
        nint fEmpty = dev.Function("spike_empty");
        nint fSpin = dev.Function("spike_spin");
        using var sink = new DeviceBuffer(64, 0);
        using var timer = new GpuTimer();

        uint wideGrid = (uint)dev.ComputeUnits * 4;
        (string Label, uint Grid, uint Block)[] shapes =
        [
            ("1 x 32", 1, 32),
            ($"{wideGrid} x 256", wideGrid, 256),
        ];

        // Warm-up: JIT, code object upload, queue creation.
        for (int i = 0; i < 2000; i++)
            dev.Launch(fEmpty, 1, 32, null);
        dev.Synchronize();

        uint spinIters = CalibrateSpin(dev, fSpin, sink.Ptr, timer);

        r.Line($"#### {dev}");
        r.Line();
        r.Line("| Measurement | " + string.Join(" | ", shapes.Select(s => $"grid {s.Label}")) + " |");
        r.Line("|---|" + string.Concat(shapes.Select(_ => "---:|")));

        var cpuMedian = new double[shapes.Length];
        var cpuP90 = new double[shapes.Length];
        var sustained = new double[shapes.Length];
        var gpuPerKernel = new double[shapes.Length];
        var graphCpuPerNode = new double[shapes.Length];
        var graphGpuPerNode = new double[shapes.Length];
        var graphSustained = new double[shapes.Length];
        var enqueueVsSpin = new double[shapes.Length];

        for (int s = 0; s < shapes.Length; s++)
        {
            (_, uint grid, uint block) = shapes[s];

            // A. CPU cost per launch: 400 batches of 32 launches, synchronized between batches.
            const int batches = 400, perBatch = 32;
            var samples = new double[batches];
            for (int b = 0; b < batches; b++)
            {
                long t0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < perBatch; i++)
                    dev.Launch(fEmpty, grid, block, null);
                long t1 = Stopwatch.GetTimestamp();
                dev.Synchronize();
                samples[b] = (t1 - t0) * 1e6 / Stopwatch.Frequency / perBatch;
            }
            Array.Sort(samples);
            cpuMedian[s] = Stats.Percentile(samples, 0.5);
            cpuP90[s] = Stats.Percentile(samples, 0.9);

            // B. Sustained: 20 000 launches back to back, including queue back-pressure and final sync.
            const int sustainedN = 20_000;
            long u0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < sustainedN; i++)
                dev.Launch(fEmpty, grid, block, null);
            dev.Synchronize();
            sustained[s] = (Stopwatch.GetTimestamp() - u0) * 1e6 / Stopwatch.Frequency / sustainedN;

            // C. GPU back-to-back: spin kernel blocks the queue while the chain is enqueued.
            var gpuSamples = new double[10];
            double worstEnqueueMs = 0;
            for (int rep = 0; rep < gpuSamples.Length; rep++)
            {
                LaunchSpin(dev, fSpin, spinIters, sink.Ptr);
                long e0 = Stopwatch.GetTimestamp();
                timer.Start(dev.Stream);
                for (int i = 0; i < Chain; i++)
                    dev.Launch(fEmpty, grid, block, null);
                timer.Stop(dev.Stream);
                worstEnqueueMs = Math.Max(worstEnqueueMs, (Stopwatch.GetTimestamp() - e0) * 1e3 / Stopwatch.Frequency);
                gpuSamples[rep] = timer.ElapsedMs() * 1000.0 / Chain;
            }
            Array.Sort(gpuSamples);
            gpuPerKernel[s] = Stats.Percentile(gpuSamples, 0.5);
            enqueueVsSpin[s] = worstEnqueueMs / SpinTargetMs;

            // D. hipGraph: capture the same chain once, replay it.
            Hip.Check(Hip.hipStreamBeginCapture(dev.Stream, Hip.CaptureModeGlobal));
            for (int i = 0; i < Chain; i++)
                dev.Launch(fEmpty, grid, block, null);
            Hip.Check(Hip.hipStreamEndCapture(dev.Stream, out nint graph));
            Hip.Check(Hip.hipGraphInstantiate(out nint exec, graph, 0, null, 0));
            try
            {
                for (int i = 0; i < 5; i++)
                    Hip.Check(Hip.hipGraphLaunch(exec, dev.Stream));
                dev.Synchronize();

                var graphCpu = new double[50];
                for (int rep = 0; rep < graphCpu.Length; rep++)
                {
                    long g0 = Stopwatch.GetTimestamp();
                    Hip.Check(Hip.hipGraphLaunch(exec, dev.Stream));
                    long g1 = Stopwatch.GetTimestamp();
                    dev.Synchronize();
                    graphCpu[rep] = (g1 - g0) * 1e6 / Stopwatch.Frequency;
                }
                Array.Sort(graphCpu);
                graphCpuPerNode[s] = Stats.Percentile(graphCpu, 0.5) / Chain;

                var graphGpu = new double[10];
                for (int rep = 0; rep < graphGpu.Length; rep++)
                {
                    LaunchSpin(dev, fSpin, spinIters, sink.Ptr);
                    timer.Start(dev.Stream);
                    Hip.Check(Hip.hipGraphLaunch(exec, dev.Stream));
                    timer.Stop(dev.Stream);
                    graphGpu[rep] = timer.ElapsedMs() * 1000.0 / Chain;
                }
                Array.Sort(graphGpu);
                graphGpuPerNode[s] = Stats.Percentile(graphGpu, 0.5);

                // Sustained graph replay: same total node count as B, back to back.
                const int graphReplays = sustainedN / Chain;
                long h0 = Stopwatch.GetTimestamp();
                for (int rep = 0; rep < graphReplays; rep++)
                    Hip.Check(Hip.hipGraphLaunch(exec, dev.Stream));
                dev.Synchronize();
                graphSustained[s] = (Stopwatch.GetTimestamp() - h0) * 1e6 / Stopwatch.Frequency / (graphReplays * Chain);
            }
            finally
            {
                Hip.hipGraphExecDestroy(exec);
                Hip.hipGraphDestroy(graph);
            }
        }

        string Row(string label, double[] values, string fmt = "F2") =>
            $"| {label} | " + string.Join(" | ", values.Select(v => v.ToString(fmt, System.Globalization.CultureInfo.InvariantCulture))) + " |";

        r.Line(Row("A. CPU per launch, median (µs)", cpuMedian));
        r.Line(Row("A. CPU per launch, p90 (µs)", cpuP90));
        r.Line(Row("B. Sustained per launch incl. back-pressure (µs)", sustained));
        r.Line(Row($"C. GPU per dependent kernel, chain of {Chain} (µs)", gpuPerKernel));
        r.Line(Row($"D. hipGraphLaunch CPU per node, {Chain} nodes (µs)", graphCpuPerNode, "F3"));
        r.Line(Row("D. Graph GPU per node (µs)", graphGpuPerNode));
        r.Line(Row("D. Graph sustained per node incl. back-pressure (µs)", graphSustained));
        r.Line(Row($"Est. CPU per token, {LaunchesPerToken} launches, stream (ms)", cpuMedian.Select(v => v * LaunchesPerToken / 1000).ToArray(), "F3"));
        r.Line(Row($"Est. GPU launch gaps per token, {LaunchesPerToken} kernels, stream (ms)", gpuPerKernel.Select(v => v * LaunchesPerToken / 1000).ToArray(), "F3"));
        r.Line(Row($"Est. CPU per token, graph (ms)", graphCpuPerNode.Select(v => v * LaunchesPerToken / 1000).ToArray(), "F3"));
        r.Line(Row($"Est. GPU launch gaps per token, graph (ms)", graphGpuPerNode.Select(v => v * LaunchesPerToken / 1000).ToArray(), "F3"));
        r.Line();
        r.Line($"Spin barrier: {SpinTargetMs} ms ({spinIters} iterations). Worst chain enqueue / spin: " +
               string.Join(", ", enqueueVsSpin.Select(v => v.ToString("P0", System.Globalization.CultureInfo.InvariantCulture))) +
               " (must be < 100 % for C to be a pure GPU measurement).");
        r.Line();
    }

    private static void LaunchSpin(SpikeDevice dev, nint fSpin, uint iterations, nint sinkPtr)
    {
        void** args = stackalloc void*[2];
        args[0] = &iterations;
        args[1] = &sinkPtr;
        dev.Launch(fSpin, 1, 32, args);
    }

    private static uint CalibrateSpin(SpikeDevice dev, nint fSpin, nint sinkPtr, GpuTimer timer)
    {
        const uint probe = 1000;
        LaunchSpin(dev, fSpin, probe, sinkPtr); // warm-up
        dev.Synchronize();
        timer.Start(dev.Stream);
        LaunchSpin(dev, fSpin, probe, sinkPtr);
        timer.Stop(dev.Stream);
        double ms = timer.ElapsedMs();
        return (uint)Math.Clamp(probe * SpinTargetMs / Math.Max(ms, 1e-3), 100, 50_000_000);
    }
}
