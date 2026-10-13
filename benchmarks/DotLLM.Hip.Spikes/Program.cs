using DotLLM.Hip.Spikes;

// H0.2 / H0.3 spikes - see docs/HIP_PLAN.md and docs/HIP.md.
//
//   dotnet run -c Release --project benchmarks/DotLLM.Hip.Spikes -- [launch|bandwidth|all] [--device gfx1100|gfx1201|all] [--out file.md]
//
// Defaults: "all"; launch overhead on gfx1100 (H0.2 is specified for the 7900 XTX), bandwidth on every supported card.

string mode = "all";
string? deviceFilter = null;
string? outPath = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--device": deviceFilter = args[++i]; break;
        case "--out": outPath = args[++i]; break;
        case "launch" or "bandwidth" or "all": mode = args[i]; break;
        default:
            Console.Error.WriteLine($"Unknown argument '{args[i]}'. Usage: [launch|bandwidth|all] [--device gfx1100|gfx1201|all] [--out file.md]");
            return 2;
    }
}

List<SpikeDevice> devices = SpikeDevice.Enumerate(Console.Out);
try
{
    if (devices.Count == 0)
    {
        Console.Error.WriteLine("No supported HIP device (gfx1100/gfx1201) visible.");
        return 1;
    }

    IEnumerable<SpikeDevice> Select(string defaultFilter)
    {
        string filter = deviceFilter ?? defaultFilter;
        return filter == "all" ? devices : devices.Where(d => d.Arch == filter);
    }

    var report = new Report();
    report.Line($"_Machine: {Environment.MachineName}, {DateTime.Now:yyyy-MM-dd HH:mm}, .NET {Environment.Version}_");
    report.Line();

    if (mode is "launch" or "all")
    {
        report.Line("### H0.2 – launch overhead");
        report.Line();
        foreach (SpikeDevice d in Select("gfx1100"))
            LaunchOverheadSpike.Run(d, report);
    }

    if (mode is "bandwidth" or "all")
    {
        report.Line("### H0.3 – read bandwidth and MALL");
        report.Line();
        foreach (SpikeDevice d in Select("all"))
            BandwidthSpike.Run(d, report);
    }

    if (outPath is not null)
    {
        File.WriteAllText(outPath, report.ToString());
        Console.WriteLine($"Report written to {Path.GetFullPath(outPath)}");
    }
    return 0;
}
finally
{
    foreach (SpikeDevice d in devices)
        d.Dispose();
}
