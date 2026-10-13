using System.Runtime.InteropServices;
using System.Text;

namespace DotLLM.Hip.Spikes;

/// <summary>
/// One supported discrete GPU with the spike module loaded and a non-blocking stream.
/// The architecture is determined by which embedded code object the runtime accepts, which
/// avoids the versioned <c>hipDeviceProp_t</c> (W9) in throwaway code.
/// </summary>
internal sealed unsafe class SpikeDevice : IDisposable
{
    private static readonly string[] s_archs = ["gfx1100", "gfx1201"];
    private void* _image;

    public int Ordinal { get; }
    public string Name { get; }
    public string Arch { get; }
    public int Wgp { get; }
    public int ComputeUnits => 2 * Wgp; // W11: multiProcessorCount reports WGPs.
    public int L2Bytes { get; }
    public nint Module { get; }
    public nint Stream { get; }

    private SpikeDevice(int ordinal, string name, string arch, int wgp, int l2, nint module, void* image)
    {
        Ordinal = ordinal; Name = name; Arch = arch; Wgp = wgp; L2Bytes = l2; Module = module; _image = image;
        Hip.Check(Hip.hipStreamCreateWithFlags(out nint stream, Hip.StreamNonBlocking));
        Stream = stream;
    }

    /// <summary>Enumerates visible devices, skipping iGPUs and unsupported architectures (D6).</summary>
    public static List<SpikeDevice> Enumerate(TextWriter log)
    {
        Hip.Check(Hip.hipInit(0));
        Hip.Check(Hip.hipRuntimeGetVersion(out int version));
        if (version / 10_000_000 != 7)
            throw new InvalidOperationException($"HIP runtime {version}: major 7 required (D4).");
        Hip.Check(Hip.hipGetDeviceCount(out int count));
        log.WriteLine($"HIP runtime {version / 10_000_000}.{version / 100_000 % 100}.{version % 100_000}, {count} visible device(s)");

        var devices = new List<SpikeDevice>();
        byte* buffer = stackalloc byte[256];
        for (int ordinal = 0; ordinal < count; ordinal++)
        {
            Hip.Check(Hip.hipSetDevice(ordinal));
            Hip.Check(Hip.hipDeviceGetName(buffer, 256, ordinal));
            string name = Marshal.PtrToStringAnsi((nint)buffer) ?? "?";
            Hip.Check(Hip.hipDeviceGetAttribute(out int integrated, Hip.AttrIntegrated, ordinal));
            Hip.Check(Hip.hipDeviceGetAttribute(out int wgp, Hip.AttrMultiprocessorCount, ordinal));
            Hip.Check(Hip.hipDeviceGetAttribute(out int l2, Hip.AttrL2CacheSize, ordinal));
            if (integrated != 0)
            {
                log.WriteLine($"  #{ordinal} {name}: integrated GPU - ignored");
                continue;
            }

            SpikeDevice? device = null;
            foreach (string arch in s_archs)
            {
                void* image = LoadEmbedded($"hsaco/{arch}/spike_kernels.hsaco", out _);
                if (image == null)
                    throw new InvalidOperationException("Spike kernels not built (hipcc missing at build time).");
                int err = Hip.hipModuleLoadData(out nint module, image);
                if (err == 0)
                {
                    device = new SpikeDevice(ordinal, name, arch, wgp, l2, module, image);
                    break;
                }
                NativeMemory.Free(image);
            }
            if (device is null)
            {
                log.WriteLine($"  #{ordinal} {name}: unsupported architecture - ignored");
                continue;
            }
            log.WriteLine($"  #{ordinal} {name}: {device.Arch}, {device.Wgp} WGP = {device.ComputeUnits} CU, L2 {l2 >> 20} MB");
            devices.Add(device);
        }
        return devices;
    }

    private static void* LoadEmbedded(string resource, out int length)
    {
        using Stream? s = typeof(SpikeDevice).Assembly.GetManifestResourceStream(resource);
        length = 0;
        if (s is null)
            return null;
        length = checked((int)s.Length);
        void* image = NativeMemory.Alloc((nuint)length);
        s.ReadExactly(new Span<byte>(image, length));
        return image;
    }

    /// <summary>W10: the current device is thread-local - call before every measurement.</summary>
    public void MakeCurrent() => Hip.Check(Hip.hipSetDevice(Ordinal));

    public nint Function(string name)
    {
        Hip.Check(Hip.hipModuleGetFunction(out nint f, Module, name));
        return f;
    }

    public void Launch(nint function, uint grid, uint block, void** args)
        => Hip.Check(Hip.hipModuleLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, Stream, args, null));

    public void Synchronize() => Hip.Check(Hip.hipStreamSynchronize(Stream));

    public override string ToString() => $"{Name} ({Arch}, HIP #{Ordinal})";

    public void Dispose()
    {
        if (Stream != 0) Hip.hipStreamDestroy(Stream);
        if (Module != 0) Hip.hipModuleUnload(Module);
        if (_image != null) { NativeMemory.Free(_image); _image = null; }
    }
}

/// <summary>Pair of HIP events measuring GPU time on a stream.</summary>
internal sealed class GpuTimer : IDisposable
{
    private readonly nint _start;
    private readonly nint _stop;

    public GpuTimer()
    {
        Hip.Check(Hip.hipEventCreate(out _start));
        Hip.Check(Hip.hipEventCreate(out _stop));
    }

    public void Start(nint stream) => Hip.Check(Hip.hipEventRecord(_start, stream));
    public void Stop(nint stream) => Hip.Check(Hip.hipEventRecord(_stop, stream));

    /// <summary>Blocks until the stop event completed and returns elapsed GPU milliseconds.</summary>
    public double ElapsedMs()
    {
        Hip.Check(Hip.hipEventSynchronize(_stop));
        Hip.Check(Hip.hipEventElapsedTime(out float ms, _start, _stop));
        return ms;
    }

    public void Dispose()
    {
        Hip.hipEventDestroy(_start);
        Hip.hipEventDestroy(_stop);
    }
}

/// <summary>Device buffer owned by the spike.</summary>
internal sealed class DeviceBuffer : IDisposable
{
    public nint Ptr { get; }
    public nuint Bytes { get; }

    public DeviceBuffer(nuint bytes, byte fill)
    {
        Hip.Check(Hip.hipMalloc(out nint p, bytes));
        Ptr = p;
        Bytes = bytes;
        Hip.Check(Hip.hipMemset(p, fill, bytes));
        Hip.Check(Hip.hipDeviceSynchronize());
    }

    public void Dispose() => Hip.hipFree(Ptr);
}

internal static class Stats
{
    public static double Percentile(double[] sorted, double p)
    {
        if (sorted.Length == 0) return double.NaN;
        double idx = p * (sorted.Length - 1);
        int lo = (int)Math.Floor(idx), hi = (int)Math.Ceiling(idx);
        return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
    }
}

/// <summary>Collects Markdown output for docs/HIP.md and echoes it to the console.</summary>
internal sealed class Report
{
    private readonly StringBuilder _md = new();

    public void Line(string text = "")
    {
        _md.AppendLine(text);
        Console.WriteLine(text);
    }

    public override string ToString() => _md.ToString();
}
