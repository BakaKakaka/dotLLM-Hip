using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Raw device information used during enumeration and mock tests.
/// </summary>
public readonly record struct HipDeviceInfo(
    int Ordinal,
    string Name,
    string Arch,
    bool IsIntegrated,
    long TotalMemoryBytes,
    int Wgp,
    long L2CacheBytes,
    int WarpSize);

/// <summary>
/// Represents a supported discrete AMD GPU device (RDNA3 <c>gfx1100</c> or RDNA4 <c>gfx1201</c>).
/// Disregards integrated GPUs (<c>isIntegrated=1</c>) and unsupported architectures (D6, D9).
/// </summary>
public sealed class HipDevice
{
    /// <summary>Primary target architecture (Radeon RX 7900 XTX).</summary>
    public const string PrimaryArch = "gfx1100";

    /// <summary>Secondary target architecture (Radeon RX 9070 XT).</summary>
    public const string SecondaryArch = "gfx1201";

    /// <summary>Architectures supported by DotLLM.Hip (D6).</summary>
    public static readonly string[] SupportedArchitectures = [PrimaryArch, SecondaryArch];

    /// <summary>HIP device ordinal (0-based).</summary>
    public int Ordinal { get; }

    /// <summary>Device name reported by driver (e.g. "AMD Radeon RX 7900 XTX").</summary>
    public string Name { get; }

    /// <summary>Target architecture string (e.g. "gfx1100" or "gfx1201").</summary>
    public string Arch { get; }

    /// <summary>Total dedicated VRAM in bytes.</summary>
    public long TotalMemoryBytes { get; }

    /// <summary>Number of Workgroup Processors (WGP) reported by multiProcessorCount (W11).</summary>
    public int Wgp { get; }

    /// <summary>
    /// Total Compute Units (CU). W11: multiProcessorCount reports WGPs, so CU = 2 * WGP.
    /// </summary>
    public int ComputeUnits => 2 * Wgp;

    /// <summary>L2 cache size in bytes.</summary>
    public long L2CacheBytes { get; }

    /// <summary>Wavefront size (always 32 on RDNA3/RDNA4).</summary>
    public int WarpSize { get; }

    /// <summary>Total device VRAM formatted as a string (e.g. "24.0 GB").</summary>
    public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F2} GB";

    /// <summary>L2 cache size formatted in megabytes.</summary>
    public string L2CacheFormatted => $"{L2CacheBytes >> 20} MB";

    internal HipDevice(HipDeviceInfo info)
    {
        Ordinal = info.Ordinal;
        Name = info.Name;
        Arch = info.Arch;
        TotalMemoryBytes = info.TotalMemoryBytes;
        Wgp = info.Wgp;
        L2CacheBytes = info.L2CacheBytes;
        WarpSize = info.WarpSize;
    }

    /// <summary>
    /// Sets this device as the current thread's active HIP device (W10).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MakeCurrent()
    {
        HipApi.hipSetDevice(Ordinal).ThrowOnError();
    }

    /// <summary>
    /// Checks whether HIP is available on this system (Windows, SDK configured, kernels built, supported GPU).
    /// </summary>
    public static bool IsAvailable() => IsAvailable(out _);

    /// <summary>
    /// Checks whether HIP is available, providing a reason if unavailable.
    /// </summary>
    public static bool IsAvailable(out string? reason)
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            reason = "HIP backend is only supported on Windows.";
            return false;
        }

        if (!HipLibraryResolver.TryGetHipPath(out _, out string? pathError))
        {
            reason = pathError ?? "HIP_PATH is not configured.";
            return false;
        }

        try
        {
            HipLibraryResolver.EnsureHipLoaded();
        }
        catch (Exception ex)
        {
            reason = $"HIP runtime could not be loaded: {ex.Message}";
            return false;
        }

        if (!HipKernelLoader.AreKernelsAvailable(out string kernelReason))
        {
            reason = kernelReason;
            return false;
        }

        try
        {
            var supported = GetSupportedDevices();
            if (supported.Count == 0)
            {
                reason = "No supported discrete AMD HIP GPU (gfx1100 or gfx1201) found.";
                return false;
            }

            reason = null;
            return true;
        }
        catch (Exception ex)
        {
            reason = $"Failed to enumerate HIP devices: {ex.Message}";
            return false;
        }
    }

    /// <summary>
    /// Enumerates all visible devices from the HIP driver without filtering.
    /// </summary>
    public static unsafe IReadOnlyList<HipDeviceInfo> QueryPhysicalDevices()
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();
        HipApi.hipGetDeviceCount(out int count).ThrowOnError();

        var list = new List<HipDeviceInfo>(count);
        byte* nameBuffer = stackalloc byte[256];

        for (int ordinal = 0; ordinal < count; ordinal++)
        {
            HipApi.hipSetDevice(ordinal).ThrowOnError();
            HipApi.hipDeviceGetName(nameBuffer, 256, ordinal).ThrowOnError();
            int nullIdx = 0;
            while (nullIdx < 256 && nameBuffer[nullIdx] != 0) nullIdx++;
            string name = Encoding.ASCII.GetString(nameBuffer, nullIdx).Trim();

            HipApi.hipDeviceGetAttribute(out int integrated, HipApi.hipDeviceAttributeIntegrated, ordinal).ThrowOnError();
            HipApi.hipDeviceGetAttribute(out int wgp, HipApi.hipDeviceAttributeMultiprocessorCount, ordinal).ThrowOnError();
            HipApi.hipDeviceGetAttribute(out int l2, HipApi.hipDeviceAttributeL2CacheSize, ordinal).ThrowOnError();
            HipApi.hipDeviceGetAttribute(out int warpSize, HipApi.hipDeviceAttributeWarpSize, ordinal).ThrowOnError();

            HipApi.hipMemGetInfo(out _, out nuint totalMem).ThrowOnError();

            // Read gcnArchName via hipGetDevicePropertiesR0600 (W9)
            HipApi.hipGetDevicePropertiesR0600(out var prop, ordinal).ThrowOnError();
            string rawArch = prop.GcnArchName;
            string arch = rawArch.Split(':', StringSplitOptions.TrimEntries)[0];

            list.Add(new HipDeviceInfo(
                Ordinal: ordinal,
                Name: name,
                Arch: arch,
                IsIntegrated: integrated != 0,
                TotalMemoryBytes: (long)totalMem,
                Wgp: wgp,
                L2CacheBytes: l2,
                WarpSize: warpSize));
        }

        return list;
    }

    /// <summary>
    /// Returns all supported discrete AMD GPUs on the system.
    /// </summary>
    public static IReadOnlyList<HipDevice> GetSupportedDevices()
    {
        var raw = QueryPhysicalDevices();
        return FilterSupportedDevices(raw);
    }

    /// <summary>
    /// Resolves a device specifier string per decision D9.
    /// Accepted formats:
    /// <list type="bullet">
    ///   <item><c>"hip"</c> or <c>null</c>: first <c>gfx1100</c> device (primary target).</item>
    ///   <item><c>"hip:gfx1100"</c>: device with <c>Arch == "gfx1100"</c>.</item>
    ///   <item><c>"hip:gfx1201"</c>: device with <c>Arch == "gfx1201"</c>.</item>
    ///   <item><c>"hip:N"</c>: device by ordinal N (rejects iGPU or unsupported arch).</item>
    /// </list>
    /// </summary>
    public static HipDevice ResolveDevice(string? specifier = null)
    {
        var all = QueryPhysicalDevices();
        return ResolveDevice(all, specifier);
    }

    /// <summary>
    /// Filters a list of device descriptors, removing integrated and unsupported GPUs (D6).
    /// </summary>
    internal static List<HipDevice> FilterSupportedDevices(IEnumerable<HipDeviceInfo> allDevices)
    {
        var result = new List<HipDevice>();
        foreach (var d in allDevices)
        {
            if (d.IsIntegrated)
                continue;

            if (!SupportedArchitectures.Contains(d.Arch, StringComparer.OrdinalIgnoreCase))
                continue;

            result.Add(new HipDevice(d));
        }
        return result;
    }

    /// <summary>
    /// Resolves a specifier against a specific device list (enabling isolated mock testing).
    /// </summary>
    internal static HipDevice ResolveDevice(IReadOnlyList<HipDeviceInfo> allDevices, string? specifier = null)
    {
        string spec = string.IsNullOrWhiteSpace(specifier) ? "hip" : specifier.Trim();

        if (spec.Equals("hip", StringComparison.OrdinalIgnoreCase))
        {
            var primary = allDevices.FirstOrDefault(d => !d.IsIntegrated &&
                string.Equals(d.Arch, PrimaryArch, StringComparison.OrdinalIgnoreCase));
            if (primary != default)
                return new HipDevice(primary);

            var supported = allDevices.FirstOrDefault(d => !d.IsIntegrated &&
                SupportedArchitectures.Contains(d.Arch, StringComparer.OrdinalIgnoreCase));
            if (supported != default)
                return new HipDevice(supported);

            throw new InvalidOperationException("No supported AMD HIP GPU found (gfx1100 or gfx1201 required).");
        }

        if (spec.StartsWith("hip:", StringComparison.OrdinalIgnoreCase))
        {
            string sub = spec["hip:".Length..].Trim();

            if (SupportedArchitectures.Contains(sub, StringComparer.OrdinalIgnoreCase))
            {
                var match = allDevices.FirstOrDefault(d => !d.IsIntegrated &&
                    string.Equals(d.Arch, sub, StringComparison.OrdinalIgnoreCase));
                if (match == default)
                    throw new InvalidOperationException($"No supported discrete AMD HIP GPU with architecture '{sub}' found.");
                return new HipDevice(match);
            }

            if (int.TryParse(sub, out int ordinal))
            {
                var dev = allDevices.FirstOrDefault(d => d.Ordinal == ordinal);
                if (dev == default)
                    throw new ArgumentException($"HIP device ordinal {ordinal} does not exist (visible count is {allDevices.Count}).");

                if (dev.IsIntegrated)
                    throw new ArgumentException($"HIP device #{ordinal} ({dev.Name}) is an integrated GPU, which is not supported (D6/D9).");

                if (!SupportedArchitectures.Contains(dev.Arch, StringComparer.OrdinalIgnoreCase))
                    throw new ArgumentException(
                        $"HIP device #{ordinal} ({dev.Name}) has unsupported architecture '{dev.Arch}'. Supported: {string.Join(", ", SupportedArchitectures)} (D6).");

                return new HipDevice(dev);
            }

            throw new ArgumentException($"Unrecognized HIP device specifier '{spec}'. Valid forms: 'hip', 'hip:gfx1100', 'hip:gfx1201', 'hip:N'.");
        }

        throw new ArgumentException($"Device specifier '{specifier}' is not a HIP specifier. Expected 'hip' or 'hip:<arch|ordinal>'.");
    }

    /// <inheritdoc/>
    public override string ToString() =>
        $"GPU {Ordinal}: {Name} ({Arch}, {TotalMemoryFormatted}, {ComputeUnits} CU [{Wgp} WGP], L2 {L2CacheFormatted})";
}
