using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Entry for a kernel source or header file in the embedded kernel manifest.
/// </summary>
public sealed record HipKernelManifestFile(string Name, string Sha256);

/// <summary>
/// Deserialized representation of embedded <c>hsaco/kernels.manifest.json</c> (D7).
/// </summary>
public sealed class HipKernelManifest
{
    /// <summary>Manifest schema version.</summary>
    public int Schema { get; init; }

    /// <summary>HIP compiler version used at build time.</summary>
    public string HipccVersion { get; init; } = string.Empty;

    /// <summary>Compiler command-line flags used to compile code objects.</summary>
    public string Flags { get; init; } = string.Empty;

    /// <summary>Target GPU architectures included in this build.</summary>
    public string[] Archs { get; init; } = [];

    /// <summary>List of compiled source and header files with their SHA-256 digests.</summary>
    public HipKernelManifestFile[] Files { get; init; } = [];
}

/// <summary>
/// Loads AOT-compiled HIP code objects (<c>.hsaco</c>) embedded in <c>DotLLM.Hip.dll</c> (D7).
/// Kernel binaries are embedded at build time under <c>hsaco/{arch}/{name}.hsaco</c>.
/// </summary>
public static class HipKernelLoader
{
    private const string ManifestResource = "hsaco/kernels.manifest.json";
    private static readonly Assembly s_assembly = typeof(HipKernelLoader).Assembly;

    /// <summary>
    /// Checks whether embedded HIP kernels are present in the assembly.
    /// </summary>
    public static bool AreKernelsAvailable(out string reason)
    {
        if (s_assembly.GetManifestResourceInfo(ManifestResource) == null)
        {
            reason = "HIP kernels not built: embedded manifest 'hsaco/kernels.manifest.json' missing.";
            return false;
        }

        // Verify at least one hsaco resource exists
        bool hasHsaco = s_assembly.GetManifestResourceNames()
            .Any(n => n.StartsWith("hsaco/", StringComparison.OrdinalIgnoreCase) && n.EndsWith(".hsaco", StringComparison.OrdinalIgnoreCase));

        if (!hasHsaco)
        {
            reason = "HIP kernels not built: no embedded .hsaco binaries found.";
            return false;
        }

        reason = string.Empty;
        return true;
    }

    /// <summary>
    /// Checks whether a specific kernel module is embedded for the given architecture.
    /// </summary>
    public static bool HasKernel(string arch, string kernelName)
    {
        string resName = $"hsaco/{arch}/{kernelName}.hsaco";
        return s_assembly.GetManifestResourceInfo(resName) != null;
    }

    /// <summary>
    /// Loads an embedded kernel module into HIP device memory via <c>hipModuleLoadData</c>.
    /// </summary>
    /// <param name="arch">Target GPU architecture (e.g. <c>"gfx1100"</c> or <c>"gfx1201"</c>).</param>
    /// <param name="kernelName">Base name of the kernel (e.g. <c>"toolchain_smoke"</c>).</param>
    /// <returns>A loaded <see cref="HipModule"/> instance.</returns>
    public static unsafe HipModule LoadEmbeddedModule(string arch, string kernelName)
    {
        string resName = $"hsaco/{arch}/{kernelName}.hsaco";
        using Stream? stream = s_assembly.GetManifestResourceStream(resName);
        if (stream == null)
        {
            throw new InvalidOperationException(
                $"Embedded HIP kernel '{resName}' not found in {s_assembly.GetName().Name}. " +
                $"Ensure that '{kernelName}.hip' was compiled for {arch}.");
        }

        int length = checked((int)stream.Length);
        void* image = NativeMemory.Alloc((nuint)length);
        try
        {
            stream.ReadExactly(new Span<byte>(image, length));
            HipApi.hipModuleLoadData(out nint module, image).ThrowOnError();
            return new HipModule(module);
        }
        finally
        {
            NativeMemory.Free(image);
        }
    }

    /// <summary>
    /// Reads and parses the embedded <c>hsaco/kernels.manifest.json</c>.
    /// </summary>
    public static HipKernelManifest GetEmbeddedManifest()
    {
        using Stream? stream = s_assembly.GetManifestResourceStream(ManifestResource);
        if (stream == null)
            throw new InvalidOperationException($"Embedded kernel manifest '{ManifestResource}' not found.");

        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        int schema = root.GetProperty("schema").GetInt32();
        string hipcc = root.GetProperty("hipccVersion").GetString() ?? string.Empty;
        string flags = root.GetProperty("flags").GetString() ?? string.Empty;
        string[] archs = root.GetProperty("archs").EnumerateArray()
            .Select(e => e.GetString() ?? string.Empty)
            .ToArray();

        var files = new List<HipKernelManifestFile>();
        foreach (var item in root.GetProperty("files").EnumerateArray())
        {
            string name = item.GetProperty("name").GetString() ?? string.Empty;
            string sha = item.GetProperty("sha256").GetString() ?? string.Empty;
            files.Add(new HipKernelManifestFile(name, sha));
        }

        return new HipKernelManifest
        {
            Schema = schema,
            HipccVersion = hipcc,
            Flags = flags,
            Archs = archs,
            Files = files.ToArray()
        };
    }
}
