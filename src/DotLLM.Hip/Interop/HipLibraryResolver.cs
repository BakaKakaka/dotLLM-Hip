using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Resolves HIP and rocBLAS native libraries from <c>%HIP_PATH%\bin</c> on Windows (W5, W6, W14).
/// Enforces:
/// 1. Loading <c>amdhip64_7.dll</c> by full path from <c>%HIP_PATH%\bin</c>.
/// 2. Validating HIP runtime major version 7.2.x (D4).
/// 3. Loading <c>rocblas.dll</c> strictly after <c>amdhip64_7.dll</c> (W6).
/// 4. Verifying that exactly one <c>amdhip64_*</c> module is loaded in the process and is v7 (W5, W6).
/// </summary>
public static class HipLibraryResolver
{
    private const string HipLibName = "amdhip64_7";
    private const string RocblasLibName = "rocblas";
    private const int RequiredMajorVersion = 7;

    private static readonly object s_lock = new();
    private static int s_registered;
    private static nint s_hipHandle;
    private static nint s_rocblasHandle;
    private static int s_runtimeVersion;

    /// <summary>
    /// Registers the DllImport resolver for <see cref="DotLLM.Hip"/>. Idempotent.
    /// </summary>
    public static void Register()
    {
        if (Interlocked.Exchange(ref s_registered, 1) != 0)
            return;

        NativeLibrary.SetDllImportResolver(typeof(HipLibraryResolver).Assembly, Resolve);
    }

    /// <summary>
    /// Ensures that the HIP runtime (<c>amdhip64_7.dll</c>) is loaded and verified.
    /// </summary>
    /// <returns>Native handle to <c>amdhip64_7.dll</c>.</returns>
    public static nint EnsureHipLoaded()
    {
        if (s_hipHandle != 0)
            return s_hipHandle;

        lock (s_lock)
        {
            if (s_hipHandle != 0)
                return s_hipHandle;

            string hipPath = GetValidatedHipPath();
            string hipDllPath = Path.Combine(hipPath, "bin", "amdhip64_7.dll");
            if (!File.Exists(hipDllPath))
            {
                throw new DllNotFoundException(
                    $"'{hipDllPath}' not found. HIP_PATH must be the SDK root (W14).");
            }

            nint handle = NativeLibrary.Load(hipDllPath);

            // Step 2: Validate runtime major version via export before returning.
            unsafe
            {
                nint export = NativeLibrary.GetExport(handle, "hipRuntimeGetVersion");
                delegate* unmanaged[Cdecl]<out int, int> pGetVer =
                    (delegate* unmanaged[Cdecl]<out int, int>)export;

                int res = pGetVer(out int version);
                if (res != 0)
                    throw new HipException(res, $"hipRuntimeGetVersion failed during resolver validation (error {res})");

                int major = version / 10_000_000;
                if (major != RequiredMajorVersion)
                {
                    NativeLibrary.Free(handle);
                    throw new HipException(
                        $"HIP runtime reports version {version} (major {major}); major {RequiredMajorVersion} is required (D4).");
                }

                s_runtimeVersion = version;
            }

            // Step 4: Verify single HIP runtime module in process (W5, W6).
            VerifySingleHipRuntimeModule(hipDllPath);

            s_hipHandle = handle;
            return s_hipHandle;
        }
    }

    /// <summary>
    /// Ensures that <c>rocblas.dll</c> is loaded after <c>amdhip64_7.dll</c> (W6).
    /// </summary>
    /// <returns>Native handle to <c>rocblas.dll</c>.</returns>
    public static nint EnsureRocblasLoaded()
    {
        if (s_rocblasHandle != 0)
            return s_rocblasHandle;

        lock (s_lock)
        {
            if (s_rocblasHandle != 0)
                return s_rocblasHandle;

            // Step 1: HIP runtime must be loaded first (W6).
            EnsureHipLoaded();

            string hipPath = GetValidatedHipPath();
            string rocblasDllPath = Path.Combine(hipPath, "bin", "rocblas.dll");
            if (!File.Exists(rocblasDllPath))
                throw new DllNotFoundException($"'{rocblasDllPath}' not found (HIP SDK 7.2 required).");

            string rocblasLibDir = Path.Combine(hipPath, "bin", "rocblas", "library");
            if (!Directory.Exists(rocblasLibDir))
            {
                throw new DirectoryNotFoundException(
                    $"rocBLAS library directory '{rocblasLibDir}' not found. rocBLAS cannot load Tensile kernels (W7).");
            }

            nint handle = NativeLibrary.Load(rocblasDllPath);

            // Verify again that loading rocBLAS did not load a rogue amdhip64_6.dll
            string hipDllPath = Path.Combine(hipPath, "bin", "amdhip64_7.dll");
            VerifySingleHipRuntimeModule(hipDllPath);

            s_rocblasHandle = handle;
            return s_rocblasHandle;
        }
    }

    /// <summary>
    /// Returns the verified HIP runtime version (e.g. 70260201).
    /// </summary>
    public static int GetRuntimeVersion()
    {
        EnsureHipLoaded();
        return s_runtimeVersion;
    }

    /// <summary>
    /// Verifies that exactly one <c>amdhip64_*</c> module is loaded in the current process,
    /// matching the expected full path.
    /// </summary>
    public static string VerifySingleHipRuntimeModule(string? expectedFullPath = null)
    {
        var modules = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Where(m => m.ModuleName.StartsWith("amdhip64_", StringComparison.OrdinalIgnoreCase))
            .ToList();

        if (modules.Count == 0)
            throw new HipException("No amdhip64_* module found in the current process.");

        if (modules.Count > 1)
        {
            string loaded = string.Join(", ", modules.Select(m => m.FileName));
            throw new HipException(
                $"Expected exactly one amdhip64_* module loaded in the process, found {modules.Count}: [{loaded}] (W5, W6).");
        }

        string loadedPath = modules[0].FileName;
        if (!string.IsNullOrEmpty(expectedFullPath) &&
            !string.Equals(loadedPath, expectedFullPath, StringComparison.OrdinalIgnoreCase))
        {
            throw new HipException(
                $"amdhip64 loaded from '{loadedPath}', expected '{expectedFullPath}' (W5).");
        }

        return loadedPath;
    }

    /// <summary>
    /// Checks if <c>HIP_PATH</c> is configured properly, returning false with a reason if not.
    /// </summary>
    public static bool TryGetHipPath(out string hipPath, out string? error)
    {
        hipPath = string.Empty;
        error = null;

        string? env = Environment.GetEnvironmentVariable("HIP_PATH");
        if (string.IsNullOrWhiteSpace(env))
        {
            error = "HIP_PATH environment variable is not set (HIP SDK 7.2.3 required).";
            return false;
        }

        string root = env.TrimEnd('\\', '/');
        if (Path.GetFileName(root).Equals("bin", StringComparison.OrdinalIgnoreCase))
        {
            error = $"HIP_PATH points at the bin directory ('{env}'). It must be the SDK root (W14).";
            return false;
        }

        string header = Path.Combine(root, "include", "hip", "hip_runtime.h");
        string hipDll = Path.Combine(root, "bin", "amdhip64_7.dll");
        if (!File.Exists(header) || !File.Exists(hipDll))
        {
            error = $"HIP_PATH ('{env}') is missing include\\hip\\hip_runtime.h or bin\\amdhip64_7.dll (W14).";
            return false;
        }

        hipPath = root;
        return true;
    }

    private static string GetValidatedHipPath()
    {
        if (!TryGetHipPath(out string hipPath, out string? error))
            throw new DllNotFoundException(error);
        return hipPath;
    }

    private static nint Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == HipLibName)
            return EnsureHipLoaded();

        if (libraryName == RocblasLibName)
            return EnsureRocblasLoaded();

        return 0;
    }
}
