using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Hip.Spikes;

/// <summary>
/// Minimal, throwaway HIP bindings for the H0.2/H0.3 spikes. The product interop is H1.1
/// (<c>DotLLM.Hip/Interop</c>). Loads <c>amdhip64_7.dll</c> by full path from <c>%HIP_PATH%\bin</c> (W5).
/// Enum values verified against HIP SDK 7.2.3 headers with a compiled probe.
/// </summary>
internal static unsafe partial class Hip
{
    private const string Lib = "amdhip64_7";

    public const int AttrIntegrated = 16;
    public const int AttrL2CacheSize = 19;
    public const int AttrMultiprocessorCount = 63;
    public const uint StreamNonBlocking = 1;
    public const int CaptureModeGlobal = 0;
    public const int ErrorNoBinaryForGpu = 209;

    static Hip() => NativeLibrary.SetDllImportResolver(typeof(Hip).Assembly, Resolve);

    private static nint Resolve(string name, System.Reflection.Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Lib)
            return 0;
        string? hipPath = Environment.GetEnvironmentVariable("HIP_PATH");
        if (string.IsNullOrEmpty(hipPath))
            throw new DllNotFoundException("HIP_PATH is not set (HIP SDK 7.2.3 required).");
        string full = Path.Combine(hipPath, "bin", "amdhip64_7.dll");
        if (!File.Exists(full))
            throw new DllNotFoundException($"'{full}' not found - HIP_PATH must be the SDK root (W14).");
        return NativeLibrary.Load(full);
    }

    [LibraryImport(Lib)] public static partial int hipInit(uint flags);
    [LibraryImport(Lib)] public static partial int hipRuntimeGetVersion(out int version);
    [LibraryImport(Lib)] public static partial int hipGetDeviceCount(out int count);
    [LibraryImport(Lib)] public static partial int hipSetDevice(int device);
    [LibraryImport(Lib)] public static partial int hipDeviceGetName(byte* name, int len, int device);
    [LibraryImport(Lib)] public static partial int hipDeviceGetAttribute(out int value, int attr, int device);
    [LibraryImport(Lib)] public static partial int hipMalloc(out nint devicePtr, nuint size);
    [LibraryImport(Lib)] public static partial int hipFree(nint devicePtr);
    [LibraryImport(Lib)] public static partial int hipMemset(nint devicePtr, int value, nuint size);
    [LibraryImport(Lib)] public static partial int hipDeviceSynchronize();
    [LibraryImport(Lib)] public static partial int hipStreamCreateWithFlags(out nint stream, uint flags);
    [LibraryImport(Lib)] public static partial int hipStreamDestroy(nint stream);
    [LibraryImport(Lib)] public static partial int hipStreamSynchronize(nint stream);
    [LibraryImport(Lib)] public static partial int hipModuleLoadData(out nint module, void* image);
    [LibraryImport(Lib, StringMarshalling = StringMarshalling.Utf8)] public static partial int hipModuleGetFunction(out nint function, nint module, string name);
    [LibraryImport(Lib)] public static partial int hipModuleUnload(nint module);
    [LibraryImport(Lib)]
    public static partial int hipModuleLaunchKernel(nint function, uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ, uint sharedMemBytes, nint stream, void** kernelParams, void** extra);
    [LibraryImport(Lib)] public static partial int hipEventCreate(out nint evt);
    [LibraryImport(Lib)] public static partial int hipEventRecord(nint evt, nint stream);
    [LibraryImport(Lib)] public static partial int hipEventSynchronize(nint evt);
    [LibraryImport(Lib)] public static partial int hipEventElapsedTime(out float ms, nint start, nint stop);
    [LibraryImport(Lib)] public static partial int hipEventDestroy(nint evt);
    [LibraryImport(Lib)] public static partial int hipStreamBeginCapture(nint stream, int mode);
    [LibraryImport(Lib)] public static partial int hipStreamEndCapture(nint stream, out nint graph);
    [LibraryImport(Lib)] public static partial int hipGraphInstantiate(out nint graphExec, nint graph, nint errorNode, byte* logBuffer, nuint bufferSize);
    [LibraryImport(Lib)] public static partial int hipGraphLaunch(nint graphExec, nint stream);
    [LibraryImport(Lib)] public static partial int hipGraphExecDestroy(nint graphExec);
    [LibraryImport(Lib)] public static partial int hipGraphDestroy(nint graph);
    [LibraryImport(Lib)] public static partial byte* hipGetErrorName(int error);

    /// <summary>Throws if <paramref name="error"/> is not hipSuccess.</summary>
    public static void Check(int error, [CallerArgumentExpression(nameof(error))] string? call = null)
    {
        if (error != 0)
            throw new InvalidOperationException($"{call} failed: {ErrorName(error)} ({error})");
    }

    /// <summary>Returns the symbolic HIP error name.</summary>
    public static string ErrorName(int error) => Marshal.PtrToStringAnsi((nint)hipGetErrorName(error)) ?? error.ToString();
}
