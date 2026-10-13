using System.Runtime.InteropServices;
using System.Text;

namespace DotLLM.Hip.Interop;

/// <summary>
/// Layout of <c>hipDeviceProp_t</c> (R0600) for reading <c>gcnArchName</c> (HIP 7.2, W9).
/// Total struct size is 1472 bytes; <c>gcnArchName</c> is at offset 1160.
/// </summary>
[StructLayout(LayoutKind.Explicit, Size = ExpectedSize)]
public unsafe struct HipDevicePropR0600
{
    /// <summary>Expected sizeof(hipDeviceProp_t) in HIP SDK 7.2.3.</summary>
    public const int ExpectedSize = 1472;

    /// <summary>Offset of <c>char gcnArchName[256]</c> in <c>hipDeviceProp_t</c>.</summary>
    public const int GcnArchNameOffset = 1160;

    /// <summary>Length of <c>gcnArchName</c> character array.</summary>
    public const int GcnArchNameLength = 256;

    /// <summary>Raw fixed byte buffer containing the null-terminated GCN architecture name.</summary>
    [FieldOffset(GcnArchNameOffset)]
    public fixed byte GcnArchNameBytes[GcnArchNameLength];

    /// <summary>Extracts the null-terminated GCN architecture string (e.g. "gfx1100:xnack-").</summary>
    public string GcnArchName
    {
        get
        {
            fixed (byte* p = GcnArchNameBytes)
            {
                int len = 0;
                while (len < GcnArchNameLength && p[len] != 0) len++;
                return Encoding.ASCII.GetString(p, len);
            }
        }
    }
}

/// <summary>
/// Direct P/Invoke declarations against AMD HIP runtime (<c>amdhip64_7.dll</c>).
/// Uses <see cref="HipLibraryResolver"/> to load exclusively from <c>%HIP_PATH%\bin</c> (W5).
/// </summary>
internal static unsafe partial class HipApi
{
    private const string LibName = "amdhip64_7";

    public const int hipSuccess = 0;

    // Stream flags
    public const uint hipStreamDefault = 0;
    public const uint hipStreamNonBlocking = 1;

    // Memcpy kinds
    public const int hipMemcpyHostToHost = 0;
    public const int hipMemcpyHostToDevice = 1;
    public const int hipMemcpyDeviceToHost = 2;
    public const int hipMemcpyDeviceToDevice = 3;
    public const int hipMemcpyDefault = 4;

    // HostMalloc flags
    public const uint hipHostMallocDefault = 0;
    public const uint hipHostMallocNonCoherent = 0x40000000;

    // Event flags
    public const uint hipEventDefault = 0;
    public const uint hipEventBlockingSync = 1;
    public const uint hipEventDisableTiming = 2;

    // Device attributes (verified against HIP 7.2.3 headers)
    public const int hipDeviceAttributeIntegrated = 16;
    public const int hipDeviceAttributeL2CacheSize = 19;
    public const int hipDeviceAttributeMaxThreadsPerBlock = 56;
    public const int hipDeviceAttributeMultiprocessorCount = 63;
    public const int hipDeviceAttributeWarpSize = 87;

    static HipApi() => HipLibraryResolver.Register();

    // ── Runtime & Device ─────────────────────────────────────────────

    [LibraryImport(LibName)]
    public static partial int hipInit(uint flags);

    [LibraryImport(LibName)]
    public static partial int hipRuntimeGetVersion(out int version);

    [LibraryImport(LibName)]
    public static partial int hipDriverGetVersion(out int version);

    [LibraryImport(LibName)]
    public static partial int hipGetDeviceCount(out int count);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipSetDevice(int deviceId);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipGetDevice(out int deviceId);

    [LibraryImport(LibName)]
    public static partial int hipDeviceSynchronize();

    [LibraryImport(LibName)]
    public static partial int hipDeviceGetAttribute(out int value, int attribute, int deviceId);

    [LibraryImport(LibName)]
    public static partial int hipDeviceGetName(byte* name, int len, int deviceId);

    [LibraryImport(LibName)]
    public static partial int hipGetDevicePropertiesR0600(out HipDevicePropR0600 prop, int deviceId);

    [LibraryImport(LibName)]
    public static partial int hipMemGetInfo(out nuint free, out nuint total);

    // ── Memory Management ────────────────────────────────────────────

    [LibraryImport(LibName)]
    public static partial int hipMalloc(out nint devicePtr, nuint sizeBytes);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipFree(nint devicePtr);

    [LibraryImport(LibName)]
    public static partial int hipHostMalloc(out nint hostPtr, nuint sizeBytes, uint flags);

    [LibraryImport(LibName)]
    public static partial int hipHostFree(nint hostPtr);

    [LibraryImport(LibName)]
    public static partial int hipMemcpy(nint dst, nint src, nuint sizeBytes, int kind);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipMemcpyAsync(nint dst, nint src, nuint sizeBytes, int kind, nint stream);

    [LibraryImport(LibName)]
    public static partial int hipMemset(nint dst, int value, nuint sizeBytes);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipMemsetAsync(nint dst, int value, nuint sizeBytes, nint stream);

    // ── Stream Operations ────────────────────────────────────────────

    [LibraryImport(LibName)]
    public static partial int hipStreamCreate(out nint stream);

    [LibraryImport(LibName)]
    public static partial int hipStreamCreateWithFlags(out nint stream, uint flags);

    [LibraryImport(LibName)]
    public static partial int hipStreamDestroy(nint stream);

    [LibraryImport(LibName)]
    public static partial int hipStreamSynchronize(nint stream);

    // ── Module & Kernel Execution ────────────────────────────────────

    [LibraryImport(LibName)]
    public static partial int hipModuleLoadData(out nint module, void* image);

    [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int hipModuleGetFunction(out nint function, nint module, string name);

    [LibraryImport(LibName)]
    public static partial int hipModuleUnload(nint module);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipModuleLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes,
        nint stream,
        void** kernelParams,
        void** extra);

    // ── Event Operations ─────────────────────────────────────────────

    [LibraryImport(LibName)]
    public static partial int hipEventCreate(out nint evt);

    [LibraryImport(LibName)]
    public static partial int hipEventCreateWithFlags(out nint evt, uint flags);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial int hipEventRecord(nint evt, nint stream);

    [LibraryImport(LibName)]
    public static partial int hipEventSynchronize(nint evt);

    [LibraryImport(LibName)]
    public static partial int hipEventElapsedTime(out float ms, nint start, nint stop);

    [LibraryImport(LibName)]
    public static partial int hipEventDestroy(nint evt);

    // ── Error Information ────────────────────────────────────────────

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial byte* hipGetErrorName(int error);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    public static partial byte* hipGetErrorString(int error);

    /// <summary>Gets the symbolic error name (e.g., "hipErrorInvalidValue").</summary>
    public static string GetErrorNameString(int error)
    {
        try
        {
            byte* p = hipGetErrorName(error);
            return p != null ? Marshal.PtrToStringAnsi((nint)p) ?? error.ToString() : error.ToString();
        }
        catch
        {
            return error.ToString();
        }
    }

    /// <summary>Gets the human-readable error description.</summary>
    public static string GetErrorStringText(int error)
    {
        try
        {
            byte* p = hipGetErrorString(error);
            return p != null ? Marshal.PtrToStringAnsi((nint)p) ?? "Unknown HIP error" : "Unknown HIP error";
        }
        catch
        {
            return "Unknown HIP error";
        }
    }
}
