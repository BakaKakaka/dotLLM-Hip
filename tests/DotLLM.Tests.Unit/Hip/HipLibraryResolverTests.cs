using System.Diagnostics;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Tests HIP runtime and rocBLAS library resolution and version verification (H1.1, W5, W6, W14).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipLibraryResolverTests
{
    [SkippableFact]
    public void HipPath_IsConfiguredAndPointsToSdkRoot()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        bool ok = HipLibraryResolver.TryGetHipPath(out string hipPath, out string? error);
        Assert.True(ok, error);
        Assert.False(string.IsNullOrEmpty(hipPath));
        Assert.False(hipPath.EndsWith("\\bin", StringComparison.OrdinalIgnoreCase));
        Assert.True(File.Exists(Path.Combine(hipPath, "bin", "amdhip64_7.dll")));
        Assert.True(File.Exists(Path.Combine(hipPath, "include", "hip", "hip_runtime.h")));
    }

    [SkippableFact]
    public void HipRuntime_LoadsSuccessfully_AndReportsMajorVersion7()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        nint handle = HipLibraryResolver.EnsureHipLoaded();
        Assert.NotEqual(0, handle);

        int version = HipLibraryResolver.GetRuntimeVersion();
        int major = version / 10_000_000;
        Assert.Equal(7, major);
    }

    [SkippableFact]
    public void Rocblas_LoadsSuccessfully_AfterHipRuntime()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        nint handle = HipLibraryResolver.EnsureRocblasLoaded();
        Assert.NotEqual(0, handle);
    }

    [SkippableFact]
    public void SingleHipRuntimeModule_EnforcedInProcess()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        HipLibraryResolver.EnsureHipLoaded();
        HipLibraryResolver.EnsureRocblasLoaded();

        string loadedPath = HipLibraryResolver.VerifySingleHipRuntimeModule();
        Assert.False(string.IsNullOrEmpty(loadedPath));
        Assert.EndsWith("amdhip64_7.dll", loadedPath, StringComparison.OrdinalIgnoreCase);

        // Verify with direct process module inspection
        var hipModules = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Where(m => m.ModuleName.StartsWith("amdhip64_", StringComparison.OrdinalIgnoreCase))
            .ToList();

        Assert.Single(hipModules);
        Assert.Equal("amdhip64_7.dll", hipModules[0].ModuleName, ignoreCase: true);
    }
}
