using System.Runtime.InteropServices;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Tests HIP device enumeration, property queries, and architecture resolution (H1.2, D6, D9, W4, W11).
/// </summary>
[Collection("HipGpu")]
public class HipDeviceTests
{
    [Fact]
    public unsafe void StructLayout_HipDevicePropR0600_MatchesExpectedSizeAndOffset()
    {
        // W9 / H1.1 requirement: sizeof-test for hipDeviceProp_t (R0600)
        Assert.Equal(1472, sizeof(HipDevicePropR0600));
        Assert.Equal(1472, Marshal.SizeOf<HipDevicePropR0600>());

        int offset = (int)Marshal.OffsetOf<HipDevicePropR0600>(nameof(HipDevicePropR0600.GcnArchNameBytes));
        Assert.Equal(1160, offset);
    }

    [Fact]
    public void MockEnumeration_FiltersOutIntegratedAndUnsupportedGpus()
    {
        var mockDevices = new List<HipDeviceInfo>
        {
            new(Ordinal: 0, Name: "AMD Radeon(TM) Graphics", Arch: "gfx1036", IsIntegrated: true,
                TotalMemoryBytes: 512 * 1024 * 1024, Wgp: 1, L2CacheBytes: 1024 * 1024, WarpSize: 32),
            new(Ordinal: 1, Name: "AMD Radeon RX 9070 XT", Arch: "gfx1201", IsIntegrated: false,
                TotalMemoryBytes: 16L * 1024 * 1024 * 1024, Wgp: 32, L2CacheBytes: 8 * 1024 * 1024, WarpSize: 32),
            new(Ordinal: 2, Name: "AMD Radeon RX 7900 XTX", Arch: "gfx1100", IsIntegrated: false,
                TotalMemoryBytes: 24L * 1024 * 1024 * 1024, Wgp: 48, L2CacheBytes: 6 * 1024 * 1024, WarpSize: 32),
            new(Ordinal: 3, Name: "AMD Radeon VII", Arch: "gfx906", IsIntegrated: false,
                TotalMemoryBytes: 16L * 1024 * 1024 * 1024, Wgp: 30, L2CacheBytes: 4 * 1024 * 1024, WarpSize: 64)
        };

        var filtered = HipDevice.FilterSupportedDevices(mockDevices);

        Assert.Equal(2, filtered.Count);
        Assert.Contains(filtered, d => d.Arch == "gfx1201" && d.Ordinal == 1);
        Assert.Contains(filtered, d => d.Arch == "gfx1100" && d.Ordinal == 2);
        Assert.DoesNotContain(filtered, d => d.Arch == "gfx1036");
        Assert.DoesNotContain(filtered, d => d.Arch == "gfx906");
    }

    [Fact]
    public void MockEnumeration_ResolvesDeviceSpecifiersCorrectly()
    {
        // Ordinal order has 9070 XT first, 7900 XTX second, with an iGPU at ordinal 0.
        var mockDevices = new List<HipDeviceInfo>
        {
            new(Ordinal: 0, Name: "AMD Ryzen iGPU", Arch: "gfx1036", IsIntegrated: true,
                TotalMemoryBytes: 512 * 1024 * 1024, Wgp: 1, L2CacheBytes: 1024 * 1024, WarpSize: 32),
            new(Ordinal: 1, Name: "AMD Radeon RX 9070 XT", Arch: "gfx1201", IsIntegrated: false,
                TotalMemoryBytes: 16L * 1024 * 1024 * 1024, Wgp: 32, L2CacheBytes: 8 * 1024 * 1024, WarpSize: 32),
            new(Ordinal: 2, Name: "AMD Radeon RX 7900 XTX", Arch: "gfx1100", IsIntegrated: false,
                TotalMemoryBytes: 24L * 1024 * 1024 * 1024, Wgp: 48, L2CacheBytes: 6 * 1024 * 1024, WarpSize: 32)
        };

        // "hip" default chooses primary architecture gfx1100, NOT the lower ordinal 1
        var defaultDev = HipDevice.ResolveDevice(mockDevices, "hip");
        Assert.Equal("gfx1100", defaultDev.Arch);
        Assert.Equal(2, defaultDev.Ordinal);
        Assert.Equal(96, defaultDev.ComputeUnits); // 2 * 48 WGP (W11)

        // "hip:gfx1100" chooses gfx1100
        var dev1100 = HipDevice.ResolveDevice(mockDevices, "hip:gfx1100");
        Assert.Equal("gfx1100", dev1100.Arch);
        Assert.Equal(2, dev1100.Ordinal);

        // "hip:gfx1201" chooses gfx1201
        var dev1201 = HipDevice.ResolveDevice(mockDevices, "hip:gfx1201");
        Assert.Equal("gfx1201", dev1201.Arch);
        Assert.Equal(1, dev1201.Ordinal);
        Assert.Equal(64, dev1201.ComputeUnits); // 2 * 32 WGP

        // "hip:1" and "hip:2" choose respective ordinals
        Assert.Equal(1, HipDevice.ResolveDevice(mockDevices, "hip:1").Ordinal);
        Assert.Equal(2, HipDevice.ResolveDevice(mockDevices, "hip:2").Ordinal);

        // "hip:0" (iGPU) is explicitly rejected
        var exIgpu = Assert.Throws<ArgumentException>(() => HipDevice.ResolveDevice(mockDevices, "hip:0"));
        Assert.Contains("integrated GPU", exIgpu.Message, StringComparison.OrdinalIgnoreCase);

        // Unrecognized format
        Assert.Throws<ArgumentException>(() => HipDevice.ResolveDevice(mockDevices, "hip:gfx9999"));
        Assert.Throws<ArgumentException>(() => HipDevice.ResolveDevice(mockDevices, "cuda"));
    }

    [SkippableFact]
    [Trait("Category", "GPU-AMD")]
    public void PhysicalDevices_EnumerateSupportedGpusWithAccurateProperties()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var supported = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(supported);

        // Target machine has both RX 7900 XTX and RX 9070 XT
        var xtx = supported.FirstOrDefault(d => d.Arch == "gfx1100");
        Assert.NotNull(xtx);
        Assert.True(xtx.ComputeUnits >= 96); // 48 WGP = 96 CU
        Assert.True(xtx.TotalMemoryBytes > 20L * 1024 * 1024 * 1024);
        Assert.Equal(32, xtx.WarpSize);

        var r9070 = supported.FirstOrDefault(d => d.Arch == "gfx1201");
        if (r9070 != null)
        {
            Assert.True(r9070.ComputeUnits >= 64); // 32 WGP = 64 CU
            Assert.True(r9070.TotalMemoryBytes > 14L * 1024 * 1024 * 1024);
            Assert.Equal(32, r9070.WarpSize);
        }
    }

    [SkippableFact]
    [Trait("Category", "GPU-AMD")]
    public void PhysicalDevices_ResolveSpecifiersMatchTargets()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var defaultHip = HipDevice.ResolveDevice("hip");
        Assert.Equal("gfx1100", defaultHip.Arch);

        var dev1100 = HipDevice.ResolveDevice("hip:gfx1100");
        Assert.Equal("gfx1100", dev1100.Arch);

        var supported = HipDevice.GetSupportedDevices();
        if (supported.Any(d => d.Arch == "gfx1201"))
        {
            var dev1201 = HipDevice.ResolveDevice("hip:gfx1201");
            Assert.Equal("gfx1201", dev1201.Arch);
        }
    }
}
