using DotLLM.Core.Tensors;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Tests HIP tensor memory allocations, pinned host buffers, and backend transfers (H1.3).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipTensorTests
{
    [SkippableFact]
    public void AllocateTensor_AssignsPropertiesAndAllocatesMemory()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var device = HipDevice.ResolveDevice("hip");
        device.MakeCurrent();

        var shape = new TensorShape(16, 32);
        using var tensor = HipTensor.Allocate(shape, DType.Float16, device.Ordinal);

        Assert.Equal(16 * 32, tensor.ElementCount);
        Assert.Equal(16 * 32 * 2, tensor.ByteCount); // FP16 = 2 bytes
        Assert.Equal(device.Ordinal, tensor.DeviceId);
        Assert.NotEqual(0, tensor.DataPointer);
    }

    [SkippableFact]
    public void AllocateBytes_AllocatesExplicitByteSize()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var device = HipDevice.ResolveDevice("hip");
        device.MakeCurrent();

        var shape = new TensorShape(64);
        long explicitBytes = 256;
        using var tensor = HipTensor.AllocateBytes(shape, DType.Q4_0, device.Ordinal, explicitBytes);

        Assert.Equal(explicitBytes, tensor.ByteCount);
        Assert.NotEqual(0, tensor.DataPointer);
    }

    [SkippableFact]
    public void PinnedBuffer_AllocatesAndAccessesHostMemory()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        int count = 1024;
        long bytes = count * sizeof(float);
        using var pinned = new HipPinnedBuffer(bytes);

        Assert.NotEqual(0, pinned.Pointer);
        Assert.Equal(bytes, pinned.ByteCount);

        var span = pinned.AsSpan<float>();
        for (int i = 0; i < count; i++)
            span[i] = i * 1.5f;

        var readOnly = pinned.AsReadOnlySpan<float>();
        for (int i = 0; i < count; i++)
            Assert.Equal(i * 1.5f, readOnly[i]);
    }

    [SkippableFact]
    public unsafe void Backend_CopiesHostToDeviceAndDeviceToHostAccurately()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var device = HipDevice.ResolveDevice("hip");
        device.MakeCurrent();

        using var backend = new HipBackend();

        int n = 256;
        var shape = new TensorShape(n);
        using var hostSrc = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        using var deviceTensor = HipTensor.Allocate(shape, DType.Float32, device.Ordinal);
        using var hostDst = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);

        // Fill hostSrc with test data
        float* pSrc = (float*)hostSrc.DataPointer;
        for (int i = 0; i < n; i++)
            pSrc[i] = i * 0.25f - 10.0f;

        // Host -> Device
        backend.CopyBetweenDevices(hostSrc, deviceTensor);

        // Device -> Host
        backend.CopyBetweenDevices(deviceTensor, hostDst);

        float* pDst = (float*)hostDst.DataPointer;
        for (int i = 0; i < n; i++)
            Assert.Equal(pSrc[i], pDst[i]);
    }

    [SkippableFact]
    public unsafe void Backend_CopiesDeviceToDeviceAccurately()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        var device = HipDevice.ResolveDevice("hip");
        device.MakeCurrent();

        using var backend = new HipBackend();

        int n = 128;
        var shape = new TensorShape(n);
        using var hostSrc = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        using var devA = HipTensor.Allocate(shape, DType.Float32, device.Ordinal);
        using var devB = HipTensor.Allocate(shape, DType.Float32, device.Ordinal);
        using var hostDst = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);

        float* pSrc = (float*)hostSrc.DataPointer;
        for (int i = 0; i < n; i++)
            pSrc[i] = 42.0f + i;

        backend.CopyBetweenDevices(hostSrc, devA);
        backend.CopyBetweenDevices(devA, devB);
        backend.CopyBetweenDevices(devB, hostDst);

        float* pDst = (float*)hostDst.DataPointer;
        for (int i = 0; i < n; i++)
            Assert.Equal(pSrc[i], pDst[i]);
    }
}
