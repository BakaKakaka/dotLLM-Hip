using DotLLM.Core.Backends;
using DotLLM.Core.Tensors;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// GPU backend using AMD HIP via runtime P/Invoke.
/// Supports single-device tensor allocation and transfers between host and device.
/// </summary>
public sealed class HipBackend : IBackend
{
    /// <inheritdoc/>
    public int DeviceCount { get; }

    /// <summary>
    /// Initializes the HIP backend, validating the runtime and querying visible devices.
    /// </summary>
    public HipBackend()
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();
        HipApi.hipGetDeviceCount(out int count).ThrowOnError();
        DeviceCount = count;
    }

    /// <inheritdoc/>
    public ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype)
    {
        if (deviceId < 0)
            throw new ArgumentException($"HipBackend requires deviceId >= 0, got {deviceId}.", nameof(deviceId));
        if (deviceId >= DeviceCount)
            throw new ArgumentException($"Device {deviceId} not available (have {DeviceCount} device(s)).", nameof(deviceId));

        return HipTensor.Allocate(shape, dtype, deviceId);
    }

    /// <inheritdoc/>
    public void CopyBetweenDevices(ITensor source, ITensor destination)
    {
        if (source.ByteCount != destination.ByteCount)
        {
            throw new ArgumentException(
                $"Source ({source.ByteCount} bytes) and destination ({destination.ByteCount} bytes) sizes differ.");
        }

        nuint bytes = (nuint)source.ByteCount;

        if (source.DeviceId == -1 && destination.DeviceId >= 0)
        {
            // Host -> Device
            HipApi.hipSetDevice(destination.DeviceId).ThrowOnError();
            HipApi.hipMemcpy(destination.DataPointer, source.DataPointer, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
        }
        else if (source.DeviceId >= 0 && destination.DeviceId == -1)
        {
            // Device -> Host
            HipApi.hipSetDevice(source.DeviceId).ThrowOnError();
            HipApi.hipMemcpy(destination.DataPointer, source.DataPointer, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
        }
        else if (source.DeviceId >= 0 && destination.DeviceId >= 0)
        {
            // Device -> Device
            HipApi.hipSetDevice(destination.DeviceId).ThrowOnError();
            HipApi.hipMemcpy(destination.DataPointer, source.DataPointer, bytes, HipApi.hipMemcpyDeviceToDevice).ThrowOnError();
        }
        else
        {
            throw new ArgumentException("CPU-to-CPU copy is not a HipBackend operation.");
        }
    }

    /// <inheritdoc/>
    public void AllReduce(ReadOnlySpan<ITensor> tensors) =>
        throw new NotSupportedException("HipBackend does not support AllReduce.");

    /// <inheritdoc/>
    public void Send(ITensor tensor, int targetDevice) =>
        throw new NotSupportedException("HipBackend does not support Send.");

    /// <inheritdoc/>
    public ITensor Receive(int sourceDevice, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("HipBackend does not support Receive.");

    /// <inheritdoc/>
    public void Dispose()
    {
    }
}
