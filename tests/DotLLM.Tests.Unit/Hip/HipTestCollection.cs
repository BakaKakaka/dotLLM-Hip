using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Disables parallel test execution across HIP test classes to prevent VRAM
/// contention and device resource conflicts on the physical GPUs (W13).
/// </summary>
[CollectionDefinition("HipGpu", DisableParallelization = true)]
public class HipTestCollection
{
}
