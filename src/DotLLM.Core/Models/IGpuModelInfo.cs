namespace DotLLM.Core.Models;

/// <summary>
/// Exposes GPU-specific runtime diagnostics for models running on GPU backends (CUDA, HIP).
/// </summary>
public interface IGpuModelInfo
{
    /// <summary>
    /// Warning message if VRAM usage is critical or high, or null if within normal bounds.
    /// </summary>
    string? VramWarning { get; }
}
