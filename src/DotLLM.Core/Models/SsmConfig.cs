namespace DotLLM.Core.Models;

/// <summary>
/// Configuration for hybrid linear-attention / state-space model (SSM) layers, used by Qwen 3.5 (DeltaNet).
/// </summary>
/// <param name="FullAttentionInterval">Layer period for full attention (e.g. 4 means every 4th layer is full attention).</param>
/// <param name="ConvKernel">Kernel size for 1D depthwise causal convolution (e.g. 4).</param>
/// <param name="InnerSize">Total inner value projection dimension (num_v_heads * head_v_dim, e.g. 4096).</param>
/// <param name="StateSize">Recurrent state head dimension (head_k_dim, e.g. 128).</param>
/// <param name="GroupCount">Number of key/query heads (num_k_heads, e.g. 16).</param>
/// <param name="TimeStepRank">Number of value heads (num_v_heads, e.g. 32).</param>
/// <param name="NextnPredictLayers">Number of speculative next-n draft layers.</param>
/// <param name="RopeDimensionSections">Optional M-RoPE dimension sections.</param>
public readonly record struct SsmConfig(
    int FullAttentionInterval = 4,
    int ConvKernel = 4,
    int InnerSize = 4096,
    int StateSize = 128,
    int GroupCount = 16,
    int TimeStepRank = 32,
    int NextnPredictLayers = 0,
    int[]? RopeDimensionSections = null);
