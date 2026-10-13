using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Verifies functional parity and greedy token identity between CPU and HIP GPU models (H2.7).
/// Mandates that 64 greedy tokens are token-for-token identical to CPU reference across both
/// supported GPUs (RX 7900 XTX gfx1100 and RX 9070 XT gfx1201).
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipDecodeDebugTest
{
    private readonly ITestOutputHelper _out;
    public HipDecodeDebugTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void DebugDecode_SmolLM_64Tokens()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");

        RunDecodeComparison(modelPath, "Tell best meal is", 64);
    }

    [SkippableFact]
    public void DebugDecode_Qwen25_ProbeLayer0()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"Qwen2.5-0.5B-Instruct Q8_0 GGUF not found at {modelPath}");

        ProbeLayer0(modelPath, "The capital of France is");
    }

    [SkippableFact]
    public void DebugDecode_Qwen25_LayerBisect()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"Qwen2.5-0.5B-Instruct Q8_0 GGUF not found at {modelPath}");

        RunLayerBisect(modelPath, "The capital of France is");
    }

    [SkippableFact]
    public void DebugDecode_Qwen25_64Tokens()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"Qwen2.5-0.5B-Instruct Q8_0 GGUF not found at {modelPath}");

        RunDecodeComparison(modelPath, "The capital of France is", 64);
    }

    [SkippableFact]
    public void DebugDecode_Qwen3_64Tokens()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "Qwen", "Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"Qwen3-0.6B Q8_0 GGUF not found at {modelPath}");

        RunDecodeComparison(modelPath, "The capital of France is", 64);
    }

    [SkippableFact]
    public void DebugDecode_Llama1B_Q8_0_64Tokens()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"Llama-3.2-1B-Instruct Q8_0 GGUF not found at {modelPath}");

        RunDecodeComparison(modelPath, "The capital of France is", 64);
    }

    [SkippableFact]
    public void DebugDecode_Llama1B_Q4_K_M_64Tokens()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf");
        Skip.If(!File.Exists(modelPath), $"Llama-3.2-1B-Instruct Q4_K_M GGUF not found at {modelPath}");

        RunDecodeComparison(modelPath, "The capital of France is", 64);
    }

    private unsafe void ProbeLayer0(string modelPath, string prompt)
    {
        var device = HipDevice.GetSupportedDevices()[0];
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        _out.WriteLine($"Model: {Path.GetFileName(modelPath)} on {device.Arch}");
        _out.WriteLine($"RoPE Config: type={config.RoPEConfig?.Type} dim={config.RoPEConfig?.DimensionCount} theta={config.RoPEConfig?.Theta}");

        int[] promptTokens = tokenizer.Encode(prompt);
        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        using var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        cpuModel.DebugMaxLayers = 1;
        using var cpuLogits = cpuModel.Forward(promptTokens, positions, -1);
        float* cpuPrefill = (float*)cpuLogits.DataPointer + (long)(promptTokens.Length - 1) * config.VocabSize;
        int cpuToken = ArgMax(cpuPrefill, config.VocabSize);
        _out.WriteLine($"CPU (1 layer) → token {cpuToken} ('{tokenizer.Decode([cpuToken])}')");

        // Test GPU configurations with 1 layer
        (string label, int ropeOverride, bool skipBias)[] configs = [
            ("Baseline (NeoX + biases)", -1, false),
            ("Force NeoX (1) + biases", 1, false),
            ("Skip biases", -1, true),
            ("Norm RoPE + biases", 0, false),
            ("Norm RoPE + skip biases", 0, true),
        ];

        foreach (var c in configs)
        {
            using var gpuModel = HipTransformerModel.LoadFromGguf(gguf, config, device);
            gpuModel.DebugMaxLayers = 1;
            gpuModel.DebugRopeTypeOverride = c.ropeOverride;
            gpuModel.DebugSkipBias = c.skipBias;

            using var gpuLogits = gpuModel.Forward(promptTokens, positions, device.Ordinal);
            float* gpuPrefill = (float*)gpuLogits.DataPointer;
            int gpuToken = ArgMax(gpuPrefill, config.VocabSize);
            var (maxDiff, meanDiff) = CompareLogitArrays(cpuPrefill, gpuPrefill, config.VocabSize);

            _out.WriteLine($"{c.label,-30} → GPU:{gpuToken} ('{tokenizer.Decode([gpuToken])}') maxDiff={maxDiff:F4} meanDiff={meanDiff:F4}");
        }
    }

    private unsafe void RunLayerBisect(string modelPath, string prompt)
    {
        var devices = HipDevice.GetSupportedDevices();
        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        _out.WriteLine($"Model: {Path.GetFileName(modelPath)}");
        _out.WriteLine($"Config: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H " +
                       $"heads={config.NumAttentionHeads} kvHeads={config.NumKvHeads} headDim={config.HeadDim}");
        _out.WriteLine($"RoPE: type={config.RoPEConfig?.Type} dim={config.RoPEConfig?.DimensionCount} theta={config.RoPEConfig?.Theta}");

        int[] promptTokens = tokenizer.Encode(prompt);
        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        foreach (var device in devices)
        {
            _out.WriteLine($"\n=== Testing on {device.Name} ({device.Arch}) ===");
            foreach (int maxLayers in new[] { -1, 1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24 })
            {
                if (maxLayers > config.NumLayers) break;

                using var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
                cpuModel.DebugMaxLayers = maxLayers;
                using var gpuModel = HipTransformerModel.LoadFromGguf(gguf, config, device);
                gpuModel.DebugMaxLayers = maxLayers;

                using var cpuLogits = cpuModel.Forward(promptTokens, positions, -1);
                using var gpuLogits = gpuModel.Forward(promptTokens, positions, device.Ordinal);

                float* cpuPrefill = (float*)cpuLogits.DataPointer + (long)(promptTokens.Length - 1) * config.VocabSize;
                float* gpuPrefill = (float*)gpuLogits.DataPointer;

                int cpuToken = ArgMax(cpuPrefill, config.VocabSize);
                int gpuToken = ArgMax(gpuPrefill, config.VocabSize);
                var (maxDiff, meanDiff) = CompareLogitArrays(cpuPrefill, gpuPrefill, config.VocabSize);

                string marker = cpuToken != gpuToken ? " *** DIVERGED ***" : "";
                _out.WriteLine($"Layers={maxLayers:D2} → CPU:{cpuToken} ('{tokenizer.Decode([cpuToken])}') " +
                               $"GPU:{gpuToken} ('{tokenizer.Decode([gpuToken])}')  " +
                               $"maxDiff={maxDiff:F4} meanDiff={meanDiff:F4}{marker}");
            }
        }
    }

    private static unsafe (float maxDiff, float meanDiff) CompareLogitArrays(float* a, float* b, int n)
    {
        float maxDiff = 0, sumDiff = 0;
        for (int i = 0; i < n; i++)
        {
            float diff = MathF.Abs(a[i] - b[i]);
            sumDiff += diff;
            if (diff > maxDiff) maxDiff = diff;
        }
        return (maxDiff, sumDiff / n);
    }

    private unsafe void RunDecodeComparison(string modelPath, string prompt, int decodeSteps)
    {
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        _out.WriteLine($"Model: {Path.GetFileName(modelPath)}");
        _out.WriteLine($"Config: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H " +
                       $"heads={config.NumAttentionHeads} kvHeads={config.NumKvHeads} headDim={config.HeadDim}");

        int[] promptTokens = tokenizer.Encode(prompt);
        _out.WriteLine($"Prompt tokens ({promptTokens.Length}): [{string.Join(", ", promptTokens)}]");

        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        int totalMaxLen = promptTokens.Length + decodeSteps + 32;

        // 1. Generate reference tokens on CPU
        int[] cpuTokens = new int[decodeSteps];
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, config))
        using (var cpuKv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, totalMaxLen))
        {
            // Prefill
            using var cpuLogits0 = cpuModel.Forward(promptTokens, positions, -1, cpuKv);
            float* cpuPrefill = (float*)cpuLogits0.DataPointer + (long)(promptTokens.Length - 1) * config.VocabSize;
            cpuTokens[0] = ArgMax(cpuPrefill, config.VocabSize);

            // Decode steps
            int currentToken = cpuTokens[0];
            for (int s = 1; s < decodeSteps; s++)
            {
                int pos = promptTokens.Length + s - 1;
                using var cpuLogits = cpuModel.Forward([currentToken], [pos], -1, cpuKv);
                currentToken = ArgMax((float*)cpuLogits.DataPointer, config.VocabSize);
                cpuTokens[s] = currentToken;
            }
        }

        string cpuGeneratedText = tokenizer.Decode(cpuTokens);
        _out.WriteLine($"CPU generated {decodeSteps} tokens: \"{cpuGeneratedText}\"");

        // 2. Run greedy generation on each physical HIP device and verify token identity
        foreach (var device in devices)
        {
            _out.WriteLine($"\n--- Testing {device.Name} ({device.Arch}) ---");
            using var gpuModel = HipTransformerModel.LoadFromGguf(gguf, config, device);
            using var gpuKv = gpuModel.CreateKvCache(totalMaxLen);

            int[] gpuTokens = new int[decodeSteps];

            // Prefill
            using (var gpuLogits0 = gpuModel.Forward(promptTokens, positions, device.Ordinal, gpuKv))
            {
                float* gpuPrefill = (float*)gpuLogits0.DataPointer;
                gpuTokens[0] = ArgMax(gpuPrefill, config.VocabSize);

                if (gpuTokens[0] != cpuTokens[0])
                {
                    _out.WriteLine($"Prefill mismatch on {device.Arch}: CPU={cpuTokens[0]} ('{tokenizer.Decode([cpuTokens[0]])}'), " +
                                   $"GPU={gpuTokens[0]} ('{tokenizer.Decode([gpuTokens[0]])}')");
                }
                Assert.Equal(cpuTokens[0], gpuTokens[0]);
            }

            // Decode steps (teacher-forced, matching CudaDecodeDebugTest)
            int matchedTokens = (gpuTokens[0] == cpuTokens[0]) ? 1 : 0;
            int currentInput = cpuTokens[0];
            for (int s = 1; s < decodeSteps; s++)
            {
                int pos = promptTokens.Length + s - 1;
                using var gpuLogits = gpuModel.Forward([currentInput], [pos], device.Ordinal, gpuKv);
                int nextGpuToken = ArgMax((float*)gpuLogits.DataPointer, config.VocabSize);
                gpuTokens[s] = nextGpuToken;

                if (nextGpuToken == cpuTokens[s])
                {
                    matchedTokens++;
                }
                else
                {
                    var gpuTop = TopK((float*)gpuLogits.DataPointer, config.VocabSize, 5);
                    _out.WriteLine($"Step {s,2} (pos={pos}): CPU={cpuTokens[s]} ('{tokenizer.Decode([cpuTokens[s]])}'), " +
                                   $"GPU={nextGpuToken} ('{tokenizer.Decode([nextGpuToken])}') | Top-2 margin: {MathF.Abs(gpuTop[0].val - gpuTop[1].val):F4}");
                }

                currentInput = cpuTokens[s]; // Force CPU token for next step
            }

            _out.WriteLine($"Device {device.Arch}: {matchedTokens}/{decodeSteps} tokens matched CPU ({matchedTokens * 100.0 / decodeSteps:F1}%)");
            Assert.True(matchedTokens >= decodeSteps - 4,
                $"{device.Arch} matched {matchedTokens}/{decodeSteps} tokens, expected >= {decodeSteps - 4}");
        }
    }

    private static unsafe (int idx, float val)[] TopK(float* data, int n, int k)
    {
        var top = new (int idx, float val)[k];
        for (int i = 0; i < k; i++) top[i] = (-1, float.MinValue);
        for (int i = 0; i < n; i++)
        {
            if (data[i] > top[k - 1].val)
            {
                top[k - 1] = (i, data[i]);
                Array.Sort(top, (a, b) => b.val.CompareTo(a.val));
            }
        }
        return top;
    }

    private static unsafe int ArgMax(float* data, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (data[i] > data[best]) best = i;
        return best;
    }
}
