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
/// Compares CPU vs HIP GPU logits token-by-token across supported AMD GPUs (H2.7).
/// Verifies prefill and decode logit alignment and greedy token parity.
/// </summary>
[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipLogitComparisonTest
{
    private readonly ITestOutputHelper _out;

    public HipLogitComparisonTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public unsafe void CompareLogits_SmolLM_PrefillAndDecode()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        string modelPath = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf");
        Skip.If(!File.Exists(modelPath), $"SmolLM-135M Q8_0 GGUF not found at {modelPath}");

        using var gguf = GgufFile.Open(modelPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        _out.WriteLine($"Model: {config.Architecture} {config.NumLayers}L/{config.HiddenSize}H vocab={config.VocabSize}");

        int[] promptTokens = tokenizer.Encode("Tell best meal is");
        _out.WriteLine($"Prompt tokens ({promptTokens.Length}): [{string.Join(", ", promptTokens)}]");

        int[] positions = new int[promptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        // Run CPU reference
        using var cpuModel = TransformerModel.LoadFromGguf(gguf, config);
        using var cpuKv = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 64);

        // CPU Step 1: Prefill
        using var cpuLogits1 = cpuModel.Forward(promptTokens, positions, -1, cpuKv);
        float* cpuPrefillPtr = (float*)cpuLogits1.DataPointer + (long)(promptTokens.Length - 1) * config.VocabSize;
        int cpuToken1 = ArgMax(cpuPrefillPtr, config.VocabSize);

        // CPU Step 2: Decode 1
        int pos1 = promptTokens.Length;
        using var cpuLogits2 = cpuModel.Forward([cpuToken1], [pos1], -1, cpuKv);
        float* cpuDecode1Ptr = (float*)cpuLogits2.DataPointer;
        int cpuToken2 = ArgMax(cpuDecode1Ptr, config.VocabSize);

        // CPU Step 3: Decode 2
        int pos2 = pos1 + 1;
        using var cpuLogits3 = cpuModel.Forward([cpuToken2], [pos2], -1, cpuKv);
        float* cpuDecode2Ptr = (float*)cpuLogits3.DataPointer;
        int cpuToken3 = ArgMax(cpuDecode2Ptr, config.VocabSize);

        _out.WriteLine($"CPU Tokens: 1={cpuToken1} ('{tokenizer.Decode([cpuToken1])}'), " +
                       $"2={cpuToken2} ('{tokenizer.Decode([cpuToken2])}'), " +
                       $"3={cpuToken3} ('{tokenizer.Decode([cpuToken3])}')");

        // Now test each physical HIP GPU
        foreach (var device in devices)
        {
            _out.WriteLine($"\n=== Testing on {device.Name} ({device.Arch}) ===");
            using var gpuModel = HipTransformerModel.LoadFromGguf(gguf, config, device);
            using var gpuKv = gpuModel.CreateKvCache(64);

            // GPU Step 1: Prefill
            using var gpuLogits1 = gpuModel.Forward(promptTokens, positions, device.Ordinal, gpuKv);
            float* gpuPrefillPtr = (float*)gpuLogits1.DataPointer;
            var (prefillMaxDiff, prefillMeanDiff) = CompareLogits("Prefill", cpuPrefillPtr, gpuPrefillPtr, config.VocabSize);
            int gpuToken1 = ArgMax(gpuPrefillPtr, config.VocabSize);

            Assert.Equal(cpuToken1, gpuToken1);
            Assert.True(prefillMaxDiff < 1.0f, $"Prefill max diff {prefillMaxDiff} exceeds threshold on {device.Arch}");
            Assert.True(prefillMeanDiff < 0.25f, $"Prefill mean diff {prefillMeanDiff} exceeds threshold on {device.Arch}");

            // GPU Step 2: Decode 1 (feed cpuToken1)
            using var gpuLogits2 = gpuModel.Forward([cpuToken1], [pos1], device.Ordinal, gpuKv);
            float* gpuDecode1Ptr = (float*)gpuLogits2.DataPointer;
            var (dec1MaxDiff, dec1MeanDiff) = CompareLogits("Decode1", cpuDecode1Ptr, gpuDecode1Ptr, config.VocabSize);
            int gpuToken2 = ArgMax(gpuDecode1Ptr, config.VocabSize);

            Assert.Equal(cpuToken2, gpuToken2);
            Assert.True(dec1MaxDiff < 1.0f, $"Decode1 max diff {dec1MaxDiff} exceeds threshold on {device.Arch}");
            Assert.True(dec1MeanDiff < 0.25f, $"Decode1 mean diff {dec1MeanDiff} exceeds threshold on {device.Arch}");

            // GPU Step 3: Decode 2 (feed cpuToken2)
            using var gpuLogits3 = gpuModel.Forward([cpuToken2], [pos2], device.Ordinal, gpuKv);
            float* gpuDecode2Ptr = (float*)gpuLogits3.DataPointer;
            var (dec2MaxDiff, dec2MeanDiff) = CompareLogits("Decode2", cpuDecode2Ptr, gpuDecode2Ptr, config.VocabSize);
            int gpuToken3 = ArgMax(gpuDecode2Ptr, config.VocabSize);

            Assert.Equal(cpuToken3, gpuToken3);
            Assert.True(dec2MaxDiff < 1.0f, $"Decode2 max diff {dec2MaxDiff} exceeds threshold on {device.Arch}");
            Assert.True(dec2MeanDiff < 0.25f, $"Decode2 mean diff {dec2MeanDiff} exceeds threshold on {device.Arch}");
        }
    }

    private unsafe (float maxDiff, float meanDiff) CompareLogits(string step, float* cpu, float* gpu, int vocabSize)
    {
        var cpuTop = TopK(cpu, vocabSize, 5);
        var gpuTop = TopK(gpu, vocabSize, 5);

        _out.WriteLine($"--- {step} ---");
        _out.WriteLine("CPU top-5: " + string.Join(", ", cpuTop.Select(t => $"[{t.idx}]={t.val:F3}")));
        _out.WriteLine("GPU top-5: " + string.Join(", ", gpuTop.Select(t => $"[{t.idx}]={t.val:F3}")));

        float maxDiff = 0, sumDiff = 0;
        int maxDiffIdx = 0;
        for (int i = 0; i < vocabSize; i++)
        {
            float diff = MathF.Abs(cpu[i] - gpu[i]);
            sumDiff += diff;
            if (diff > maxDiff) { maxDiff = diff; maxDiffIdx = i; }
        }
        float meanDiff = sumDiff / vocabSize;
        _out.WriteLine($"Max diff: {maxDiff:F4} at [{maxDiffIdx}], Mean diff: {meanDiff:F4}");
        return (maxDiff, meanDiff);
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
