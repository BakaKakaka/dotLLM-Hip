using DotLLM.Cpu.Kernels;
using DotLLM.Hip;
using DotLLM.Hip.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

[Collection("HipGpu")]
[Trait("Category", "GPU-AMD")]
public class HipGatedDeltaNetKernelTests
{
    private static void ForEachDevice(Action<HipDevice, HipKernels, HipStream> action)
    {
        Assert.True(HipDevice.IsAvailable(out string? reason), reason);
        var devices = HipDevice.GetSupportedDevices();
        Assert.NotEmpty(devices);

        foreach (var device in devices)
        {
            device.MakeCurrent();
            using var stream = HipStream.Create();
            using var kernels = new HipKernels(device.Arch);
            action(device, kernels, stream);
        }
    }

    [SkippableFact]
    public unsafe void CausalConv1dSiLu_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int channels = 8192;
            var rng = new Random(42);

            Half[] inputH = new Half[channels];
            float[] inputF = new float[channels];
            for (int i = 0; i < channels; i++)
            {
                float v = (float)rng.NextDouble() * 2f - 1f;
                inputH[i] = (Half)v;
                inputF[i] = (float)inputH[i];
            }

            float[] weights = new float[channels * 4];
            for (int i = 0; i < weights.Length; i++)
                weights[i] = (float)rng.NextDouble() * 2f - 1f;

            float[] stateCpu = new float[channels * 3];
            float[] stateGpu = new float[channels * 3];
            for (int i = 0; i < stateCpu.Length; i++)
            {
                float v = (float)rng.NextDouble() * 2f - 1f;
                stateCpu[i] = v;
                stateGpu[i] = v;
            }

            // CPU reference
            float[] outputCpu = new float[channels];
            GatedDeltaNet.CausalConv1dSiLu(inputF, weights, stateCpu, outputCpu, channels);

            // GPU execution
            nuint inBytes = (nuint)(channels * sizeof(Half));
            nuint wBytes = (nuint)(channels * 4 * sizeof(float));
            nuint sBytes = (nuint)(channels * 3 * sizeof(float));
            nuint outBytes = (nuint)(channels * sizeof(Half));

            HipApi.hipMalloc(out nint dInput, inBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dWeights, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dState, sBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOutput, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pIn = inputH)
                fixed (float* pW = weights, pS = stateGpu)
                {
                    HipApi.hipMemcpy(dInput, (nint)pIn, inBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dWeights, (nint)pW, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dState, (nint)pS, sBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchCausalConv1dSiLu(dInput, dWeights, dState, dOutput, channels, stream.Handle);
                stream.Synchronize();

                Half[] outputGpuH = new Half[channels];
                float[] finalStateGpu = new float[channels * 3];

                fixed (Half* pOut = outputGpuH)
                fixed (float* pS = finalStateGpu)
                {
                    HipApi.hipMemcpy((nint)pOut, dOutput, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pS, dState, sBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                // Validate output
                for (int i = 0; i < channels; i++)
                {
                    float exp = outputCpu[i];
                    float act = (float)outputGpuH[i];
                    Assert.True(MathF.Abs(exp - act) < 5e-3f,
                        $"{device.Arch} conv out[{i}]: expected {exp}, got {act}");
                }

                // Validate updated state
                for (int i = 0; i < stateCpu.Length; i++)
                {
                    Assert.Equal(stateCpu[i], finalStateGpu[i], 5);
                }
            }
            finally
            {
                HipApi.hipFree(dInput);
                HipApi.hipFree(dWeights);
                HipApi.hipFree(dState);
                HipApi.hipFree(dOutput);
            }
        });
    }

    [SkippableFact]
    public unsafe void L2NormAndGqaRepeat_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int numKHeads = 16;
            const int numVHeads = 32;
            const int headDim = 128;

            var rng = new Random(123);
            Half[] qH = new Half[numKHeads * headDim];
            Half[] kH = new Half[numKHeads * headDim];
            float[] qF = new float[numKHeads * headDim];
            float[] kF = new float[numKHeads * headDim];

            for (int i = 0; i < qH.Length; i++)
            {
                float qv = (float)rng.NextDouble() * 4f - 2f;
                float kv = (float)rng.NextDouble() * 4f - 2f;
                qF[i] = qv; qH[i] = (Half)qv;
                kF[i] = kv; kH[i] = (Half)kv;
            }

            // CPU reference
            float[] qCpu = new float[numVHeads * headDim];
            float[] kCpu = new float[numVHeads * headDim];
            GatedDeltaNet.L2NormAndGqaRepeat(qF, kF, qCpu, kCpu, numKHeads, numVHeads, headDim);

            // GPU execution
            nuint inKBytes = (nuint)(numKHeads * headDim * sizeof(Half));
            nuint outVBytes = (nuint)(numVHeads * headDim * sizeof(Half));

            HipApi.hipMalloc(out nint dQin, inKBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dKin, inKBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dQout, outVBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dKout, outVBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = qH, pK = kH)
                {
                    HipApi.hipMemcpy(dQin, (nint)pQ, inKBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dKin, (nint)pK, inKBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchL2NormAndGqaRepeat(
                    dQin, dKin, dQout, dKout,
                    numKHeads, numVHeads, headDim, 1e-6f, stream.Handle);

                stream.Synchronize();

                Half[] qGpuH = new Half[numVHeads * headDim];
                Half[] kGpuH = new Half[numVHeads * headDim];

                fixed (Half* pQ = qGpuH, pK = kGpuH)
                {
                    HipApi.hipMemcpy((nint)pQ, dQout, outVBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pK, dKout, outVBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int i = 0; i < qCpu.Length; i++)
                {
                    float expQ = qCpu[i];
                    float actQ = (float)qGpuH[i];
                    Assert.True(MathF.Abs(expQ - actQ) < 3e-3f,
                        $"{device.Arch} Q[{i}]: expected {expQ}, got {actQ}");

                    float expK = kCpu[i];
                    float actK = (float)kGpuH[i];
                    Assert.True(MathF.Abs(expK - actK) < 3e-3f,
                        $"{device.Arch} K[{i}]: expected {expK}, got {actK}");
                }
            }
            finally
            {
                HipApi.hipFree(dQin);
                HipApi.hipFree(dKin);
                HipApi.hipFree(dQout);
                HipApi.hipFree(dKout);
            }
        });
    }

    [SkippableFact]
    public unsafe void ComputeSsmGates_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int numVHeads = 32;
            var rng = new Random(456);

            float[] betaRaw = new float[numVHeads];
            float[] alphaRaw = new float[numVHeads];
            float[] ssmA = new float[numVHeads];
            float[] ssmDtBias = new float[numVHeads];

            for (int i = 0; i < numVHeads; i++)
            {
                betaRaw[i] = (float)rng.NextDouble() * 4f - 2f;
                alphaRaw[i] = (float)rng.NextDouble() * 4f - 2f;
                ssmA[i] = -((float)rng.NextDouble() * 0.5f + 0.01f);
                ssmDtBias[i] = (float)rng.NextDouble() * 4f - 2f;
            }

            // CPU reference
            float[] betaCpu = new float[numVHeads];
            float[] gCpu = new float[numVHeads];
            GatedDeltaNet.ComputeGates(betaRaw, alphaRaw, ssmA, ssmDtBias, betaCpu, gCpu, numVHeads);

            // GPU
            nuint bytes = (nuint)(numVHeads * sizeof(float));

            HipApi.hipMalloc(out nint dBetaRaw, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dAlphaRaw, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dSsmA, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dSsmDtBias, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dBetaOut, bytes).ThrowOnError();
            HipApi.hipMalloc(out nint dGOut, bytes).ThrowOnError();

            try
            {
                fixed (float* pB = betaRaw, pA = alphaRaw, pS = ssmA, pD = ssmDtBias)
                {
                    HipApi.hipMemcpy(dBetaRaw, (nint)pB, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dAlphaRaw, (nint)pA, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dSsmA, (nint)pS, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dSsmDtBias, (nint)pD, bytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchComputeSsmGates(
                    dBetaRaw, dAlphaRaw, dSsmA, dSsmDtBias,
                    dBetaOut, dGOut, numVHeads, stream.Handle);

                stream.Synchronize();

                float[] betaGpu = new float[numVHeads];
                float[] gGpu = new float[numVHeads];

                fixed (float* pB = betaGpu, pG = gGpu)
                {
                    HipApi.hipMemcpy((nint)pB, dBetaOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pG, dGOut, bytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int i = 0; i < numVHeads; i++)
                {
                    Assert.Equal(betaCpu[i], betaGpu[i], 5);
                    Assert.Equal(gCpu[i], gGpu[i], 5);
                }
            }
            finally
            {
                HipApi.hipFree(dBetaRaw);
                HipApi.hipFree(dAlphaRaw);
                HipApi.hipFree(dSsmA);
                HipApi.hipFree(dSsmDtBias);
                HipApi.hipFree(dBetaOut);
                HipApi.hipFree(dGOut);
            }
        });
    }

    [SkippableFact]
    public unsafe void ComputeSsmGatesF16_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int numVHeads = 32;
            var rng = new Random(789);

            Half[] betaRawH = new Half[numVHeads];
            Half[] alphaRawH = new Half[numVHeads];
            float[] betaRawF = new float[numVHeads];
            float[] alphaRawF = new float[numVHeads];
            float[] ssmA = new float[numVHeads];
            float[] ssmDtBias = new float[numVHeads];

            for (int i = 0; i < numVHeads; i++)
            {
                float bVal = (float)rng.NextDouble() * 4f - 2f;
                float aVal = (float)rng.NextDouble() * 4f - 2f;
                betaRawH[i] = (Half)bVal;
                alphaRawH[i] = (Half)aVal;
                betaRawF[i] = (float)betaRawH[i];
                alphaRawF[i] = (float)alphaRawH[i];

                ssmA[i] = -((float)rng.NextDouble() * 0.5f + 0.01f);
                ssmDtBias[i] = (float)rng.NextDouble() * 4f - 2f;
            }

            // CPU reference
            float[] betaCpu = new float[numVHeads];
            float[] gCpu = new float[numVHeads];
            GatedDeltaNet.ComputeGates(betaRawF, alphaRawF, ssmA, ssmDtBias, betaCpu, gCpu, numVHeads);

            // GPU
            nuint hBytes = (nuint)(numVHeads * sizeof(Half));
            nuint fBytes = (nuint)(numVHeads * sizeof(float));

            HipApi.hipMalloc(out nint dBetaRaw, hBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dAlphaRaw, hBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dSsmA, fBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dSsmDtBias, fBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dBetaOut, fBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dGOut, fBytes).ThrowOnError();

            try
            {
                fixed (Half* pB = betaRawH, pA = alphaRawH)
                fixed (float* pS = ssmA, pD = ssmDtBias)
                {
                    HipApi.hipMemcpy(dBetaRaw, (nint)pB, hBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dAlphaRaw, (nint)pA, hBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dSsmA, (nint)pS, fBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dSsmDtBias, (nint)pD, fBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchComputeSsmGatesF16(
                    dBetaRaw, dAlphaRaw, dSsmA, dSsmDtBias,
                    dBetaOut, dGOut, numVHeads, stream.Handle);

                stream.Synchronize();

                float[] betaGpu = new float[numVHeads];
                float[] gGpu = new float[numVHeads];

                fixed (float* pB = betaGpu, pG = gGpu)
                {
                    HipApi.hipMemcpy((nint)pB, dBetaOut, fBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pG, dGOut, fBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int i = 0; i < numVHeads; i++)
                {
                    Assert.Equal(betaCpu[i], betaGpu[i], 3);
                    Assert.Equal(gCpu[i], gGpu[i], 3);
                }
            }
            finally
            {
                HipApi.hipFree(dBetaRaw);
                HipApi.hipFree(dAlphaRaw);
                HipApi.hipFree(dSsmA);
                HipApi.hipFree(dSsmDtBias);
                HipApi.hipFree(dBetaOut);
                HipApi.hipFree(dGOut);
            }
        });
    }


    [SkippableFact]
    public unsafe void RecurrentDeltaRule_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int numVHeads = 32;
            const int headKDim = 128;
            const int headVDim = 128;
            const int stateElements = numVHeads * headKDim * headVDim;

            var rng = new Random(789);

            Half[] qH = new Half[numVHeads * headKDim];
            Half[] kH = new Half[numVHeads * headKDim];
            Half[] vH = new Half[numVHeads * headVDim];
            float[] qF = new float[numVHeads * headKDim];
            float[] kF = new float[numVHeads * headKDim];
            float[] vF = new float[numVHeads * headVDim];

            for (int i = 0; i < qH.Length; i++)
            {
                float qv = (float)rng.NextDouble() * 0.2f - 0.1f;
                float kv = (float)rng.NextDouble() * 0.2f - 0.1f;
                qF[i] = qv; qH[i] = (Half)qv;
                kF[i] = kv; kH[i] = (Half)kv;
            }
            for (int i = 0; i < vH.Length; i++)
            {
                float vv = (float)rng.NextDouble() * 2f - 1f;
                vF[i] = vv; vH[i] = (Half)vv;
            }

            float[] beta = new float[numVHeads];
            float[] g = new float[numVHeads];
            for (int i = 0; i < numVHeads; i++)
            {
                beta[i] = (float)rng.NextDouble() * 0.8f + 0.1f;
                g[i] = (float)rng.NextDouble() * 0.8f + 0.1f;
            }

            float[] stateCpu = new float[stateElements];
            float[] stateGpu = new float[stateElements];
            for (int i = 0; i < stateElements; i++)
            {
                float sv = (float)rng.NextDouble() * 0.1f - 0.05f;
                stateCpu[i] = sv;
                stateGpu[i] = sv;
            }

            // CPU reference
            float[] outCpu = new float[numVHeads * headVDim];
            GatedDeltaNet.RecurrentDeltaRuleStep(qF, kF, vF, beta, g, stateCpu, outCpu, numVHeads, headKDim, headVDim);

            // GPU execution
            nuint qBytes = (nuint)(qH.Length * sizeof(Half));
            nuint kBytes = (nuint)(kH.Length * sizeof(Half));
            nuint vBytes = (nuint)(vH.Length * sizeof(Half));
            nuint gateBytes = (nuint)(numVHeads * sizeof(float));
            nuint stateBytes = (nuint)(stateElements * sizeof(float));
            nuint outBytes = (nuint)(numVHeads * headVDim * sizeof(Half));

            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dK, kBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dV, vBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dBeta, gateBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dG, gateBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dState, stateBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, outBytes).ThrowOnError();

            try
            {
                fixed (Half* pQ = qH, pK = kH, pV = vH)
                fixed (float* pB = beta, pG = g, pS = stateGpu)
                {
                    HipApi.hipMemcpy(dQ, (nint)pQ, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dK, (nint)pK, kBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dV, (nint)pV, vBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dBeta, (nint)pB, gateBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dG, (nint)pG, gateBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dState, (nint)pS, stateBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchGatedDeltaNetRecurrent(
                    dQ, dK, dV, dBeta, dG, dState, dOut,
                    numVHeads, headKDim, headVDim, stream.Handle);

                stream.Synchronize();

                Half[] outGpuH = new Half[numVHeads * headVDim];
                float[] finalStateGpu = new float[stateElements];

                fixed (Half* pO = outGpuH)
                fixed (float* pS = finalStateGpu)
                {
                    HipApi.hipMemcpy((nint)pO, dOut, outBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pS, dState, stateBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                // Validate output
                for (int i = 0; i < outCpu.Length; i++)
                {
                    float exp = outCpu[i];
                    float act = (float)outGpuH[i];
                    Assert.True(MathF.Abs(exp - act) < 5e-3f,
                        $"{device.Arch} output[{i}]: expected {exp}, got {act}");
                }

                // Validate updated state matrix
                for (int i = 0; i < stateElements; i++)
                {
                    float exp = stateCpu[i];
                    float act = finalStateGpu[i];
                    Assert.True(MathF.Abs(exp - act) < 1e-4f,
                        $"{device.Arch} state[{i}]: expected {exp}, got {act}");
                }
            }
            finally
            {
                HipApi.hipFree(dQ);
                HipApi.hipFree(dK);
                HipApi.hipFree(dV);
                HipApi.hipFree(dBeta);
                HipApi.hipFree(dG);
                HipApi.hipFree(dState);
                HipApi.hipFree(dOut);
            }
        });
    }

    [SkippableFact]
    public unsafe void GatedRmsNormSiLu_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int numVHeads = 32;
            const int headVDim = 128;
            const int total = numVHeads * headVDim;

            var rng = new Random(101);

            Half[] oH = new Half[total];
            Half[] zH = new Half[total];
            float[] oF = new float[total];
            float[] zF = new float[total];
            float[] normWeight = new float[headVDim];

            for (int i = 0; i < total; i++)
            {
                float ov = (float)rng.NextDouble() * 4f - 2f;
                float zv = (float)rng.NextDouble() * 4f - 2f;
                oF[i] = ov; oH[i] = (Half)ov;
                zF[i] = zv; zH[i] = (Half)zv;
            }
            for (int i = 0; i < headVDim; i++)
                normWeight[i] = (float)rng.NextDouble() * 1.5f + 0.5f;

            // CPU reference
            float[] outCpu = new float[total];
            GatedDeltaNet.GatedRmsNormSiLu(oF, zF, normWeight, outCpu, numVHeads, headVDim);

            // GPU execution
            nuint totalBytes = (nuint)(total * sizeof(Half));
            nuint wBytes = (nuint)(headVDim * sizeof(float));

            HipApi.hipMalloc(out nint dO, totalBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dZ, totalBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dW, wBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dOut, totalBytes).ThrowOnError();

            try
            {
                fixed (Half* pO = oH, pZ = zH)
                fixed (float* pW = normWeight)
                {
                    HipApi.hipMemcpy(dO, (nint)pO, totalBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dZ, (nint)pZ, totalBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    HipApi.hipMemcpy(dW, (nint)pW, wBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchGatedRmsNormSiLu(
                    dO, dZ, dW, dOut,
                    numVHeads, headVDim, 1e-6f, stream.Handle);

                stream.Synchronize();

                Half[] outGpuH = new Half[total];
                fixed (Half* pOut = outGpuH)
                {
                    HipApi.hipMemcpy((nint)pOut, dOut, totalBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int i = 0; i < total; i++)
                {
                    float exp = outCpu[i];
                    float act = (float)outGpuH[i];
                    Assert.True(MathF.Abs(exp - act) < 5e-3f,
                        $"{device.Arch} gated rmsnorm[{i}]: expected {exp}, got {act}");
                }
            }
            finally
            {
                HipApi.hipFree(dO);
                HipApi.hipFree(dZ);
                HipApi.hipFree(dW);
                HipApi.hipFree(dOut);
            }
        });
    }

    [SkippableFact]
    public unsafe void DeinterleaveAndMulSigmoidGate_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(out string? reason), reason);

        ForEachDevice((device, kernels, stream) =>
        {
            const int numHeads = 16;
            const int headDim = 256;
            const int totalQ = numHeads * headDim;
            const int totalQG = totalQ * 2;

            var rng = new Random(202);

            Half[] qgH = new Half[totalQG];
            float[] qgF = new float[totalQG];
            for (int i = 0; i < totalQG; i++)
            {
                float v = (float)rng.NextDouble() * 4f - 2f;
                qgH[i] = (Half)v;
                qgF[i] = (float)qgH[i];
            }

            // CPU deinterleave
            float[] qCpu = new float[totalQ];
            float[] gateCpu = new float[totalQ];
            GatedDeltaNet.DeinterleaveQGate(qgF, qCpu, gateCpu, numHeads, headDim);

            // GPU deinterleave
            nuint qgBytes = (nuint)(totalQG * sizeof(Half));
            nuint qBytes = (nuint)(totalQ * sizeof(Half));

            HipApi.hipMalloc(out nint dQg, qgBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dQ, qBytes).ThrowOnError();
            HipApi.hipMalloc(out nint dGate, qBytes).ThrowOnError();

            try
            {
                fixed (Half* pQg = qgH)
                {
                    HipApi.hipMemcpy(dQg, (nint)pQg, qgBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                }

                kernels.LaunchDeinterleaveQGate(
                    dQg, dQ, dGate,
                    numHeads, headDim, stream.Handle);

                stream.Synchronize();

                Half[] qGpuH = new Half[totalQ];
                Half[] gateGpuH = new Half[totalQ];

                fixed (Half* pQ = qGpuH, pG = gateGpuH)
                {
                    HipApi.hipMemcpy((nint)pQ, dQ, qBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    HipApi.hipMemcpy((nint)pG, dGate, qBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                }

                for (int i = 0; i < totalQ; i++)
                {
                    Assert.Equal((float)qGpuH[i], qCpu[i], 3);
                    Assert.Equal((float)gateGpuH[i], gateCpu[i], 3);
                }

                // MulSigmoidGate test
                Half[] attnOutH = new Half[totalQ];
                float[] attnOutCpu = new float[totalQ];
                for (int i = 0; i < totalQ; i++)
                {
                    float v = (float)rng.NextDouble() * 2f - 1f;
                    attnOutCpu[i] = v;
                    attnOutH[i] = (Half)v;
                }

                GatedDeltaNet.ApplySigmoidGate(attnOutCpu, gateCpu);

                HipApi.hipMalloc(out nint dAttnOut, qBytes).ThrowOnError();
                try
                {
                    fixed (Half* pA = attnOutH)
                    {
                        HipApi.hipMemcpy(dAttnOut, (nint)pA, qBytes, HipApi.hipMemcpyHostToDevice).ThrowOnError();
                    }

                    kernels.LaunchMulSigmoidGate(dAttnOut, dGate, totalQ, stream.Handle);
                    stream.Synchronize();

                    Half[] finalOutH = new Half[totalQ];
                    fixed (Half* pA = finalOutH)
                    {
                        HipApi.hipMemcpy((nint)pA, dAttnOut, qBytes, HipApi.hipMemcpyDeviceToHost).ThrowOnError();
                    }

                    for (int i = 0; i < totalQ; i++)
                    {
                        float exp = attnOutCpu[i];
                        float act = (float)finalOutH[i];
                        Assert.True(MathF.Abs(exp - act) < 5e-3f,
                            $"{device.Arch} mul sigmoid[{i}]: expected {exp}, got {act}");
                    }
                }
                finally
                {
                    HipApi.hipFree(dAttnOut);
                }
            }
            finally
            {
                HipApi.hipFree(dQg);
                HipApi.hipFree(dQ);
                HipApi.hipFree(dGate);
            }
        });
    }
}
