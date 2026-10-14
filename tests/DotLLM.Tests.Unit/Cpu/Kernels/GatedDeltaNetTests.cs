using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

public class GatedDeltaNetTests
{
    [Fact]
    public void CausalConv1dSiLu_SingleStep_ComputesAndUpdatesState()
    {
        const int channels = 4;
        // Weights: 4 channels, each has 4 weights [w0, w1, w2, w3]
        float[] weights = [
            1.0f, 0.5f, 0.25f, 0.1f,
            0.1f, 0.2f, 0.3f, 0.4f,
            -0.5f, 0.5f, -0.5f, 0.5f,
            2.0f, 1.0f, 0.5f, 0.25f
        ];

        // Previous state: [s0, s1, s2] per channel
        float[] state = [
            1.0f, 2.0f, 3.0f,
            0.5f, 0.5f, 0.5f,
            1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 2.0f
        ];

        float[] input = [4.0f, 1.0f, -1.0f, 2.0f];
        float[] output = new float[channels];

        GatedDeltaNet.CausalConv1dSiLu(input, weights, state, output, channels);

        // Channel 0:
        // conv = 1*1 + 0.5*2 + 0.25*3 + 0.1*4 = 1 + 1 + 0.75 + 0.4 = 3.15
        // silu = 3.15 / (1 + exp(-3.15))
        float expectedConv0 = 3.15f;
        float expectedSilu0 = expectedConv0 / (1.0f + MathF.Exp(-expectedConv0));
        Assert.Equal(expectedSilu0, output[0], 4);

        // Verify state shifted:
        // Channel 0 state should now be [2.0, 3.0, 4.0]
        Assert.Equal(2.0f, state[0]);
        Assert.Equal(3.0f, state[1]);
        Assert.Equal(4.0f, state[2]);
    }

    [Fact]
    public void L2NormAndGqaRepeat_NormalizesAndExpands()
    {
        int numKHeads = 2;
        int numVHeads = 4;
        int headDim = 4;

        float[] q = [
            3.0f, 4.0f, 0.0f, 0.0f,    // norm = 5, sqrt(4) = 2 -> scale = 1/(5*2) = 0.1
            1.0f, 1.0f, 1.0f, 1.0f     // norm = 2, sqrt(4) = 2 -> scale = 1/(2*2) = 0.25
        ];

        float[] k = [
            0.0f, 6.0f, 8.0f, 0.0f,    // norm = 10 -> scale = 0.1
            2.0f, 0.0f, 0.0f, 0.0f     // norm = 2 -> scale = 0.5
        ];

        float[] qOut = new float[numVHeads * headDim];
        float[] kOut = new float[numVHeads * headDim];

        GatedDeltaNet.L2NormAndGqaRepeat(q, k, qOut, kOut, numKHeads, numVHeads, headDim);

        // Head 0 of Q expanded to HV 0 and 2 (block/tiled repeat):
        // 3 * 0.1 = 0.3, 4 * 0.1 = 0.4
        Assert.Equal(0.3f, qOut[0], 4);
        Assert.Equal(0.4f, qOut[1], 4);
        Assert.Equal(0.3f, qOut[8], 4); // repeated head 0 at block 1 (hv = 2)
        Assert.Equal(0.4f, qOut[9], 4);

        // Head 0 of K expanded to HV 0 and 2 (block/tiled repeat):
        // 6 * 0.1 = 0.6, 8 * 0.1 = 0.8
        Assert.Equal(0.6f, kOut[1], 4);
        Assert.Equal(0.8f, kOut[2], 4);
        Assert.Equal(0.6f, kOut[9], 4); // repeated head 0 at block 1 (hv = 2)
        Assert.Equal(0.8f, kOut[10], 4);
    }


    [Fact]
    public void ComputeGates_CalculatesCorrectDecayAndBeta()
    {
        int numVHeads = 2;
        float[] betaRaw = [0.0f, 2.0f];
        float[] alphaRaw = [1.0f, -2.0f];
        float[] ssmA = [-0.5f, -0.2f];
        float[] ssmDtBias = [0.5f, 1.0f];

        float[] beta = new float[numVHeads];
        float[] g = new float[numVHeads];

        GatedDeltaNet.ComputeGates(betaRaw, alphaRaw, ssmA, ssmDtBias, beta, g, numVHeads);

        // Head 0: beta = sigmoid(0) = 0.5
        Assert.Equal(0.5f, beta[0], 5);

        // Head 0: u = 1.0 + 0.5 = 1.5, softplus(1.5) = ln(1 + exp(1.5))
        float sp0 = MathF.Log(1.0f + MathF.Exp(1.5f));
        float expectedG0 = MathF.Exp(-0.5f * sp0);
        Assert.Equal(expectedG0, g[0], 4);
    }

    [Fact]
    public void RecurrentDeltaRuleStep_SingleHead_MatchesManualTrace()
    {
        // 1 head, dk = 2, dv = 2
        int numVHeads = 1;
        int headKDim = 2;
        int headVDim = 2;

        float[] q = [1.0f, 0.0f];
        float[] k = [0.0f, 1.0f];
        float[] v = [2.0f, 3.0f];
        float[] beta = [0.5f];
        float[] g = [0.8f];

        // Initial state S:
        // [1.0, 2.0]
        // [3.0, 4.0]
        float[] state = [1.0f, 2.0f, 3.0f, 4.0f];
        float[] output = new float[2];

        // Trace:
        // 1. Decay S by g = 0.8:
        // S = [0.8, 1.6]
        //     [2.4, 3.2]
        // 2. Retrieved r = S^T * k:
        // k = [0, 1]
        // r_0 = S_{0,0}*0 + S_{1,0}*1 = 2.4
        // r_1 = S_{0,1}*0 + S_{1,1}*1 = 3.2
        // 3. delta = beta * (v - r):
        // delta_0 = 0.5 * (2.0 - 2.4) = -0.2
        // delta_1 = 0.5 * (3.0 - 3.2) = -0.1
        // 4. Update S <- S + k (x) delta:
        // k = [0, 1]
        // row 0: + 0 * delta -> [0.8, 1.6]
        // row 1: + 1 * delta -> [2.4 - 0.2, 3.2 - 0.1] = [2.2, 3.1]
        // 5. Output o = S^T * q:
        // q = [1, 0]
        // o_0 = S_{0,0}*1 + S_{1,0}*0 = 0.8
        // o_1 = S_{0,1}*1 + S_{1,1}*0 = 1.6

        GatedDeltaNet.RecurrentDeltaRuleStep(q, k, v, beta, g, state, output, numVHeads, headKDim, headVDim);

        Assert.Equal(0.8f, output[0], 4);
        Assert.Equal(1.6f, output[1], 4);

        // Check updated state:
        Assert.Equal(0.8f, state[0], 4);
        Assert.Equal(1.6f, state[1], 4);
        Assert.Equal(2.2f, state[2], 4);
        Assert.Equal(3.1f, state[3], 4);
    }

    [Fact]
    public void GatedRmsNormSiLu_ComputesNormAndGating()
    {
        int numVHeads = 1;
        int headVDim = 2;

        float[] o = [3.0f, 4.0f]; // sumSq = 9 + 16 = 25, mean = 12.5, rms = sqrt(12.5) ~ 3.5355339
        float[] z = [0.0f, 1.0f];
        float[] normWeight = [2.0f, 1.0f];
        float[] y = new float[2];

        GatedDeltaNet.GatedRmsNormSiLu(o, z, normWeight, y, numVHeads, headVDim);

        float rms = MathF.Sqrt(12.5f + 1e-6f);
        float silu0 = 0.0f;
        float silu1 = 1.0f / (1.0f + MathF.Exp(-1.0f));

        Assert.Equal((3.0f / rms) * 2.0f * silu0, y[0], 4);
        Assert.Equal((4.0f / rms) * 1.0f * silu1, y[1], 4);
    }

    [Fact]
    public void DeinterleaveQGate_SplitsHeadsCorrectly()
    {
        int numHeads = 2;
        int headDim = 3;

        // Head 0: Q=[1, 2, 3], Gate=[4, 5, 6]
        // Head 1: Q=[7, 8, 9], Gate=[10, 11, 12]
        float[] qg = [
            1f, 2f, 3f, 4f, 5f, 6f,
            7f, 8f, 9f, 10f, 11f, 12f
        ];

        float[] q = new float[6];
        float[] gate = new float[6];

        GatedDeltaNet.DeinterleaveQGate(qg, q, gate, numHeads, headDim);

        Assert.Equal([1f, 2f, 3f, 7f, 8f, 9f], q);
        Assert.Equal([4f, 5f, 6f, 10f, 11f, 12f], gate);
    }
}
