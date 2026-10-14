# Diagnostik & Konsultationsunderlag: Qwen 3.5 på AMD HIP (gfx1100 & gfx1201)

Detta dokument sammanfattar den aktuella implementationen, verifierade resultat samt specifika tekniska frågor för arkitekturkonsultation gällande **Qwen 3.5** (`qwen35` dense) i **dotLLM-Hip**.

---

## 1. Sammanfattning & Nuläge

Vi har implementerat stöd för Qwen 3.5 (hybridarkitektur: 3:1 Gated DeltaNet + Sigmoid-Gated Full Attention) för AMD RDNA3 (`gfx1100`) och RDNA4 (`gfx1201`).

### Vad som fungerar och är verifierat:
1. **AOT-kompilering & ISA:** Alla HIP-kärnor (`causal_conv1d_silu`, `l2_norm_and_gqa_repeat`, `compute_ssm_gates_f16`, `gated_delta_net_recurrent_f16`, `gated_rmsnorm_silu_f16`, `deinterleave_q_gate_f16`, `mul_sigmoid_gate_f16`) kompileras rent till HSACO för både `gfx1100` och `gfx1201` och laddas felfritt via HIP-modul-API.
2. **Enhetstester:** 7 av 7 HIP-kärntester passerar på båda fysiska GPU:er (`gfx1100` och `gfx1201`) och matchar CPU-referensresultaten med hög precision ($< 10^{-3}$ relativ felmarginal).
3. **Prestanda & Minne:**
   - Modellens vikter laddas via memory-mapping (GGUF Q4_K_L).
   - KV-cache allokeras selektivt endast för de 8 Full Attention-lagren (minskar VRAM-allokering med 75%).
   - Decode uppnår **~115 tokens/sekund** på `gfx1100`.
4. **Modelligenkänning:**
   - Single-token forward pass predicerar naturliga ord (t.ex. för `"A"` predikeras `" person"`, `" researcher"`, `" car"`, `" shop"`).
   - Med ChatML-prompt (`<|im_start|>user...`) genererar modellen korrekt `<think>` som första token, vilket visar att tokenisering och prompt-struktur tolkas rätt.

### Problemet vi observerar:
Efter att modellen genererar `<think>`, hamnar den i en upprepningsfas (ofta upprepade radbrytningar/blanksteg eller generiska tokens) istället för att genomföra ett koherent resonemang och svara. Detta indikerar en mindre matematisk avvikelse eller dimensionslayout/permutations-diskrepans i antingen DeltaNet- eller Full Attention-blocket.

---

## 2. Arkitekturdetaljer (Qwen 3.5 4B)

- **Lager totalt:** 32 trunk-lager.
- **Hybrid-mönster:**
  - Lager där `(layer_idx + 1) % 4 != 0` (lager 0, 1, 2, 4, 5, 6, ...) är **Gated DeltaNet** (totalt 24 st).
  - Var 4:e lager (`layer_idx = 3, 7, 11, 15, 19, 23, 27, 31`) är **Full Attention** (totalt 8 st).
- **Hidden size:** 2560
- **Intermediate size:** 9216

### A. Gated DeltaNet-lager
- $H_k = 16$ (num_k_heads), $d_k = 128$ $\rightarrow$ $qk\_dim = 2048$
- $H_v = 32$ (num_v_heads), $d_v = 128$ $\rightarrow$ $v\_dim = 4096$
- Konvolution: `ssm_conv1d.weight` har form `[4, 8192]`, djupgående 1D causalt filter över $Q, K, V$ ($2048 + 2048 + 4096 = 8192$).
- Rekurrenstillstånd: $S \in \mathbb{R}^{32 \times 128 \times 128}$ (float32).

### B. Full Attention-lager
- $H_q = 16$, $d = 256$
- $H_{kv} = 4$, $d = 256$
- $Q$-projektionen (`attn_q.weight`) genererar $16 \times 256 \times 2 = 8192$ element (både $Q$ och Attention Gate).
- Partiell RoPE: `rope_dim = 64` av $d = 256$.

---

## 3. Frågor att rådfråga experten om

Här är de 5 kritiska punkterna där en implementationsdetalj i Qwen 3.5 / Qwen3-Next kan skilja sig:

### Fråga 1: Hur repeteras/ordnas Q och K från 16 till 32 huvuden i DeltaNet?
I Gated DeltaNet har vi 16 $K$-huvuden och 32 $V$-huvuden ($H_v / H_k = 2$).
Efter L2-norm på $Q$ och $K$:
- **Alternativ A (Contiguous/Interleaved repeat):**
  Huvud $hk \in [0, 16)$ mappas till $hv = hk \times 2$ och $hv = hk \times 2 + 1$.
  (dvs. `[h0, h0, h1, h1, h2, h2, ...]`)
- **Alternativ B (Block repeat):**
  Huvud $hk \in [0, 16)$ mappas till $hv = hk$ och $hv = hk + 16$.
  (dvs. `[h0, h1, ..., h15, h0, h1, ..., h15]`)

*Vilken av dessa två layouter förväntar sig Qwen 3.5 för de 32 rekurrenstillstånden och $V$-huvudena?*

---

### Fråga 2: Uppdateras tillståndet $S_t$ FÖRE eller EFTER att output $o_t$ beräknas?
I vår implementation av Delta Rule:
1. $r_t = g_t \cdot S_{t-1}^T \tilde{k}_t$
2. $\delta_t = \beta_t (v_t - r_t)$
3. $S_t = g_t \cdot S_{t-1} + \tilde{k}_t \otimes \delta_t$
4. $o_t = S_t^T \tilde{q}_t$  *(dvs. läser från det nyuppdaterade tillståndet $S_t$)*

*Eller ska output beräknas från tillståndet före $\delta$-uppdateringen, dvs:*
$$o_t = (g_t \cdot S_{t-1})^T \tilde{q}_t \quad \text{eller} \quad o_t = S_{t-1}^T \tilde{q}_t \text{?}$$

---

### Fråga 3: De-interleaving av `attn_q.weight` i Full Attention
`attn_q.weight` projicerar till storlek `[num_heads * head_dim * 2]` ($16 \times 512 = 8192$).
Vår kärna separerar per huvud:
- För huvud $h$: de första 256 värdena är $Q_h$, de sista 256 värdena är $Gate_h$.
  $Q = [Q_0, Q_1, ..., Q_{15}]$
  $Gate = [G_0, G_1, ..., G_{15}]$

*Stämmer detta mönster per huvud, eller är hela projektionen delad i två halvor över hela tensorn (dvs. de första 4096 elementen är hela $Q$, och de sista 4096 elementen är hela $Gate$)?*

---

### Fråga 4: Ordning på QK-Norm och RoPE i Full Attention
Qwen 3.5 har per-head RMSNorm på både $Q$ och $K$ (`attn_q_norm.weight` och `attn_k_norm.weight`) samt partiell RoPE (`rope_dim = 64`).
I vår pipeline:
1. RMSNorm appliceras per huvud över hela dimensionen ($d = 256$).
2. Därefter appliceras Rotary Embedding på de första 64 elementen i varje huvud.

*Appliceras QK-norm FÖRE RoPE, eller appliceras RoPE före QK-norm i Qwen 3.5?*

---

### Fråga 5: Causal Conv1D viktorientering i GGUF
`ssm_conv1d.weight` har formen `[4, 8192]`.
För kanal $c$: har vi 4 vikter $W[0], W[1], W[2], W[3]$.
Givet historiken $s_0$ (äldst), $s_1$, $s_2$ och nuvarande token $x$:
Beräknar vi:
$$\text{conv} = W[0] \cdot s_0 + W[1] \cdot s_1 + W[2] \cdot s_2 + W[3] \cdot x$$
eller är vikterna lagrade i omvänd tidsordning i GGUF (dvs $W[0]$ multipliceras med $x$ och $W[3]$ med $s_0$)?

---

## 4. Reproduktionskommandon

För att demonstrera beteendet direkt på testmaskinen:

```powershell
# Kör forward-pass diagnostik (visar top-10 logit-prediktioner)
dotnet run --project src/DotLLM.Cli -- debug forward-pass "c:\Users\benny\.dotllm\models\Qwen\Qwen3.5-4b-GGUF\Qwen_Qwen3.5-4B-Q4_K_L.gguf" -p "<|im_start|>user`nWhat is the capital of France?<|im_end|>`n<|im_start|>assistant`n" --device hip:gfx1100

# Kör fullständig textgenerering
dotnet run --project src/DotLLM.Cli -c Release -- run "c:\Users\benny\.dotllm\models\Qwen\Qwen3.5-4b-GGUF\Qwen_Qwen3.5-4B-Q4_K_L.gguf" -p "<|im_start|>user`nWhat is the capital of France?<|im_end|>`n<|im_start|>assistant`n" -n 64 --device hip:gfx1100
```
