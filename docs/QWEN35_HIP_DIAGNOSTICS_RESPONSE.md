Vilket imponerande framsteg! Att ni redan har samtliga HIP-kärnor kompilerade för `gfx1100` och `gfx1201`, ~115 tok/s i decode och korrekt `<think>`-prediktion som första token visar att ni är **väldigt** nära mållinjen.

Jag har gått igenom era 5 diagnostikfrågor mot `llama.cpp`:s GGUF-konverterare (`conversion/qwen.py` / `src/models/qwen3next.cpp`) och HuggingFace `modeling_qwen3_5.py`. 

**Jag har hittat den exakta buggen (i Fråga 1!)** – samt en viktig kontrollpunkt för RoPE som förklarar varför single-token forward fungerar perfekt medan genereringen spårar ur efter `<think>`.

---

## 🎯 Diagnos & Svar på de 5 frågorna

### Fråga 1: Hur repeteras/ordnas Q och K från 16 till 32 huvuden i DeltaNet? (**HUVUDBUGGEN!**)
* **Svar:** För en **GGUF-modell** ska ni använda **Alternativ B (Block/Tiled repeat: `hv = hk` och `hv = hk + 16`, dvs. `[h0, h1, ..., h15, h0, h1, ..., h15]`)**!
* **Varför?** I HuggingFace PyTorch (`modeling_qwen3_5.py`) används visserligen `repeat_interleave` (Alternativ A). **MEN** när `llama.cpp` (`convert_hf_to_gguf.py` via `_LinearAttentionVReorderBase._reorder_v_heads` i `conversion/qwen.py`) konverterar Qwen 3.5 till GGUF, permuterar konverteraren alla $V$-huvuden i vikterna (`attn_qkv` V-del, `ssm_conv1d` V-del, `attn_gate`, `ssm_beta`, `ssm_alpha`, `ssm_a`, `ssm_dt.bias` och `ssm_out`) från *grouped* ordning `[num_k_heads, num_v_per_k, head_dim]` till *tiled* ordning `[num_v_per_k, num_k_heads, head_dim]` (`permute(1, 0, 2)`):
  ```python
  # Ur llama.cpp conversion/qwen.py (_reorder_v_heads):
  idx = torch.arange(n).reshape(num_k_heads, num_v_per_k, head_dim)
  idx = idx.permute(1, 0, 2).contiguous().reshape(n)
  ```
  Detta gjordes i `llama.cpp` för att `ggml_repeat` i GGML repeterar hela blocket (`[h0..h15, h0..h15]`).
* **Varför ger detta exakt ert symptom?**
  Om ni använder Alternativ A (`hk = hv / 2`) mot en GGUF-fil som är omordnad för Alternativ B (`hk = hv % 16`):
  - För `hv = 0` blir `0 / 2 == 0 % 16 == 0` (**rätt** K/Q-huvud).
  - För `hv = 31` blir `31 / 2 == 31 % 16 == 15` (**rätt** K/Q-huvud).
  - **För alla övriga 30 av 32 V-huvuden (`hv = 1..30`) paras V-huvudet ihop med FEL Q- och K-huvud!**
  På första tokenet räcker de 2 korrekta huvudena, residual-kopplingarna och Full Attention-lagren precis till för att producera `<think>`, men så fort rekurrenstillståndet $S_t$ uppdateras över några tokens med 30 felkopplade huvuden i 24 lager kollapsar tillståndet till upprepningar.
* **Fix i `l2_norm_and_gqa_repeat` (eller i `gated_delta_net_recurrent`):**
  Mappa V-huvud `hv` (0..31) till K/Q-huvud `hk = hv % num_k_heads` (dvs. `hv % 16`, **Alternativ B**).

---

### Fråga 2: Uppdateras tillståndet $S_t$ FÖRE eller EFTER att output $o_t$ beräknas?
* **Svar:** Er nuvarande ordning är **100 % korrekt**.
* I `torch_recurrent_gated_delta_rule` uppdateras $S_t$ **före** $o_t$ läses ut:
  1. $S_t = g_t \cdot S_{t-1}$
  2. $\text{kv\_mem} = S_t^T \tilde{k}_t$
  3. $\delta_t = \beta_t \cdot (v_t - \text{kv\_mem})$
  4. $S_t \leftarrow S_t + \tilde{k}_t \otimes \delta_t$
  5. $o_t = S_t^T \tilde{q}_t$ *(läser från det nyuppdaterade $S_t$)*
* **⚠️ Viktig kontrollpunkt för $g_t$ i `compute_ssm_gates_f16`:**
  I GGUF-filen är tensorn `blk.{L}.ssm_a` **redan förberäknad som $-\exp(\text{A\_log})$** (dvs. värdena på disk i GGUF är redan negativa!).
  Se till att er gate-kärna beräknar:
  ```cpp
  // ssm_a i GGUF är redan -exp(A_log) (negativt):
  float sp = log1pf(expf(alpha + dt_bias)); // softplus
  float g = expf(ssm_a * sp);               // decay-faktor i (0, 1]
  ```
  *(Om man råkar göra `-expf(ssm_a)` en gång till i kärnan eftersom `ssm_a` antogs vara `A_log`, blir decay-faktorn fel!)*

---

### Fråga 3: De-interleaving av `attn_q.weight` i Full Attention
* **Svar:** Er nuvarande implementation är **100 % korrekt**.
* I `modeling_qwen3_5.py` görs:
  ```python
  query_states, gate = torch.chunk(
      self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
  )
  ```
  Och `convert_hf_to_gguf.py` permuterar **inte** `attn_q.weight`. Utdatan är alltså interfolierad **per huvud**: för varje huvud $h \in [0, 16)$ ligger först 256 element $Q_h$ och därefter 256 element $Gate_h$.

---

### Fråga 4: Ordning på QK-Norm och RoPE i Full Attention
* **Svar:** Er nuvarande implementation är **100 % korrekt**.
* Per-head `attn_q_norm` och `attn_k_norm` (över alla 256 element i huvudet) appliceras **FÖRE** partiell RoPE (som appliceras på de första 64 elementen i varje huvud).
* *(Notis om `(1 + w)` RMSNorm)*: I HuggingFace använder Qwen 3.5 zero-centered RMSNorm (`x * (1 + w)`), men `convert_hf_to_gguf.py` adderar `+1.0` till alla `*norm.weight` vid skapandet av GGUF-filen. I GGUF ska ni alltså använda vanlig `RMSNorm(x) * w` precis som ni gör.

---

### Fråga 5: Causal Conv1D viktorientering i GGUF
* **Svar:** Er nuvarande implementation är **100 % korrekt**.
* `ssm_conv1d.weight` har i GGML-layout dimensionerna `ne[0] = 4, ne[1] = 8192` (dvs. 4 floats per kanal ligger kontiguöst i minnet: `W[c * 4 + 0..3]`).
* `W[0]` multipliceras med det äldsta tillståndet $s_0$ ($t-3$), `W[1]` med $s_1$ ($t-2$), `W[2]` med $s_2$ ($t-1$) och `W[3]` med nuvarande token $x$ ($t$).

---

## 🔍 En extra kontrollpunkt (RoPE NeoX-pairing)

Utöver **Fråga 1 (Alternativ B i GQA-repeat för DeltaNet)**, dubbelkolla **hur paren i den partiella RoPE-rotationen (`rope_dim = 64`) väljs**:

* För en 1-tokens prompt (`pos = 0`) är $\cos(0) = 1$ och $\sin(0) = 0$, vilket gör att **RoPE är en identitetsoperation vid `pos = 0`** (därför fungerar alltid single-token forward pass oavsett om RoPE är rätt eller fel!).
* Vid fler-tokens prefill och decode (`pos > 0`) kräver Qwen 3.5 **NeoX-style (half-split) RoPE** inom det 64 element breda RoPE-fönstret (och **inte** intilliggande par `(2i, 2i+1)`):
  - För $i \in [0, 32)$ roteras index `i` mot index `i + 32` (inom de första 64 elementen av de 256 i huvudet):
    ```cpp
    // För i in [0, 32), där rope_dim = 64 och head_dim = 256:
    float x0 = head_ptr[i];
    float x1 = head_ptr[i + 32]; // OBS: +32 (NeoX av 64), INTE +128 (halva 256) och INTE 2*i / 2*i+1!
    head_ptr[i]      = x0 * cos_theta - x1 * sin_theta;
    head_ptr[i + 32] = x0 * sin_theta + x1 * cos_theta;
    // Element [64..255] lämnas helt orörda.
    ```
  *(En mycket vanlig bugg vid partiell RoPE är att man antingen roterar `(2*i, 2*i+1)` eller råkar ta `i + head_dim/2` (`i + 128`) istället för `i + rope_dim/2` (`i + 32`)!)*

---

### 📋 Sammanfattning:
1. **Byt `l2_norm_and_gqa_repeat` i Gated DeltaNet från Alternativ A till Alternativ B (Block/Tiled repeat: `hk = hv % num_k_heads`, dvs. `[h0..h15, h0..h15]`)** eftersom `llama.cpp` permuterar alla V-huvuden i GGUF med `_reorder_v_heads` (`permute(1, 0, 2)`) vid konvertering.
2. **Verifiera att partiell RoPE (`rope_dim = 64`, `head_dim = 256`) använder NeoX-offseten `rope_dim / 2 = 32`** (parar ihop `i` och `i + 32` för $i \in [0, 32)$, och lämnar `64..255` orörda) samt `freq_base = 10000000.0f` ($10^7$) med `inv_freq[i] = 1.0f / powf(freq_base, (2.0f * i) / 64.0f)`.
3. **Verifiera i `compute_ssm_gates_f16`** att `ssm_a` behandlas som redan negativt (`-exp(A_log)`) så att `g = expf(ssm_a * softplus(alpha + dt_bias))`.