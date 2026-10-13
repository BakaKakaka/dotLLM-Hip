# HIP-backend – fallgropar och faktakorrigeringar

> Obligatorisk läsning före arbete på `DotLLM.Hip`. Hör ihop med [HIP_PLAN.md](HIP_PLAN.md).
> Dokumentet **ersätter** de tekniska detaljerna i [PLAN.md](../PLAN.md) där de skiljer sig åt.

## 0. Verifierat på målmaskinen (2026-10-13)

| Egenskap | Värde |
|---|---|
| HIP SDK | **7.2.3** (HIP 7.2.60201, clang 21). `HIP_PATH` **måste** vara SDK-roten `C:\Program Files\AMD\ROCm\7.2\` (inte `...\bin\`), se W14. |
| Adrenalin-drivrutin | 32.0.22001.17002 (dGPU) |
| Visual Studio | Community 2026 (18.10), MSVC 14.51/14.52, Windows SDK 10.0.26100. **Krav:** arbetsbelastningen "Desktop development with C++", se W15. |
| Toolchain-smoke (2026-10-13) | `hipcc --genco` → `hipModuleLoadData` → `hipModuleLaunchKernel` via P/Invoke: **PASS på båda korten**. ISA verifierad: `v_dot4_i32_iu8`, `v_wmma_f32_16x16x16_f16` (gfx1100: A/B = 8 VGPR replikerat, gfx1201: 4 VGPR), `.wavefront_size: 32`. Exakt en `amdhip64_7.dll` laddad. |
| `HIP_VISIBLE_DEVICES` | `0,2` (Machine + User) döljer iGPU:n. Ordinalerna nedan gäller **med** variabeln. |
| HIP-ordinal 0 | **RX 7900 XTX**, `gfx1100`, 23,98 GB, 48 WGP (= 96 CU), L2 6 MB |
| HIP-ordinal 1 | **RX 9070 XT**, `gfx1201`, 15,92 GB, 32 WGP (= 64 CU), L2 8 MB |
| Dold | iGPU `gfx1036` (Ryzen, `isIntegrated=1`). Utan variabeln hamnar den på ordinal 1 och 9070 XT på ordinal 2. |
| Runtime-DLL | `amdhip64_7.dll` finns **endast** i `%HIP_PATH%\bin`. `amdhip64_6.dll` i System32 (+ DriverStore `amdocl.inf_*`) tillhör Adrenalin-drivrutinen. **Ska inte tas bort.** Vi laddar den aldrig. |
| Bibliotek i SDK | `rocblas.dll` + `rocblas\library` (96 filer gfx1100 / 56 filer gfx1201), `hipblaslt.dll`, `hiprtc0702.dll`, rocWMMA-headers |
| `warpSize` | 32 på båda korten |
| P2P | Inga peers (irrelevant, en GPU per modell) |

---

## 1. Faktakorrigeringar av PLAN.md

| # | PLAN.md | Korrekt | Åtgärd i planen |
|---|---|---|---|
| 1 | "9070 gt (gfx1200)" | RX **9070 XT = gfx1201**. gfx1200 = RX 9060 XT. Code objects går inte att byta mellan dem. | Bygg `gfx1100` + `gfx1201` (D7). |
| 2 | `-mwavefrontsize64=false` | Flaggan finns inte. Rätt flagga är `-mno-wavefrontsize64`. | Låst i byggflaggorna (H0.4). |
| 3 | `__attribute__((amdgpu_wave_size(32)))` | Inget dokumenterat Clang-attribut. Wave-storleken styrs per target. | Används inte. En probe-kernel verifierar `warpSize==32` (H2.1). |
| 4 | "Kompilatorn väljer ofta Wave64" | Wave32 är redan default på gfx11/12. | Flaggan sätts explicit som skydd. |
| 5 | "Wave32 halverar registertrycket" | Antalet VGPR/lane är detsamma. Vinsten är att en wave32 matchar GGUF-blockets 32 värden. | – |
| 6 | "1 tråd avkodar 1 värde" | Anti-mönster. Byte-laddningar når inte bandbreddstaket. | Varje lane laddar 16 B och kör `sudot4`/`sudot8` (H3.2). |
| 7 | `__builtin_amdgcn_wmma_f16_16x16x16_f16` | Namnen har suffix `_w32`. gfx12 har egna `_gfx12`-varianter med **annan operandlayout**. | rocWMMA abstraherar detta (D8), och layouttester är obligatoriska. |
| 8 | FP16-ackumulator | Tappar precision över K = 4096–14336. | Alltid `f32`-ackumulator. |
| 9 | `16x16x32_iu4` "RDNA3/4" | Finns bara på gfx12. RDNA3 har `16x16x16_iu4`. | Ingår i gfx1201-epicen (fas 5). |
| 10 | "Dedikerade matrisacceleratorer" | RDNA3-WMMA körs i VALU-vägen. RDNA4 har ~2× WMMA/CU, FP8 och sparsity. | Prestandamålen sätts per kort. |
| 11 | `v_dot4_i32_i8` | På gfx11/12 heter den `v_dot4_i32_iu8` (`__builtin_amdgcn_sudot4`). Dessutom finns `sudot8` (iu4) och `fdot2` (f16). | Kräver Q8_1-kvantisering av aktiveringarna (H3.2). |
| 12 | "GGUF med FP8" | GGUF saknar FP8-typ. FP8 finns i safetensors. RDNA3 saknar FP8-WMMA. | Valfri epic, bara för gfx1201. |
| 13 | "9070 XT har också stor L3" | 64 MB IC, 16 GB, ~640 GB/s. 7900 XTX: 96 MB IC, 24 GB, ~960 GB/s. | Mål per kort. Hybrid-offload för 9070 XT (fas 6). |
| 14 | "dotLLM kör anrop via `IBackend`" | `IBackend` gör bara alloc/copy. Compute ligger i `CudaTransformerModel` + `CudaKernels`. | Egen `HipTransformerModel : IModel`. |
| 15 | "Vi behöver inte PagedAttention" | PagedAttention handlar om fragmentering och prefix-delning, inte om cache. | Sammanhängande KV-cache (en användare). Rätt slutsats, men av andra skäl. |

---

## 2. Infinity Cache, kvantitativt (Llama-3.1-8B Q4_K_M, decode)

| Trafik per token | Storlek | I MALL? |
|---|---|---|
| Vikter | ~4,9 GB | **Nej.** Strömmas igenom och trashar cachen. |
| KV (FP16) | 128 KB per kontext-token (512 MB @ 4K) | Nej. Trycks ut av vikt-strömmen. |
| Aktiveringar | 8–28 KB | Ja, men ligger redan i L2. |

**Roofline (teoretisk, från spec):** 7900 XTX ≈ 960 / 4,9 ≈ 195 tok/s, 9070 XT ≈ 130 tok/s.
**Roofline (gäller för gater):** uppmätt referensbandbredd / faktiskt lästa bytes per token (≈ 4,6 GB, exklusive `token_embd`), vilket ger 7900 XTX ≈ **183 tok/s** och 9070 XT ≈ **137 tok/s**. Definitionen finns i [HIP_PLAN.md, M3](HIP_PLAN.md).

**Slutsats:** MALL gör verklig nytta vid **chunked prefill** och möjligen för KV-cachen om vikterna laddas non-temporal.

**Uppmätt i H0.3 (se [HIP.md §5.2](HIP.md)):**
- Infinity Cache levererar ~3,4–3,9 TB/s upp till ~64 MB på 7900 XTX och ~2 TB/s vid 64 MB på 9070 XT.
- Non-temporal-laddningar **allokerar inte i MALL**.
- DRAM-referensen är 845 GB/s (XTX) respektive 633 GB/s (9070 XT).
- Använd **aldrig** NT för data som läses igen, såsom KV, aktiveringar och prefill-tiles.

---

## 3. Windows- och miljöfallgropar

| # | Fallgrop | Åtgärd (paket) |
|---|---|---|
| W1 | **TDR (~2 s)** återställer GPU:n vid långa kernels, och skärmen påverkas om den sitter på samma kort. | Chunked prefill och kernel-budget < 100 ms (H4.4). Under utveckling: kör compute på kortet **utan** skärm om möjligt. Förläng aldrig `TdrDelay` som produktlösning. |
| W2 | **Launch-overhead under WDDM:** ~14 launches/lager ≈ 450/token. **Uppmätt (H0.2):** CPU ≈ 0,4 µs/launch (inte flaskhals), GPU-gap ≈ 1,5 µs per *beroende* kernel ≈ 0,7 ms/token (~9 % av budgeten). HIP Graphs sänker inte GPU-gapet. | **Fusion (H3.1, P0).** HIP Graphs är P1 och provås bara om fusionen inte räcker. Mät om gapet med riktiga kernels, eftersom cache-flush och barriärer kan öka det. |
| W3 | **Tyst VRAM-spill** till delat minne, ingen OOM. | Budgetkontroll med `hipMemGetInfo` och avbrott över 95 % (H2.5). |
| W4 | **Ordinalerna beror på miljön.** iGPU:n hamnar mellan dGPU:erna om `HIP_VISIBLE_DEVICES` saknas. Med `0,2` byter 9070 XT ordinal från 2 till 1. | Välj kort efter arkitektur, `hip:gfx1100` (H1.2). Tester får **aldrig** hårdkoda ordinaler. Filtret på `isIntegrated` gäller oavsett variabeln. |
| W5 | **DLL-versioner:** drivrutinen lägger `amdhip64_6.dll` i System32. v7 finns bara i SDK:n. | Ladda `amdhip64_7.dll` med **full sökväg** från `HIP_PATH`. Avbryt om major ≠ 7 (H1.1). Rör aldrig drivrutinens v6. |
| W6 | **Två HIP-runtimes i samma process:** om `rocblas.dll` laddas före vår HIP-DLL kan en annan kopia av `amdhip64_7.dll` hamna i processen, och då blir device-pekarna ogiltiga. | Ladda HIP-runtime **först**, därefter rocBLAS. Ett starttest verifierar att exakt en `amdhip64_*`-modul är laddad (H1.1). |
| W7 | **rocBLAS kräver `rocblas\library`**, och täckningen för gfx1201 är tunnare. | Tydligt fel om katalogen saknas. Egen GEMM (H4.2). |
| W8 | **Profilering:** rocprof/omniperf finns bara för Linux. | `hipEvent`-tidtagning inbyggd i benchmark-harness, RGA för ISA och RGP om det stöds (H0.2). |
| W9 | **`hipDeviceProp_t` är versionerad** (R0600). | Använd `hipDeviceGetAttribute`. Hela structen bara med `sizeof`-test (H1.1). |
| W10 | **Current device är trådlokal.** Server byter tråd via async/await. | `hipSetDevice` vid varje `Forward`-ingång (H2.5). |
| W11 | **`multiProcessorCount` returnerar WGP**, inte CU. | Grid-heuristiker räknar CU = 2 × WGP (H1.2). |
| W12 | **Mellanslag i `HIP_PATH`.** Dessutom vidarebefordrar `hipcc.exe` ett sammanslaget `-I"dir med mellanslag"` **ociterat** till clang (verifierat 2026-10-13), medan `-I "dir"` som separata argument fungerar. | Citera alla sökvägar i MSBuild `Exec` och PowerShell, och skicka alltid `-I` och katalogen som separata argument (H0.4, `check-hip-env.ps1`). |
| W13 | **xUnit kör testklasser parallellt.** Flera HIP-kontexter på samma GPU ger VRAM-konkurrens och flaky tester. | `[Collection("HipGpu")]` med `DisableParallelization = true` (H1.3). |
| W14 | **Fel `HIP_PATH`.** Pekar den på `...\7.2\bin\` hittar hipcc inte `include\` och MSBuild letar efter `bin\bin\hipcc.exe`. Miljöändringar syns dessutom bara i **nystartade** processer (IDE, terminal, testkörare). | `check-hip-env.ps1` och `HipLibraryResolver` validerar att `%HIP_PATH%\include\hip\hip_runtime.h` och `%HIP_PATH%\bin\amdhip64_7.dll` finns, och ger annars ett tydligt fel (H0.1, H1.1). |
| W15 | **hipcc kräver MSVC:s STL + Windows SDK**, även vid `--genco`. Med VS 2026 (MSVC 14.51/14.52) definierar STL:en `isgreater`/`isless`/… som `constexpr` under Clang. HIP gör constexpr-funktioner implicit `__host__ __device__`, och de krockar därför med Clangs egna `__device__`-deklarationer (`cannot overload __host__ __device__ function`). MSVC 14.52 kräver dessutom formellt Clang ≥ 22, medan HIP SDK 7.2 har Clang 21. | **Verifierad lösning:** `-Xclang -fno-cuda-host-device-constexpr` i alla kernel-byggen (H0.4). Konsekvens: `std::`-constexpr-funktioner (t.ex. `std::min`/`std::max`) går inte att anropa i device-kod. Använd egna `__device__`-hjälpare i `rdna_common.h` eller HIP:s `fminf`/`fmaxf`/`min`/`max`. `check-hip-env.ps1` provkompilerar en smoke-kernel och fångar regressioner vid VS-uppdateringar. Nödlösning vid STL1000-fel: `-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH`. |

---

## 4. Fallgropar vid portering CUDA → HIP

| Område | Problem | Åtgärd |
|---|---|---|
| Warp-intrinsics | ROCm 7: `__shfl_*_sync` kräver 64-bitars mask. | Hjälpare i `rdna_common.h`. Inga råa `0xFFFFFFFF`. |
| `warpSize` | Är inte `constexpr` i ROCm 7. | `constexpr int WAVE = 32`. |
| `__launch_bounds__(N, M)` | M = min waves/EU i HIP, min blocks/SM i CUDA. | Ange bara M med kommentar. |
| Ingen PTX-motsvarighet | Code objects måste byggas per arkitektur. | D7: AOT per arkitektur, inbäddade. |
| `--genco`-format | Utdata är ett **clang offload bundle** (`__CLANG_OFFLOAD_BUNDLE__`), inte rå ELF. `hipModuleLoadData` accepterar det (verifierat). | Ladda bundlen direkt. För ISA-inspektion: `clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--<arch>` och sedan `llvm-objdump -d --mcpu=<arch>`. |
| `__shfl_xor` | Kompileras till `ds_bpermute_b32` (LDS-crossbar), inte DPP. Det är långsammare i heta reduktioner. | Baseline använder `__shfl_xor`. I H3 byts wave-reduktionerna mot DPP (`__builtin_amdgcn_update_dpp`/`mov_dpp`) och `__builtin_amdgcn_permlanex16`, verifierat i disassembly. |
| `--use_fast_math` | Denormal-flush påverkar små FP16-värden. | Precis matematik i reduktioner, softmax och norm. |
| Ojusterade block | Q8_0 34 B, Q5_0 22 B, Q6_K 210 B. | Repack vid laddning (H3.3). |
| Header-beroenden | CUDA-targetens Inputs omfattar bara `.cu`, så en ändrad header bygger inte om. | Inputs inkluderar `*.h` (H0.4). |

---

## 5. Kopiera INTE dessa mönster från `DotLLM.Cuda`

1. **FP16 residual stream**, som ger känd Qwen-divergens ([ROADMAP.md](ROADMAP.md)). Använd FP32-residual (D5).
2. **Q4_0 laddas upp tre gånger:** `UploadAndDequant` läcker sin råa kopia och `UploadQuantized` laddar upp en till. Exakt en kvantkopia per vikt.
3. **Dekvantisering av varje projektion till FP16-scratch per prefill.** Ersätt med fused dequant-GEMM (H4.2).
4. **GEMV med 1 tråd per superblock:** för Q4_K vid k=4096 är bara 16 av 256 trådar aktiva. En wave per rad (H2.3/H3.2).
5. **Attention läser om K/V per query-head** och saknar split-KV. Använd flash-decoding (H3.4).
6. **Synkrona pageable-kopior per token.** Använd pinned + async (H2.5).

---

## 6. Strategiska risker

- **Två arkitekturer.** 7900 XTX är primärt och optimeras först. gfx1201 måste kompilera och klara korrekthetstesterna vid varje milstolpe, så att kodvägarna inte glider isär. gfx1201-tuning sker i fas 5.
- **Referens:** llama.cpp (HIP + Vulkan) på samma maskin är mätstickan.
- **Licens:** llama.cpp och rocWMMA är MIT, alltså förenliga med GPLv3. Attribution krävs i filhuvudet.
