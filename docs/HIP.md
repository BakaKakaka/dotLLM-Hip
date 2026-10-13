# HIP-backend – krav, bygge, test och mätvärden

> Hör ihop med [HIP_PLAN.md](HIP_PLAN.md) (arbetspaket och beslut) och [HIP_PITFALLS.md](HIP_PITFALLS.md) (obligatorisk läsning).
> Detta dokument är leveransen för H0.1 (krav, versioner, bygg/test) samt resultatdokument för spikarna H0.2 och H0.3.

## 1. Krav och versioner

| Komponent | Krav | Verifierat på målmaskinen (2026-10-13) |
|---|---|---|
| OS | Windows 11 x64 | Windows 11 |
| HIP SDK | **7.2.x** (D4, låst till 7.2.3). `HIP_PATH` = SDK-roten med avslutande `\` (W14) | HIP 7.2.60201, clang 21, `C:\Program Files\AMD\ROCm\7.2\` |
| Drivrutin | Adrenalin med HIP 7-stöd | 32.0.22001.17002 |
| Visual Studio | Arbetsbelastningen *Desktop development with C++* (MSVC x64 + Windows SDK/ucrt). Krävs av hipcc även för `--genco` (W15) | VS Community 2026 18.10.3, MSVC 14.51.36231, Windows SDK 10.0.26100.0 |
| .NET SDK | 10.0 (`global.json`) | 10.0.401 |
| GPU | `gfx1100` (primär, RX 7900 XTX), `gfx1201` (sekundär, RX 9070 XT). iGPU och övriga arkitekturer ignoreras (D6) | 7900 XTX + 9070 XT, iGPU gfx1036 dold via `HIP_VISIBLE_DEVICES=0,2` |

Runtime-DLL:er laddas **endast** från `%HIP_PATH%\bin` med full sökväg: `amdhip64_7.dll` först, därefter `rocblas.dll` (+ `rocblas\library`). Drivrutinens `System32\amdhip64_6.dll` laddas aldrig och ska inte tas bort (W5, W6).

## 2. Miljökontroll – `scripts/check-hip-env.ps1` (H0.1)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\check-hip-env.ps1            # allt, inkl. körning på GPU
powershell -ExecutionPolicy Bypass -File scripts\check-hip-env.ps1 -SkipRun    # kompilera + ISA, ingen GPU-körning
powershell -ExecutionPolicy Bypass -File scripts\check-hip-env.ps1 -SkipSmoke  # bara miljö och enheter
```

Exit code 0 = PASS (varningar tillåtna), 1 = minst ett fel. Skriptet fungerar i Windows PowerShell 5.1 och PowerShell 7.

| Steg | Kontroll | Fallgrop |
|---|---|---|
| 1 | `HIP_PATH` satt, pekar inte på `bin\`, `include\hip\hip_runtime.h` och `bin\hipcc.exe` finns | W14 |
| 2 | `hipcc --version` = HIP 7.2.x | D4 |
| 3 | `amdhip64_7.dll`, `rocblas.dll`, `rocblas\library` med filer för gfx1100 och gfx1201 | W5, W7 |
| 4 | `HIP_VISIBLE_DEVICES` (process/user/machine) skrivs ut | W4 |
| 5 | MSVC x64 via `vswhere`, Windows SDK ucrt via registret | W15 |
| 6 | `hipInfo.exe`: enheter listas; gfx1100 krävs, gfx1201 förväntas, `isIntegrated=1` och okända arkitekturer flaggas som **IGNORED** | D6, D9, W11 |
| 7 | [`toolchain_smoke.hip`](../native/hip/kernels/toolchain_smoke.hip) kompileras med exakt samma flaggor som bygget för båda arkitekturerna. Disassembly verifieras: `v_dot4_i32_iu8`, `v_wmma_f32_16x16x16_f16`, `.wavefront_size: 32` | W15 |
| 8 | Smoke-kernlarna körs på varje kort via `hipModuleLoadData`/`hipModuleLaunchKernel` i en barnprocess (`dot4`, wave32-shuffle med 64-bitarsmask, WMMA f16→f32). Exakt en `amdhip64_*`-modul ska vara laddad, från `%HIP_PATH%\bin` | W5, W6 |

Verifierat 2026-10-13: PASS på båda korten. Utan `HIP_VISIBLE_DEVICES` syns iGPU:n som ordinal 1 och flaggas `IGNORED (integrated GPU)`, 9070 XT hamnar på ordinal 2 och körs ändå korrekt (val efter arkitektur, inte ordinal). `HIP_PATH=...\7.2\bin\` ger ett tydligt W14-fel.

## 3. Bygga (H0.4, D7)

```powershell
dotnet build dotLLM.slnx -c Release
```

- Projektet [`src/DotLLM.Hip`](../src/DotLLM.Hip/DotLLM.Hip.csproj) importerar [`native/hip/HipKernels.targets`](../native/hip/HipKernels.targets). Targeten `CompileHipKernels` (`BeforeTargets="BeforeBuild"`) kompilerar varje `native/hip/kernels/*.hip` per arkitektur i `HipArchs` (default `gfx1100;gfx1201`) till `obj\<cfg>\<tfm>\hsaco\<arch>\<namn>.hsaco`.
- Flaggor: `--genco -O3 -mno-wavefrontsize64 -std=c++17 -Xclang -fno-cuda-host-device-constexpr` (W15) plus `--offload-arch=<arch> -I "<kernels>"`.
- Resultatet bäddas in som `EmbeddedResource` med `LogicalName` `hsaco/<arch>/<namn>.hsaco`. Manifestet bäddas in som `hsaco/kernels.manifest.json` (schema 1: `hipccVersion`, `flags`, `archs`, `files[] { name, sha256 }` för alla `*.hip` och `*.h`, SHA-256 i versaler hex). Inget skrivs till källträdet; `native/hip/**/*.hsaco` är git-ignorerat.
- **Inkrementellt:** ombyggnad sker när någon `*.hip` eller `*.h` ändras, när källmängden/flaggorna/arkitekturerna ändras (stämpelfil `kernels.inputs.txt`, skrivs bara vid skillnad), när `HipKernels.targets` ändras eller när `hipcc.exe` byts (SDK-uppgradering). Annars hoppas targeten över. Verifierat: en touchad `.h` bygger om båda arkitekturerna.
- Endast `.hsaco` för den **aktuella** källmängden bäddas in, så borttagna kernels kan inte ligga kvar som gamla binärer.
- Targeten kontrollerar MSVC och Windows SDK (W15) och att hipcc rapporterar 7.2.x (D4) – bara när kernels faktiskt behöver byggas.
- **Sökvägar med mellanslag (W12):** alla sökvägar citeras. `-I` och katalogen **måste** vara separata argument – `hipcc.exe` vidarebefordrar ett sammanslaget `-I"dir med mellanslag"` ociterat till clang. Verifierat med kernelkatalog, obj-katalog och WorkDir med mellanslag.

### Utan HIP SDK och på målmaskinen

| Situation | Resultat |
|---|---|
| `HIP_PATH` saknas / pekar på `bin\` / ej Windows (t.ex. GitHub CI) | Varning `DLLMHIP001`, assemblyn byggs **utan** kernels. `HipDevice.IsAvailable()` ska då rapportera "kernels not built" (H1.2) och GPU-testerna skippas. |
| MSVC/Windows SDK saknas | Varning `DLLMHIP002` (fel om `DotLLMRequireHip=true`). |
| hipcc ≠ 7.2.x | Fel `DLLMHIP003`. |
| `DotLLMRequireHip=true` och toolchain saknas | Fel `DLLMHIP001`. |

På målmaskinen sätts `DotLLMRequireHip=true` i `Directory.Build.props.user` i repo-roten (git-ignorerad via `*.user`, importeras av `Directory.Build.props`):

```xml
<Project>
  <PropertyGroup>
    <DotLLMRequireHip>true</DotLLMRequireHip>
  </PropertyGroup>
</Project>
```

> [!NOTE]
> Miljöändringar (`HIP_PATH`, `HIP_VISIBLE_DEVICES`) syns bara i **nystartade** processer – starta om terminal, IDE och testkörare efter ändring (W14).

### Lägga till en kernel

1. Lägg `<namn>.hip` (och ev. headers) i `native/hip/kernels/` (D3). Kernels exporteras som `extern "C" __global__`.
2. `std::`-constexpr-funktioner (`std::min`, `std::max`, …) går inte att anropa i device-kod p.g.a. `-fno-cuda-host-device-constexpr` (W15). Använd hjälpare i `rdna_common.h` (H2.1) eller `fminf`/`fmaxf`/`min`/`max`.
3. `dotnet build` – kerneln finns sedan som `hsaco/<arch>/<namn>.hsaco` i `DotLLM.Hip.dll`.

## 4. Testa

```powershell
dotnet test tests/DotLLM.Tests.Unit -c Release                              # allt
dotnet test tests/DotLLM.Tests.Unit -c Release --filter "Category=GPU-AMD"  # bara HIP-GPU-tester (från H1.3)
```

Från H1.3 gäller: HIP-GPU-tester har `[Trait("Category","GPU-AMD")]`, ligger i `[Collection("HipGpu")]` med `DisableParallelization = true` (W13) och skippas med `Skip.IfNot(HipDevice.IsAvailable(...))`. Testerna väljer kort efter arkitektur och hårdkodar aldrig ordinaler (W4).

> [!WARNING]
> CI kör idag `--filter "Category!=GPU"`, vilket **inte** exkluderar `GPU-AMD`. **Beslut:** i H1.3-PR:en ändras filtret till `--filter "Category!=GPU&Category!=GPU-AMD"`, och testerna ska **dessutom** skippa sig själva via `IsAvailable`. Båda skydden krävs.

## 5. Spikar (`benchmarks/DotLLM.Hip.Spikes`)

```powershell
dotnet run -c Release --project benchmarks/DotLLM.Hip.Spikes -- all                       # H0.2 på gfx1100 + H0.3 på alla kort
dotnet run -c Release --project benchmarks/DotLLM.Hip.Spikes -- launch --device all       # H0.2 på båda korten
dotnet run -c Release --project benchmarks/DotLLM.Hip.Spikes -- bandwidth --out h03.md    # H0.3, Markdown till fil
```

Spike-projektet återanvänder `HipKernels.targets` med egen `HipKernelDir` (`Kernels/spike_kernels.hip`) och har en egen, medvetet minimal P/Invoke-yta (produktens interop är H1.1).

### 5.1 H0.2 – launch-overhead (2026-10-13)

Metod: tom kernel via `hipModuleLaunchKernel` på en non-blocking stream.
- **A** = CPU-tid för enqueue, median över 400 batchar à 32 launches (synk mellan batchar, ingen kö-backpressure).
- **B** = 20 000 launches i följd inkl. backpressure och slutsynk.
- **C** = GPU-tid per beroende kernel: en spin-kernel (30 ms, kalibrerad) blockerar kön medan kedjan på 256 kernels köas, så event-tiden mäter ren GPU back-to-back (enqueue tog ≤ 1 % av spin-tiden).
- **D** = samma kedja fångad med stream capture och spelad upp med `hipGraphLaunch`.

Typiska värden (två körningar, avvikelse < 10 %):

| Mätning | 7900 XTX, 1×32 | 7900 XTX, 384×256 | 9070 XT, 1×32 | 9070 XT, 256×256 |
|---|---:|---:|---:|---:|
| A. CPU per launch, median (µs) | 0,40 | 0,37 | 0,41 | 0,40 |
| B. Sustained per launch inkl. backpressure (µs) | 2,0 | 2,0 | 1,5 | 1,5 |
| C. GPU per beroende kernel (µs) | 1,5 | 1,6 | 1,2 | 1,3 |
| D. `hipGraphLaunch` CPU per nod (µs) | 0,36 | 0,36 | 0,36 | 0,35 |
| D. Graph GPU per nod (µs) | 1,5 | 1,6 | 1,3 | 1,3 |
| D. Graph sustained per nod (µs) | 1,8 | 1,9 | 1,4 | 1,4 |

Uppskattning per token med ~450 launches (W2), 7900 XTX: CPU ≈ **0,18 ms**, GPU-gap ≈ **0,7 ms**. Mot decode-målet ~137 tok/s (7,3 ms/token) är CPU-tiden ~2,5 % och GPU-gapen ~9 %.

**Beslut (underlag för H3.1):**
1. **HIP Graphs är inte P0.** CPU-enqueue under WDDM kostar bara ~0,4 µs/launch, och graph-replay sänker varken GPU-gapet per kernel (≈ 1,5 µs i båda fallen) eller CPU-tiden nämnvärt (0,36 vs 0,40 µs). Sustained förbättras bara ~10 %. H3.1:s CPU-krav (< 10 % av GPU-tiden) uppfylls redan med vanliga stream-launches.
2. **Fusion är P0.** Den dominerande kostnaden är GPU-sidans ~1,5 µs per beroende kernel; den minskar bara genom färre kernels (≤ 6 launches/lager i H3.1 ⇒ ~200 launches/token ⇒ ~0,3 ms).
3. Graphs omprövas om fusionen inte räcker eller om senare mätningar med riktiga kernels (barriärer/cacheflush mellan beroende kernels) visar större gap.

**RGP på Windows:** se 5.3.

### 5.2 H0.3 – läsbandbredd och MALL (2026-10-13)

Metod: grid-stride-läsning, 16 B per lane och 4 laddningar i luften per iteration, grid ≤ 16 block/CU × 256 trådar. Bufferten fylls med pseudoslumpdata (identiska resultat som konstant fyllnad ⇒ ingen kompressionseffekt). Arbetsmängder < 256 MB läses i flera pass per launch (≥ 256 MB per launch) så att cache-bandbredden inte döljs av launch-overhead. Bästa av 3 samplingar à ≥ 4 GB. *Normal* = vanlig load, *NT* = `__builtin_nontemporal_load`.

| Arbetsmängd | 7900 XTX normal | 7900 XTX NT | 9070 XT normal | 9070 XT NT |
|---:|---:|---:|---:|---:|
| 1 MB | 5 150 | 5 430 | 3 740 | 4 900 |
| 2 MB | 6 550 | 6 810 | 4 970 | 4 650 |
| 4 MB | 7 160 | 7 290 | 4 520 | 3 920 |
| 8 MB | 8 120 | 7 680 | 7 780 | 5 220 |
| 16 MB | 7 070 | 2 190 | 7 780 | 2 030 |
| 32 MB | 3 900 | 1 140 | 6 850 | 830 |
| 64 MB | 3 370 | 950 | 2 000 | 690 |
| 128 MB | 1 440 | 890 | 990 | 650 |
| 256 MB | 850 | 860 | 630 | 640 |
| 512 MB | 840 | 850 | 630 | 640 |
| 1 GB | 840 | 850 | 630 | 630 |
| 2 GB | 830 | **845** | 630 | **633** |

(GB/s, 1 GB/s = 10⁹ B/s.)

**Referensvärden för alla prestandagater (H0.3-acceptans):**

| Kort | Uppmätt DRAM-topp | Teoretiskt | Roofline Llama-3.1-8B Q4_K_M (≈ 4,6 GB lästa/token, exkl. `token_embd`) |
|---|---:|---:|---:|
| RX 7900 XTX | **845 GB/s** (88 %) | 960 GB/s | ≈ 183 tok/s |
| RX 9070 XT | **633 GB/s** (99 %) | 640 GB/s | ≈ 138 tok/s |

Härledd gate H3.2 (≥ 80 % av H0.3-bandbredden): **≥ 676 GB/s** på 7900 XTX.

> [!IMPORTANT]
> **Beslutat 2026-10-13** ([HIP_PLAN.md, M3](HIP_PLAN.md)): Rooflinen räknas på uppmätt referensbandbredd och **faktiskt lästa bytes per token**, exklusive `token_embd` (≈ 4,6 GB för Llama-3.1-8B Q4_K_M ⇒ ≈ 183 tok/s). Gaten är ≥ 70 % (≈ 128 tok/s) och ≥ llama.cpp-HIP. Sträckmålet är 80 % (≈ 146 tok/s). Referensbandbredden höjs om någon kernel mäter högre än 845 GB/s.

**Slutsatser:**
1. **MALL syns tydligt.** 7900 XTX: L2-nivå (~5–8 TB/s) upp till 8–16 MB, Infinity Cache (~3,4–3,9 TB/s) för 32–64 MB, fallande vid 128 MB (96 MB IC) och DRAM från 256 MB. 9070 XT: ~4–8 TB/s upp till 32 MB, sedan 2 TB/s vid 64 MB (= IC-storleken) och DRAM från 128–256 MB.
2. **Non-temporal-laddningar allokerar inte i MALL.** För 16–128 MB faller NT till nära DRAM-nivå (28–35 % av normal). NT är alltså en ren "streaming"-hint som lämnar Infinity Cache orörd.
3. **NT ger ingen meningsfull vinst för en ren ström:** +1–2 % på 7900 XTX, ±0 på 9070 XT vid ≥ 256 MB.
4. **Följd för H3.5:** hypotesen "vikter via NT ⇒ KV-cache stannar i MALL" är förenlig med (2) men oprövad. Den måste mätas i H3.5 med blandad trafik (NT-vikter + återläst KV); en ren viktström vinner inget på NT. Behåll bara om mätbar vinst.

### 5.3 RGP/RGA på Windows

Radeon Developer Tool Suite 2026-10-02 (RGP 2.7.1, RDP 3.5.1, RGA 2.15.0) finns på målmaskinen som portabel mapp. Placera den enligt **W16**: utanför repot, utpekad av `DOTLLM_RDTS`, aldrig i `PATH`.

- **RGA (verifierat 2026-10-13):** binary mode fungerar på våra code objects för `gfx1100` och `gfx1201` efter uppackning av offload-bundlen:
  ```powershell
  & "$env:HIP_PATH\bin\clang-offload-bundler.exe" --type=o --unbundle --targets=hipv4-amdgcn-amd-amdhsa--gfx1100 --input=<kernel>.hsaco --output=<kernel>.elf
  & "$env:DOTLLM_RDTS\rga.exe" -s bin -a stats.csv --isa isa.txt --livereg livereg.txt <kernel>.elf
  ```
  Exempel, `toolchain_smoke` `smoke` på gfx1100: 19 VGPR, 18 SGPR, 0 spills, 0 LDS, 336 B ISA. RGA kan inte läsa bundlen direkt ("not a valid object file").
- **RGP:** README:n anger HIP och OpenCL på Windows 11 för RX 7000/9000. Själva fångsten via Radeon Developer Panel är **ännu inte verifierad** och följs upp i H2.8. Fram till dess gäller `hipEvent`-tidtagning och RGA-rapporten.

## 6. Baseline-mätningar – M2 (H2.7)

Verifierat 2026-10-13 på målmaskinen:
- **GPU:** AMD Radeon RX 7900 XTX (`gfx1100`, ordinal 0)
- **OS / Drivrutin:** Windows 11 x64, Adrenalin 32.0.22001.17002, HIP SDK 7.2.3 (7.2.60201)
- **llama.cpp:** build **b11583** (officiella Windows-binärer):
  - HIP: `llama-b11583-bin-win-rocm-10.0-x64\llama-completion.exe`, enhet `-dev ROCm0`
  - Vulkan: `llama-b11583-bin-win-vulkan-x64\llama-completion.exe`, enhet `-dev Vulkan2`
- **dotLLM:** `DotLLM.Hip` på `--device hip:gfx1100` via `scripts/bench_compare.py`
- **Mätmetod:** 5 körningar (1 uppvärmningskörning bortkastad), greedy generering (`--temp 0`), max 20 tokens. Best-of-N (max tok/s, min ms) per tabell.

### 6.1 Llama-3.2-1B-Instruct Q8_0

| Prompt | Motor | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | ms/tok | Totalt tok/s | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt-medium.txt` (~229 tok) | **dotLLM (hip:gfx1100)** | 100,7 | 2 274,8 | 104,5 | 181,8 | 5,22 | 1 213,5 | 1,5 % |
| `prompt-medium.txt` (~229 tok) | **llama.cpp (ROCm0)** | 31,9 | 7 150,2 | 61,2 | 310,2 | 3,22 | 2 652,2 | 1,3 % |
| `prompt-medium.txt` (~229 tok) | **llama.cpp (Vulkan2)** | 21,8 | 10 474,6 | 42,8 | 443,6 | 2,25 | 3 823,5 | 0,8 % |
| `prompt-large.txt` (~612 tok) | **dotLLM (hip:gfx1100)** | 184,5 | 3 311,1 | 105,6 | 180,0 | 5,28 | 2 175,1 | 0,8 % |
| `prompt-large.txt` (~612 tok) | **llama.cpp (ROCm0)** | 50,3 | 12 172,6 | 59,5 | 319,1 | 3,13 | 5 745,2 | 0,4 % |
| `prompt-large.txt` (~612 tok) | **llama.cpp (Vulkan2)** | 42,5 | 14 381,4 | 43,0 | 442,0 | 2,26 | 7 377,5 | 1,2 % |

### 6.2 Llama-3.2-1B-Instruct Q4_K_M

| Prompt | Motor | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | ms/tok | Totalt tok/s | CV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `prompt-medium.txt` (~228 tok) | **dotLLM (hip:gfx1100)** | 107,2 | 2 135,6 | 51,0 | 372,6 | 2,55 | 1 574,0 | 2,7 % |
| `prompt-medium.txt` (~228 tok) | **llama.cpp (ROCm0)** | 112,1 | 2 034,2 | 50,9 | 373,0 | 2,68 | 1 515,1 | 1,1 % |
| `prompt-medium.txt` (~228 tok) | **llama.cpp (Vulkan2)** | 25,8 | 8 834,5 | 33,7 | 563,9 | 1,77 | 4 150,6 | 0,9 % |
| `prompt-large.txt` (~612 tok) | **dotLLM (hip:gfx1100)** | 185,9 | 3 286,2 | 50,9 | 373,2 | 2,54 | 2 664,7 | 0,8 % |
| `prompt-large.txt` (~612 tok) | **llama.cpp (ROCm0)** | 137,1 | 4 465,5 | 48,8 | 389,2 | 2,57 | 3 394,8 | 1,1 % |
| `prompt-large.txt` (~612 tok) | **llama.cpp (Vulkan2)** | 50,9 | 12 027,6 | 34,0 | 559,6 | 1,79 | 7 438,4 | 1,5 % |

### 6.3 Analys av baseline
1. **Decode (Q4_K_M):** dotLLM når **372,6–373,2 tok/s**, vilket är **100 % i paritet med llama.cpp HIP** (**373,0–389,2 tok/s**) redan före Fas 3-optimeringarna (mmvq, graph, split-KV).
2. **Decode (Q8_0):** dotLLM når **180,0–181,8 tok/s** mot llama.cpp HIP:s **310,2–319,1 tok/s**. Här kommer Fas 3-fusion och mmvq (H3.1–H3.2) att stänga gapet.
3. **Prefill:** dotLLM slår llama.cpp HIP på Q4_K_M medium (2 135,6 vs 2 034,2 tok/s). Vid längre sekvenser drar llama.cpp ifrån tack vare WMMA-prefill, vilket är det uttryckliga målet för Fas 4 (H4.1–H4.4).

## 7. ISA-resursrapport och profileringsverktyg (H2.8)

Skriptet [`scripts/hip-isa-report.ps1`](../scripts/hip-isa-report.ps1) packar upp alla inbäddade `.hsaco`-filer för båda arkitekturerna (`gfx1100` och `gfx1201`) via `clang-offload-bundler --unbundle` och kör Radeon GPU Analyzer (`rga.exe`) i binärt läge (`-s bin`).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\hip-isa-report.ps1               # Texttabell
powershell -ExecutionPolicy Bypass -File scripts\hip-isa-report.ps1 -AsMarkdown   # Markdown-tabell
powershell -ExecutionPolicy Bypass -File scripts\hip-isa-report.ps1 -Kernel gemv  # Filtrera på kernelnamn
```

### ISA-status för baseline (M2)
- **Registerspills:** **0 spills** (både VGPR och SGPR) i samtliga kernels på både `gfx1100` och `gfx1201`.
- **VGPR-allokering:** Alla kernels ligger långt under hårdvarugränsen på 256 VGPR (toppnotering: 56 VGPR i `quantized_gemv_q5_k`).
- Skriptet används som obligatorisk regressionskontroll i Fas 3/4-PR:er för att verifiera registertryck före och efter optimeringar.

