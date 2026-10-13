#!/usr/bin/env python3
"""Run H2.7b benchmarks across dotLLM, llama.cpp-HIP, and llama.cpp-Vulkan on RX 7900 XTX.

Per HIP_PLAN.md H2.7b:
- 128 generated tokens (--tokens 128)
- 5 runs after warmup, reporting median (--runs 5 --median)
- CV reported
- Evaluates:
    - Llama-3.2-1B Q8_0 (medium and large prompts)
    - Llama-3.2-1B Q4_K_M (medium and large prompts)
    - Llama-3.1-8B Q4_K_M (medium prompt, M3 model)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "1B_Q8_0": r"C:\Users\benny\.dotllm\models\bartowski\Llama-3.2-1B-Instruct-GGUF\Llama-3.2-1B-Instruct-Q8_0.gguf",
    "1B_Q4_K_M": r"C:\Users\benny\.dotllm\models\bartowski\Llama-3.2-1B-Instruct-GGUF\Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    "8B_Q4_K_M": r"C:\Users\benny\.dotllm\models\bartowski\Meta-Llama-3.1-8B-Instruct-GGUF\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
}

LLAMACPP_HIP = os.environ.get("DOTLLM_LLAMACPP_HIP", r"E:\Tools\llama.cpp\hip\llama-b11583-bin-win-rocm-10.0-x64")
LLAMACPP_VULKAN = os.environ.get("DOTLLM_LLAMACPP_VULKAN", r"E:\Tools\llama.cpp\hip\llama-b11583-bin-win-vulkan-x64")

HIP_BIN = os.path.join(LLAMACPP_HIP, "llama-completion.exe")
VULKAN_BIN = os.path.join(LLAMACPP_VULKAN, "llama-completion.exe")

TOKENS = 128
RUNS = 5


def run_benchmark(model_path: str, prompt_size: str, engine_type: str, temp_json: str):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "bench_compare.py"),
        "--model", model_path,
        "--prompt-size", prompt_size,
        "--tokens", str(TOKENS),
        "--runs", str(RUNS),
        "--median",
        "--export-json", temp_json,
    ]

    if engine_type == "dotllm":
        cmd.extend(["--dotllm", "--cli", "--device", "hip:gfx1100"])
    elif engine_type == "llamacpp_hip":
        cmd.extend([
            "--llamacpp",
            "--llamacpp-bin", HIP_BIN,
            "--llamacpp-device", "ROCm0",
            "--device", "hip",
        ])
    elif engine_type == "llamacpp_vulkan":
        cmd.extend([
            "--llamacpp",
            "--llamacpp-bin", VULKAN_BIN,
            "--llamacpp-device", "Vulkan2",
            "--device", "gpu",
        ])
    else:
        raise ValueError(f"Unknown engine: {engine_type}")

    print(f"\n=======================================================")
    print(f"Running {engine_type} on {Path(model_path).name} ({prompt_size})")
    print(f"Command: {' '.join(cmd)}")
    print(f"=======================================================\n")

    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"ERROR: Benchmark failed with exit code {res.returncode}", file=sys.stderr)
        return None

    if not os.path.exists(temp_json):
        print(f"ERROR: Result json not found: {temp_json}", file=sys.stderr)
        return None

    with open(temp_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"][0]


def format_swe(val: float, decimals: int = 1) -> str:
    """Format float with Swedish space for thousands and comma for decimal."""
    s = f"{val:,.{decimals}f}"
    # s is like "1,234.5" -> replace comma with space and dot with comma
    parts = s.split(".")
    int_part = parts[0].replace(",", " ")
    if len(parts) > 1:
        return f"{int_part},{parts[1]}"
    return int_part


def main():
    all_results = {}
    temp_json = str(REPO_ROOT / "benchmarks" / "temp_h27b_step.json")

    matrix = [
        # 1B Q8_0
        ("1B_Q8_0", "medium", ["dotllm", "llamacpp_hip", "llamacpp_vulkan"]),
        ("1B_Q8_0", "large", ["dotllm", "llamacpp_hip", "llamacpp_vulkan"]),
        # 1B Q4_K_M
        ("1B_Q4_K_M", "medium", ["dotllm", "llamacpp_hip", "llamacpp_vulkan"]),
        ("1B_Q4_K_M", "large", ["dotllm", "llamacpp_hip", "llamacpp_vulkan"]),
        # 8B Q4_K_M
        ("8B_Q4_K_M", "medium", ["dotllm", "llamacpp_hip", "llamacpp_vulkan"]),
    ]

    for model_key, prompt_size, engines in matrix:
        model_path = MODELS[model_key]
        for eng in engines:
            key = (model_key, prompt_size, eng)
            r = run_benchmark(model_path, prompt_size, eng, temp_json)
            if r is not None:
                all_results[key] = r
            if os.path.exists(temp_json):
                os.remove(temp_json)

    # Save aggregated raw results
    agg_path = REPO_ROOT / "benchmarks" / "h27b_results.json"
    serializable = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in all_results.items()}
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAll results saved to {agg_path}")

    # Generate Markdown Tables
    print("\n\n=================== MARKDOWN TABLES ===================\n")

    def print_1b_table(model_key: str, title: str):
        print(f"### {title}\n")
        print("| Prompt | Motor | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | ms/tok | Totalt tok/s | CV |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for prompt_size in ("medium", "large"):
            for eng, name in [
                ("dotllm", "**dotLLM (hip:gfx1100)**"),
                ("llamacpp_hip", "**llama.cpp (ROCm0)**"),
                ("llamacpp_vulkan", "**llama.cpp (Vulkan2)**"),
            ]:
                r = all_results.get((model_key, prompt_size, eng))
                if not r:
                    continue
                p_label = f"`prompt-{prompt_size}.txt` (~{r['prefill_tokens']} tok)"
                pf_ms = format_swe(r["prefill_ms"])
                pf_ts = format_swe(r["prefill_tok_per_sec"])
                dc_ms = format_swe(r["decode_ms"])
                dc_ts = format_swe(r["decode_tok_per_sec"])
                ms_tok = format_swe(r["decode_ms_per_tok"], 2)
                tot_ts = format_swe(r["total_tok_per_sec"])
                cv = f"{r['decode_cv'] * 100:.1f} %".replace(".", ",")
                print(f"| {p_label} | {name} | {pf_ms} | {pf_ts} | {dc_ms} | {dc_ts} | {ms_tok} | {tot_ts} | {cv} |")
        print()

    print_1b_table("1B_Q8_0", "6.1 Llama-3.2-1B-Instruct Q8_0")
    print_1b_table("1B_Q4_K_M", "6.2 Llama-3.2-1B-Instruct Q4_K_M")

    # 8B Table with Roofline
    print("### 6.3 Llama-3.1-8B-Instruct Q4_K_M (M3 Baseline)\n")
    print("| Motor | Prefill ms | Prefill tok/s | Decode ms | Decode tok/s | ms/tok | Effektiv GB/s | Andel roofline | CV |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    # Roofline ref: 845 GB/s, 4.60 GB read per token -> 183.7 tok/s
    BYTES_PER_TOKEN_GB = 4.60
    ROOFLINE_TOK_S = 845.0 / BYTES_PER_TOKEN_GB  # 183.695

    for eng, name in [
        ("dotllm", "**dotLLM (hip:gfx1100)**"),
        ("llamacpp_hip", "**llama.cpp (ROCm0)**"),
        ("llamacpp_vulkan", "**llama.cpp (Vulkan2)**"),
    ]:
        r = all_results.get(("8B_Q4_K_M", "medium", eng))
        if not r:
            continue
        pf_ms = format_swe(r["prefill_ms"])
        pf_ts = format_swe(r["prefill_tok_per_sec"])
        dc_ms = format_swe(r["decode_ms"])
        dc_ts_val = r["decode_tok_per_sec"]
        dc_ts = format_swe(dc_ts_val)
        ms_tok = format_swe(r["decode_ms_per_tok"], 2)
        eff_bw = dc_ts_val * BYTES_PER_TOKEN_GB
        eff_bw_str = format_swe(eff_bw)
        roofline_pct = (dc_ts_val / ROOFLINE_TOK_S) * 100.0
        roofline_str = f"{roofline_pct:.1f} %".replace(".", ",")
        cv = f"{r['decode_cv'] * 100:.1f} %".replace(".", ",")
        print(f"| {name} | {pf_ms} | {pf_ts} | {dc_ms} | {dc_ts} | {ms_tok} | {eff_bw_str} | {roofline_str} | {cv} |")
    print()


if __name__ == "__main__":
    main()
