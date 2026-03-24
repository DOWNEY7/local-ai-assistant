"""
Model Comparison Study
======================
Benchmarks multiple local Ollama models across several test prompts,
measuring Time to First Token (TTFT) and Tokens Per Second (TPS),
then prints a formatted Markdown comparison table.
"""

import sys
import os
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import ollama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = ["llama3.2", "phi3"]

TEST_PROMPTS = [
    # Coding question
    "Write a Python function that checks whether a given string is a valid IPv4 address. Include error handling.",
    # Logic puzzle
    "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. Every label is wrong. You pick one fruit from the 'Mixed' box and it's an apple. What are the correct labels for each box? Explain step by step.",
    # Summarization task
    "Summarize in 3 bullet points the key advantages of using containerization (e.g., Docker) for deploying machine learning models in production.",
]


# ---------------------------------------------------------------------------
# Benchmark a single (model, prompt) pair
# ---------------------------------------------------------------------------
def benchmark_single(model: str, prompt: str) -> dict:
    """
    Stream a response from *model* for *prompt* and return metrics.

    Returns a dict with keys:
        ttft   – Time to First Token (seconds)
        tps    – Tokens Per Second (from eval metadata)
        tokens – Total tokens generated
    """
    start = time.perf_counter()
    ttft = None
    final_chunk = None

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if ttft is None:
            ttft = time.perf_counter() - start

        # Print streamed text live
        content = chunk.get("message", {}).get("content", "")
        print(content, end="", flush=True)

        final_chunk = chunk

    print()  # newline after stream

    # Extract eval metadata from the final chunk
    eval_count = final_chunk.get("eval_count") if final_chunk else None
    eval_duration = final_chunk.get("eval_duration") if final_chunk else None

    tps = None
    if eval_count and eval_duration:
        tps = eval_count / (eval_duration / 1e9)

    return {
        "ttft": ttft,
        "tps": tps,
        "tokens": eval_count,
    }


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------
def run_comparison():
    # results[model] = list of metric dicts (one per prompt)
    results: dict[str, list[dict]] = {m: [] for m in MODELS}

    for model in MODELS:
        print("\n" + "=" * 70)
        print(f"  MODEL: {model}")
        print("=" * 70)

        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\n--- Prompt {i}/{len(TEST_PROMPTS)} ---")
            print(f"    {prompt[:80]}{'...' if len(prompt) > 80 else ''}\n")

            metrics = benchmark_single(model, prompt)
            results[model].append(metrics)

            ttft_str = f"{metrics['ttft']:.4f}s" if metrics["ttft"] else "N/A"
            tps_str = f"{metrics['tps']:.2f}" if metrics["tps"] else "N/A"
            print(f"\n    [TTFT: {ttft_str}  |  TPS: {tps_str}]")

    # ------------------------------------------------------------------
    # Compute averages
    # ------------------------------------------------------------------
    summary: dict[str, dict] = {}

    for model, runs in results.items():
        ttfts = [r["ttft"] for r in runs if r["ttft"] is not None]
        tpss = [r["tps"] for r in runs if r["tps"] is not None]
        total_tokens = sum(r["tokens"] for r in runs if r["tokens"] is not None)

        summary[model] = {
            "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else None,
            "avg_tps": sum(tpss) / len(tpss) if tpss else None,
            "total_tokens": total_tokens,
            "prompts_run": len(runs),
        }

    # ------------------------------------------------------------------
    # Print Markdown comparison table
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 70 + "\n")

    # Header
    print(f"| {'Model':<15} | {'Avg TTFT (s)':>14} | {'Avg TPS':>10} | {'Total Tokens':>14} |")
    print(f"|{'-'*17}|{'-'*16}|{'-'*12}|{'-'*16}|")

    for model, s in summary.items():
        avg_ttft = f"{s['avg_ttft']:.4f}" if s["avg_ttft"] else "N/A"
        avg_tps = f"{s['avg_tps']:.2f}" if s["avg_tps"] else "N/A"
        total = str(s["total_tokens"]) if s["total_tokens"] else "N/A"
        print(f"| {model:<15} | {avg_ttft:>14} | {avg_tps:>10} | {total:>14} |")

    print()

    # Declare a winner
    models_with_tps = {m: s["avg_tps"] for m, s in summary.items() if s["avg_tps"]}
    if len(models_with_tps) >= 2:
        fastest = max(models_with_tps, key=models_with_tps.get)
        slowest = min(models_with_tps, key=models_with_tps.get)
        diff_pct = ((models_with_tps[fastest] - models_with_tps[slowest]) / models_with_tps[slowest]) * 100
        print(f">> Winner by throughput: **{fastest}** ({diff_pct:+.1f}% faster than {slowest})")

    models_with_ttft = {m: s["avg_ttft"] for m, s in summary.items() if s["avg_ttft"]}
    if len(models_with_ttft) >= 2:
        quickest = min(models_with_ttft, key=models_with_ttft.get)
        print(f">> Fastest first token : **{quickest}** ({models_with_ttft[quickest]:.4f}s avg TTFT)")

    print()


if __name__ == "__main__":
    run_comparison()
