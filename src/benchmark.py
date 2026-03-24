"""
Ollama LLM Benchmark Script
============================
Queries the local Ollama llama3.2 model with a streaming response
and reports Time to First Token (TTFT) and Tokens Per Second (TPS).
"""

import time
import ollama


def run_benchmark():
    """Stream a response from llama3.2 and collect performance metrics."""

    prompt = "Explain the difference between threading and multiprocessing in Python."

    print("=" * 60)
    print("  Ollama Benchmark — llama3.2")
    print("=" * 60)
    print(f"\n📝 Prompt: {prompt}\n")
    print("-" * 60)

    start_time = time.perf_counter()
    ttft = None
    final_message = None

    stream = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        # Record Time to First Token on the very first chunk
        if ttft is None:
            ttft = time.perf_counter() - start_time

        # Print streamed text as it arrives
        content = chunk.get("message", {}).get("content", "")
        print(content, end="", flush=True)

        # Keep a reference to the last chunk (contains metadata)
        final_message = chunk

    total_time = time.perf_counter() - start_time

    # ------------------------------------------------------------------
    # Extract metadata from the final chunk
    # ------------------------------------------------------------------
    eval_count = None
    eval_duration = None

    if final_message:
        eval_count = final_message.get("eval_count")
        eval_duration = final_message.get("eval_duration")

    # Calculate Tokens Per Second
    tps = None
    if eval_count and eval_duration:
        tps = eval_count / (eval_duration / 1e9)

    # ------------------------------------------------------------------
    # Print Benchmark Report
    # ------------------------------------------------------------------
    print("\n")
    print("=" * 60)
    print("  📊 Benchmark Report")
    print("=" * 60)
    print(f"  Time to First Token (TTFT) : {ttft:.4f} s" if ttft else "  TTFT : N/A")
    print(f"  Total Generation Time      : {total_time:.4f} s")
    if eval_count is not None:
        print(f"  Tokens Generated           : {eval_count}")
    if tps is not None:
        print(f"  Tokens Per Second (TPS)    : {tps:.2f}")
    else:
        print("  Tokens Per Second (TPS)    : N/A (metadata not available)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
