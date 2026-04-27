# 🧠 Local AI Assistant & Benchmarking Suite

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLMs-black?style=for-the-badge&logo=ollama)
![Pydantic](https://img.shields.io/badge/Pydantic-Validation-e92063?style=for-the-badge&logo=pydantic)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A focused, high-performance toolkit for interacting with, benchmarking, and evaluating local Large Language Models (LLMs) using **Ollama**. This repository demonstrates practical patterns for local AI engineering, including performance measurement (TTFT/TPS), multi-model comparison, and robust structured JSON generation.

---

## ✨ Features

- **🚀 Performance Benchmarking**: Precisely measure streaming performance metrics including **Time to First Token (TTFT)** and **Tokens Per Second (TPS)**.
- **📊 Automated Model Comparison**: Run side-by-side evaluations of multiple local models (e.g., `llama3.2`, `phi3`) across a suite of test prompts. Automatically calculates averages and declares winners based on latency and throughput.
- **🧱 Structured Output Generation**: Enforce strict JSON schemas on local LLMs using **Pydantic**. Includes a built-in self-correction and retry mechanism for handling schema validation failures.

## 📂 Project Structure

```text
local-ai-assistant/
├── src/
│   ├── benchmark.py         # Single-model streaming benchmark (TTFT, TPS)
│   ├── compare_models.py    # Multi-model evaluation and comparison suite
│   └── structured_output.py # Pydantic-validated JSON output with retries
├── .gitignore
└── README.md
```

## 🛠️ Prerequisites

To run these scripts, you need the following installed:

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** running locally.
3. The necessary Ollama models pulled to your local machine:
   ```bash
   ollama run llama3.2
   ollama run phi3
   ```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/local-ai-assistant.git
   cd local-ai-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install the required Python packages:
   ```bash
   pip install ollama pydantic
   ```

## 🚀 Usage Guide

### 1. Single Model Benchmark (`benchmark.py`)
Queries a specific model (`llama3.2`) with a streaming prompt and calculates exact latency and throughput metrics.

```bash
python src/benchmark.py
```
**Sample Output:**
```text
============================================================
  📊 Benchmark Report
============================================================
  Time to First Token (TTFT) : 0.8421 s
  Total Generation Time      : 4.1205 s
  Tokens Generated           : 145
  Tokens Per Second (TPS)    : 35.19
============================================================
```

### 2. Multi-Model Comparison (`compare_models.py`)
Pits multiple models against a suite of logical, coding, and summarization tasks, generating a markdown table of the results.

```bash
python src/compare_models.py
```
**Sample Output:**
```text
======================================================================
  MODEL COMPARISON RESULTS
======================================================================

| Model           |   Avg TTFT (s) |    Avg TPS |   Total Tokens |
|-----------------|----------------|------------|----------------|
| llama3.2        |         0.7512 |      38.45 |            412 |
| phi3            |         0.6105 |      42.10 |            380 |

>> Winner by throughput: **phi3** (+9.5% faster than llama3.2)
>> Fastest first token : **phi3** (0.6105s avg TTFT)
```

### 3. Structured Output Validation (`structured_output.py`)
Demonstrates how to force an LLM to reply in strict JSON matching a predefined Pydantic schema, complete with automatic retries if the LLM hallucinates an invalid structure.

```bash
python src/structured_output.py
```

### 🧪 Running Unit Tests
This project includes a comprehensive test suite using `pytest` and `unittest.mock` to ensure reliability without requiring a live Ollama connection during CI/CD.

```bash
# Set PYTHONPATH so pytest can discover the src module
$env:PYTHONPATH="."
pytest -v
```

## 💡 Why This Project?

This repository serves as a technical portfolio piece demonstrating proficiency in:
- **Local AI Infrastructure**: Bypassing cloud APIs for secure, edge-based inference.
- **LLM Observability**: Tracking critical metrics like TTFT, which is crucial for user experience in streaming applications.
- **Reliable AI Engineering**: Moving beyond standard chat wrappers to build deterministic, schema-enforced systems that can be integrated into larger software pipelines.

---
*Built with ❤️ using Python and Ollama.*
