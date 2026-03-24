"""
Structured Output with Ollama + Pydantic
=========================================
Forces llama3.2 to return strict JSON matching a Pydantic schema,
with a single-retry mechanism on validation failure.
"""

import json
import sys
import os

# Force UTF-8 output on Windows to avoid cp1252 encoding errors with emoji
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import ollama
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Pydantic Schema
# ---------------------------------------------------------------------------
class TechExplanation(BaseModel):
    concept_name: str
    short_definition: str
    key_differences: list[str]


# ---------------------------------------------------------------------------
# System prompt that instructs the model to output conforming JSON
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a technical assistant. You MUST respond ONLY with valid JSON — "
    "no markdown, no commentary, no extra text.\n\n"
    "The JSON object MUST have exactly these fields:\n"
    '  • "concept_name"      — a string naming the concept being explained\n'
    '  • "short_definition"  — a one-sentence definition\n'
    '  • "key_differences"   — an array of strings, each describing one key difference\n\n'
    "Return ONLY the JSON object."
)


def _call_model(prompt: str) -> str:
    """Send a chat request to llama3.2 with format='json' and return raw text."""
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )
    return response["message"]["content"]


def get_structured_response(prompt: str) -> TechExplanation:
    """
    Query llama3.2 and validate the JSON response against TechExplanation.

    If validation fails, retries once before raising the exception.
    """
    max_attempts = 2  # initial attempt + 1 retry

    for attempt in range(1, max_attempts + 1):
        raw = _call_model(prompt)

        try:
            result = TechExplanation.model_validate_json(raw)
            return result
        except ValidationError as exc:
            if attempt < max_attempts:
                print(f"⚠️  Validation failed (attempt {attempt}/{max_attempts}), retrying …")
                print(f"    Error: {exc.error_count()} validation error(s)")
            else:
                print(f"❌ Validation failed after {max_attempts} attempts.")
                print(f"    Raw response was:\n{raw}")
                raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prompt = "Explain the difference between TCP and UDP."

    print("=" * 60)
    print("  Structured Output — llama3.2 + Pydantic")
    print("=" * 60)
    print(f"\n📝 Prompt: {prompt}\n")

    result = get_structured_response(prompt)

    print("✅ Validated response:\n")
    print(json.dumps(result.model_dump(), indent=2))
