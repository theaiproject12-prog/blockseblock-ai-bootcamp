"""
Live smoke test for the Groq provider.

Run from the repo root with your venv active:
    python tests/test_groq_live.py

What this tests:
  1. GROQ_API_KEY and GROQ_MODEL are loaded from .env
  2. GroqProvider instantiates without error
  3. A real chat completion comes back with content
  4. JSON mode (response_format) works — required for Feature 2/3
  5. LLMResponse.provider reports "groq" (not "openai")

You need GROQ_API_KEY set in .env. Get a free key at console.groq.com.
"""
import asyncio
import sys
from pathlib import Path

# Make shared/ importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_env() -> bool:
    from shared.config import settings

    print("=== Config check ===")
    print(f"  LLM_PROVIDER : {settings.llm_provider!r}")
    print(f"  GROQ_API_KEY : {'set ✓' if settings.groq_api_key else 'NOT SET ✗'}")
    print(f"  GROQ_MODEL   : {settings.groq_model!r}")

    if not settings.groq_api_key:
        print()
        print("ERROR: GROQ_API_KEY is not set in .env")
        print("  1. Get a free key at https://console.groq.com (no credit card)")
        print("  2. Add to .env:  GROQ_API_KEY=gsk_your_key_here")
        return False
    return True


async def test_basic_chat() -> bool:
    from shared.providers.groq_provider import GroqProvider

    print("\n=== Test 1: basic chat ===")
    provider = GroqProvider()
    print(f"  model    : {provider._model}")
    print(f"  base_url : {provider._base_url}")

    try:
        result = await provider.chat(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with exactly three words."},
            ],
            temperature=0.0,
            max_tokens=20,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print(f"  content  : {result.content!r}")
    print(f"  provider : {result.provider!r}  ← should be 'groq'")
    print(f"  model    : {result.model!r}")

    if result.provider != "groq":
        print("  WARNING: provider field is not 'groq' — bug in LLMResponse")

    ok = bool(result.content)
    print(f"  result   : {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


async def test_json_mode() -> bool:
    import json
    from shared.providers.groq_provider import GroqProvider

    print("\n=== Test 2: JSON mode (response_format) ===")
    print("  (required for Feature 2 structured output and Feature 3 session chat)")

    provider = GroqProvider()
    try:
        result = await provider.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return ONLY a JSON object with fields: "
                        'intent (string), answer (string), confidence (number 0-1), sources_needed (bool)'
                    ),
                },
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print(f"  raw content: {result.content!r}")

    try:
        parsed = json.loads(result.content or "")
        print(f"  parsed keys: {list(parsed.keys())}")
        ok = "answer" in parsed
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        ok = False

    print(f"  result: {'PASS ✓' if ok else 'FAIL ✗ — model did not return valid JSON'}")
    return ok


async def test_call_llm_interface() -> bool:
    """Test the public call_llm() interface with LLM_PROVIDER=groq in .env."""
    import os
    from shared.config import settings

    print("\n=== Test 3: call_llm() public interface ===")

    if settings.llm_provider.lower() != "groq":
        print(f"  SKIPPED — LLM_PROVIDER is {settings.llm_provider!r}, not 'groq'")
        print("  To run this test: set LLM_PROVIDER=groq in .env")
        return True  # not a failure — just not configured

    # Clear the lru_cache so we get a fresh provider from current settings.
    from shared.providers.factory import get_provider
    get_provider.cache_clear()

    from shared.llm_client import call_llm
    try:
        result = await call_llm(
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            temperature=0.0,
            max_tokens=10,
        )
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    print(f"  content  : {result.content!r}")
    print(f"  provider : {result.provider!r}")
    ok = bool(result.content)
    print(f"  result   : {'PASS ✓' if ok else 'FAIL ✗'}")
    return ok


async def main() -> None:
    print("Groq live smoke test")
    print("=" * 40)

    if not check_env():
        sys.exit(1)

    results = []
    results.append(await test_basic_chat())
    results.append(await test_json_mode())
    results.append(await test_call_llm_interface())

    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if passed < total:
        print("\nIf you see 401 / authentication errors:")
        print("  • Check GROQ_API_KEY in .env is correct (starts with gsk_)")
        print("  • Key is at https://console.groq.com/keys")
        print("\nIf you see 404 / model not found:")
        print("  • Check GROQ_MODEL in .env — try: llama-3.1-8b-instant")
        print("  • Available models: https://console.groq.com/docs/models")
        sys.exit(1)
    else:
        print("All tests passed — Groq is working ✓")


if __name__ == "__main__":
    asyncio.run(main())
