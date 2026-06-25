"""
Smart Router starter — Feature 6, Part A.

Your task: implement classify_query() so the Smart Router can decide
whether and how to retrieve before generating a response.

The function signature and return shape are fixed — only fill in the
marked TODO sections. The solution is in solution/main.py and shared/router.py.

WHAT YOU'RE BUILDING:
  A pre-retrieval intelligence step: instead of retrieving for EVERY query,
  the router first asks the LLM "does this query even need documents?"
  This is the "Anti-RAG" pattern — RAG with a gating step.

  Example classifications:
    "What is 2 + 2?"         → needs_retrieval=False, confidence=0.95
    "What is your refund policy?" → needs_retrieval=True, confidence=0.85
    "Tell me about your contracts" → needs_retrieval=True, query_type="professional_document"
    "Hmm, I'm not sure"     → needs_retrieval=True, confidence=0.45 (hybrid)

TWO-CALL ARCHITECTURE:
  Call 1: classify_query() — cheap, temperature=0.1, asks only "should I retrieve?"
  Call 2: smart_chat() in main.py — the actual response generation with or without context

  This separation keeps each LLM call focused. The classifier is a specialist;
  the responder is a generalist with (optionally) injected context.
"""
import json

from shared.llm_client import call_llm

# ============================================================
# TODO STEP 1: Write the classifier system prompt
#
# The LLM must return a JSON object with these exact fields:
#   {
#     "needs_retrieval": bool,
#     "confidence": float (0.0–1.0),
#     "reasoning": str (one sentence),
#     "query_type": "general" | "domain" | "professional_document" | "ambiguous"
#   }
#
# Query type definitions to include in your prompt:
#   "general"               — common knowledge, greetings, math, basic facts.
#                             No domain documents needed.
#   "domain"                — questions about the organization's products,
#                             services, policies, or procedures.
#   "professional_document" — requires precise navigation of structured
#                             professional documents (financial filings,
#                             legal contracts, regulatory docs).
#                             Answer requires finding a SPECIFIC SECTION,
#                             not just a semantically similar passage.
#   "ambiguous"             — genuinely unclear whether documents help.
#
# Confidence guide to include:
#   0.9–1.0  very clear (obvious greeting vs obvious domain question)
#   0.7–0.8  reasonably clear, some uncertainty
#   0.4–0.6  genuinely ambiguous — could go either way
#   below 0.4  you really can't tell
#
# Important: instruct the LLM to respond ONLY with the JSON object —
# no markdown, no extra text.
# ============================================================

_CLASSIFIER_SYSTEM_PROMPT = """
# TODO: write the classifier system prompt here.
# See the docstring above for what fields to request and what each means.
# Remove this comment and replace with your prompt string.
"""


async def classify_query(query: str) -> dict:
    """
    Classify whether a query needs document retrieval and how to retrieve.

    Returns:
      needs_retrieval: bool  — True → retrieve; False → answer directly
      confidence:      float — classifier's certainty (0.0–1.0)
      reasoning:       str   — one-sentence explanation (for debugging)
      query_type:      str   — "general"|"domain"|"professional_document"|"ambiguous"

    Fallback: if classification fails, returns needs_retrieval=True, confidence=0.5
    (hybrid path) — better to retrieve unnecessarily than to miss needed context.
    """
    # ============================================================
    # TODO STEP 2: Call the LLM with the classifier system prompt
    #
    # Hints:
    #   - Use call_llm() from shared.llm_client
    #   - Pass temperature=0.1 (low, for deterministic classification)
    #   - Pass response_format={"type": "json_object"}
    #   - Build the messages list with:
    #       system: _CLASSIFIER_SYSTEM_PROMPT
    #       user:   the query string
    # ============================================================

    raise NotImplementedError(
        "TODO: call the LLM with _CLASSIFIER_SYSTEM_PROMPT and temperature=0.1. "
        "See the docstring and hints above."
    )

    # ============================================================
    # TODO STEP 3: Parse the LLM response and return a dict
    #
    # Hints:
    #   - json.loads(result.content or "{}") to parse the JSON
    #   - Extract: needs_retrieval (bool), confidence (float, clamp 0.0–1.0),
    #     reasoning (str), query_type (str)
    #   - Wrap in try/except — if parsing fails, return the safe fallback:
    #       {
    #           "needs_retrieval": True,
    #           "confidence": 0.5,
    #           "reasoning": "Classification failed — defaulting to retrieval.",
    #           "query_type": "ambiguous",
    #       }
    # ============================================================


# ============================================================
# PAGEINDEX ROUTING (read-only — no implementation needed here)
#
# When your classify_query returns query_type="professional_document"
# AND ENABLE_PAGEINDEX=true, the router in main.py will take the
# PageIndex path instead of vector search.
#
# This comment is here to show you WHERE that decision happens:
# in main.py, inside smart_chat(), after you return from this function.
#
# See shared/router.py (the PAGEINDEX_ROUTING block) for the full
# integration pattern and setup instructions.
# ============================================================
