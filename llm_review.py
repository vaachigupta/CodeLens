"""
llm_review.py - LLM-powered review step for CodeLens (Milestone 2)

Takes the NEW code plus the most similar EXISTING functions retrieved from the
vector store, asks an LLM (Groq, Llama 3) to write review comments, and then
VALIDATES every citation: a comment is only kept if the function it cites was
actually one of the retrieved chunks. This stops the model from inventing
sources, which keeps the explanations trustworthy.
"""

import json
import os

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a code reviewer for ONE specific codebase.
You are given REFERENCE functions from that codebase and a NEW function.

Rules:
1. Only flag places where the NEW function deviates from conventions that are
   clearly visible in the REFERENCE functions (naming, docstrings, error
   handling, validation, return shape, typing, etc.).
2. Do NOT give generic advice that is not backed by the references.
3. Every comment must cite exactly ONE reference, using its exact REFERENCE ID.
4. If the new function follows the conventions, return an empty list.

Respond with JSON only, in this shape:
{"comments": [{"issue": "<short label>", "comment": "<1-2 sentences>", "cited_source": "<exact REFERENCE ID>"}]}
"""


def _norm(path_id: str) -> str:
    """Treat Windows (\\) and Linux (/) path separators as the same."""
    return str(path_id).replace("\\", "/")


def build_user_prompt(new_code: str, retrieved: dict) -> str:
    """Format the retrieved chunks and the new code into one prompt."""
    ids = retrieved["ids"][0]
    docs = retrieved["documents"][0]
    refs = "\n\n".join(
        f"### REFERENCE ID: {_norm(rid)}\n{doc}" for rid, doc in zip(ids, docs)
    )
    return f"{refs}\n\n### NEW FUNCTION TO REVIEW\n{new_code}"


def call_groq(system_prompt: str, user_prompt: str) -> str:
    """Call Groq's chat API and return the raw JSON string."""
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and set it in your terminal first."
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def parse_and_validate(raw: str, valid_ids: list[str]):
    """Parse the model's JSON and drop any comment citing an unknown source.

    Returns (kept_comments, dropped_comments).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], [{"error": "model did not return valid JSON", "raw": raw[:200]}]

    valid = {_norm(v) for v in valid_ids}
    kept, dropped = [], []
    for c in data.get("comments", []):
        has_text = bool(c.get("comment"))
        cites_real_source = _norm(c.get("cited_source", "")) in valid
        (kept if has_text and cites_real_source else dropped).append(c)
    return kept, dropped


def review_with_llm(new_code: str, retrieved: dict, llm_fn=call_groq):
    """Full LLM review step. llm_fn can be swapped for a fake in tests."""
    user_prompt = build_user_prompt(new_code, retrieved)
    raw = llm_fn(SYSTEM_PROMPT, user_prompt)
    return parse_and_validate(raw, retrieved["ids"][0])
