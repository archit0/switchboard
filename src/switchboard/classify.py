"""Triage: decide how hard a request is, as cheaply as possible.

Two signals, combined:
  1. Heuristics (free, instant): length, code/math markers, multi-step verbs.
  2. A tiny LLM classifier (gemini flash-lite, ~$0.0001/call) that reads a
     truncated prefix of the prompt and returns difficulty 1-5 + domain.

The LLM call is skipped when heuristics are confident (very short trivial
prompts, or obvious giant code/math prompts), so most requests pay nothing for
triage. The classifier only earns its keep on the ambiguous middle.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from . import config
from .gateway import Gateway

_CODE_RE = re.compile(r"```|def |class |import |function |SELECT |#include|=>|console\.|public static")
_MATH_RE = re.compile(r"\b(prove|integral|derivative|theorem|equation|matrix|probability|\d+\s*[+\-*/^]\s*\d+)\b", re.I)
_HARD_VERBS = re.compile(
    r"\b(design|architect|optimi[sz]e|prove|derive|refactor|debug|analy[sz]e|"
    r"compare|trade-?off|explain why|step by step|plan|strategy|edge cases?)\b",
    re.I,
)

DOMAINS = ("code", "math", "reasoning", "factual", "creative", "chat", "other")


@dataclass
class Triage:
    difficulty: float  # 1.0 (trivial) .. 5.0 (very hard)
    domain: str
    tier: str  # "cheap" | "mid" | "hard"
    source: str  # "heuristic" | "llm"
    classifier_cost: float = 0.0
    classifier_ms: float = 0.0
    note: str = ""


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):  # OpenAI content-parts form
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _tier_for(difficulty: float) -> str:
    if difficulty <= 2.0:
        return "cheap"
    if difficulty <= 3.4:
        return "mid"
    return "hard"


def _heuristic(text: str) -> tuple[float, str, bool]:
    """Return (difficulty_prior, domain_guess, confident)."""
    n = len(text)
    code = bool(_CODE_RE.search(text))
    math = bool(_MATH_RE.search(text))
    hard = len(_HARD_VERBS.findall(text))

    domain = "chat"
    if code:
        domain = "code"
    elif math:
        domain = "math"
    elif hard:
        domain = "reasoning"

    # Very short and no complexity markers -> confidently trivial.
    if n < 80 and not (code or math or hard):
        return 1.5, domain, True
    # Huge prompt with code/math and multiple hard verbs -> confidently hard.
    if (n > 2500 or hard >= 3) and (code or math or hard >= 2):
        return 4.5, domain, True

    # Otherwise produce a prior but defer to the LLM.
    prior = 2.0
    prior += min(n / 1500.0, 1.5)
    prior += 0.6 * min(hard, 3)
    prior += 0.5 if code else 0.0
    prior += 0.5 if math else 0.0
    return max(1.0, min(prior, 5.0)), domain, False


_CLASSIFIER_SYS = (
    "You are a fast request-difficulty classifier for an LLM router. "
    "Read the user request and rate how powerful a model it needs. "
    "Respond with ONLY a compact JSON object, no prose:\n"
    '{"difficulty": <1-5 int>, "domain": "code|math|reasoning|factual|creative|chat|other"}\n'
    "Scale: 1=trivial (greeting, lookup), 2=easy (short factual, simple rewrite), "
    "3=moderate (normal coding/explanation), 4=hard (multi-step reasoning, non-trivial "
    "code, careful analysis), 5=very hard (research-grade proof, complex system design)."
)


async def triage(gw: Gateway, messages: list[dict], *, use_llm: bool = True) -> Triage:
    text = _last_user_text(messages)
    prior, domain, confident = _heuristic(text)

    if confident or not use_llm:
        return Triage(
            prior, domain, _tier_for(prior), "heuristic", note="heuristic-confident" if confident else "llm-disabled"
        )

    prefix = text[:1800]
    comp = await gw.complete(
        config.CLASSIFIER_MODEL,
        [
            {"role": "system", "content": _CLASSIFIER_SYS},
            {"role": "user", "content": f"Request prefix (len={len(text)} chars):\n{prefix}"},
        ],
        max_tokens=40,
    )

    difficulty, dom = prior, domain
    if comp.ok:
        try:
            m = re.search(r"\{.*\}", comp.content, re.S)
            obj = json.loads(m.group(0)) if m else {}
            d = float(obj.get("difficulty", prior))
            difficulty = 0.5 * d + 0.5 * prior  # blend model judgement with prior
            dom = obj.get("domain", domain) if obj.get("domain") in DOMAINS else domain
        except Exception:  # noqa: BLE001
            difficulty = prior

    return Triage(
        difficulty=difficulty,
        domain=dom,
        tier=_tier_for(difficulty),
        source="llm",
        classifier_cost=comp.cost,
        classifier_ms=comp.latency_ms,
    )
