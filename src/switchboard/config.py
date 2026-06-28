"""Model pool, pricing and routing tiers.

IMPORTANT: the prices below are *public list-price proxies* in USD per 1M
tokens. They are roughly correct relative to each other (which is what makes
the routing decisions sensible), but they are almost certainly NOT what your
gateway actually bills you. Drop your real rate card into `pricing.json`
in the repo root and it will override these at load time.

The whole point of the router is the *ratios* between tiers: a cheap model is
~10-50x cheaper than Opus, a mid model ~3-10x cheaper. As long as those ratios
hold, the cost story holds.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Pricing: USD per 1,000,000 tokens (input, output).
# --------------------------------------------------------------------------- #
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    # cheap tier
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gpt-5-nano": (0.05, 0.40),
    "claude-haiku-4-5": (1.00, 5.00),
    # mid tier
    "gemini-3.5-flash": (0.30, 2.50),
    "gpt-5.4-mini": (0.25, 2.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    # strong tier (escalation targets + baselines)
    "claude-opus-4-8": (15.00, 75.00),
    "claude-fable-5": (15.00, 75.00),  # placeholder — real price unknown
    "gpt-5.5": (1.25, 10.00),
    "gemini-3.1-pro-preview": (1.25, 10.00),
}


def _load_pricing() -> dict[str, tuple[float, float]]:
    pricing = dict(_DEFAULT_PRICING)
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pricing.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                override = json.load(f)
            for k, v in override.items():
                pricing[k] = (float(v[0]), float(v[1]))
        except Exception as e:  # noqa: BLE001 - best effort, never crash on bad file
            print(f"[config] warning: could not load pricing.json: {e}")
    return pricing


PRICING = _load_pricing()


def price_of(model: str) -> tuple[float, float]:
    """(input, output) USD per 1M tokens. Falls back to a mid-tier guess."""
    return PRICING.get(model, (1.0, 5.0))


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = price_of(model)
    return (prompt_tokens / 1e6) * pin + (completion_tokens / 1e6) * pout


# --------------------------------------------------------------------------- #
# Tiers and pools. Edit these to change routing behaviour.
# --------------------------------------------------------------------------- #
# NOTE: the gateway's /v1/models list is stale — it advertises some ids that
# 404 at call time (e.g. gemini-3-pro-preview, claude-fable-5 -> "use Opus 4.8").
# Every model below has been verified to actually answer. Run
# `switchboard models` to re-check after any edit.
CHEAP = ["gemini-3.1-flash-lite", "gpt-5-nano", "claude-haiku-4-5"]
MID = ["gemini-3.5-flash", "gpt-5.4-mini", "claude-sonnet-4-6"]
STRONG = ["claude-opus-4-8", "gpt-5.5", "gemini-3.1-pro-preview"]

# Reliable, cheap, low-latency model used for triage + judging. Gemini
# flash-lite was the most reliable for short structured outputs in testing
# (gpt-5-nano spent its whole budget on hidden reasoning and returned empty).
CLASSIFIER_MODEL = "gemini-3.1-flash-lite"
JUDGE_MODEL = "gemini-3.1-flash-lite"

# Default single-model picks per tier.
DEFAULT_CHEAP = "gemini-3.1-flash-lite"
DEFAULT_MID = "gemini-3.5-flash"

# Mixture-of-Agents (used for the hard tier). Diverse *providers* give
# diversity without needing temperature (gpt-5 reasoning models reject
# non-default temperature anyway).
MOA_PROPOSERS = ["gemini-3.5-flash", "gpt-5.4-mini", "claude-haiku-4-5"]
MOA_SYNTHESIZER = "claude-sonnet-4-6"

# Baselines we benchmark the router against ("always use the big model").
# claude-fable-5 is advertised by the gateway but 404s ("use Opus 4.8"), so the
# practical frontier baseline here is Opus 4.8.
BASELINE_MODELS = ["claude-opus-4-8"]
PRIMARY_BASELINE = "claude-opus-4-8"


@dataclass
class GatewayConfig:
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> GatewayConfig:
        base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        if not base or not key:
            raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY must be set in the environment.")
        return cls(base_url=base, api_key=key)
