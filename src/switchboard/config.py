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
    "gemini-flash-lite-latest": (0.10, 0.40),  # alias: auto-tracks newest flash-lite
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gpt-5.4-nano": (0.05, 0.40),
    "gpt-5-nano": (0.05, 0.40),
    "claude-haiku-4-5": (1.00, 5.00),
    # mid tier
    "gemini-flash-latest": (0.30, 2.50),  # alias: auto-tracks newest flash
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
# Model pool & tiers — the router's "brain". This is what makes switchboard a
# tiny agent rather than a static proxy: a controller model triages, a judge
# verifies, and a Mixture-of-Agents deliberates on the hard cases.
#
# Config-driven so it never goes stale: edit the defaults below, OR drop a
# `models.json` in the repo root to override any key without touching code
# (same idea as pricing.json). We default to the `-latest` aliases for the
# cheap/mid tiers so the pool auto-tracks the newest models the gateway serves.
#
# NOTE: the gateway's /v1/models list is stale (advertises ids that 404), so
# always re-check with `switchboard models` after changing the pool.
# --------------------------------------------------------------------------- #
_DEFAULT_POOL: dict = {
    "cheap": ["gemini-flash-lite-latest", "gpt-5.4-nano", "claude-haiku-4-5"],
    "mid": ["gemini-flash-latest", "gpt-5.4-mini", "claude-sonnet-4-6"],
    "strong": ["claude-opus-4-8", "gpt-5.5", "gemini-3.1-pro-preview"],
    # The agent's controller: a cheap, fast, reliable model for triage + judging.
    "classifier": "gemini-flash-lite-latest",
    "judge": "gemini-flash-lite-latest",
    # Default single-model pick per tier.
    "default_cheap": "gemini-flash-lite-latest",
    "default_mid": "gemini-flash-latest",
    # Mixture-of-Agents for the hard tier. Diverse *providers* give diversity
    # without temperature (gpt-5 reasoning models reject non-default temperature).
    "moa_proposers": ["gemini-flash-latest", "gpt-5.4-mini", "claude-haiku-4-5"],
    "moa_synthesizer": "claude-sonnet-4-6",
    # Frontier baseline the router is benchmarked against ("always the big model").
    "baselines": ["claude-opus-4-8"],
    "primary_baseline": "claude-opus-4-8",
}


def _load_pool() -> dict:
    pool = dict(_DEFAULT_POOL)
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                pool.update(json.load(f))
        except Exception as e:  # noqa: BLE001 - best effort, never crash on bad file
            print(f"[config] warning: could not load models.json: {e}")
    return pool


_POOL = _load_pool()

CHEAP: list[str] = _POOL["cheap"]
MID: list[str] = _POOL["mid"]
STRONG: list[str] = _POOL["strong"]
CLASSIFIER_MODEL: str = _POOL["classifier"]
JUDGE_MODEL: str = _POOL["judge"]
DEFAULT_CHEAP: str = _POOL["default_cheap"]
DEFAULT_MID: str = _POOL["default_mid"]
MOA_PROPOSERS: list[str] = _POOL["moa_proposers"]
MOA_SYNTHESIZER: str = _POOL["moa_synthesizer"]
BASELINE_MODELS: list[str] = _POOL["baselines"]
PRIMARY_BASELINE: str = _POOL["primary_baseline"]


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
