"""The router engine: triage -> policy -> execute -> telemetry.

Three strategies are composed by the policy:

  single      one model answers (trivial / moderate traffic — the common case).
  moa         Mixture-of-Agents: N diverse models propose in parallel, a
              synthesizer fuses them. This is the lever that can *beat* a single
              frontier model on hard queries, at well below frontier cost.
  cascade     FrugalGPT-style: answer cheap, a cheap judge scores it, escalate
              only if the score is low. Minimises spend on the easy majority.

Every result reports its internal cost and an estimate of what always-Opus
would have cost, so savings are measured, not asserted.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from . import config
from .cache import ResponseCache
from .classify import Triage, triage
from .gateway import Completion, Gateway


@dataclass
class RouteResult:
    content: str
    route: str
    tier: str
    models_used: list[str]
    prompt_tokens: int
    completion_tokens: int
    cost: float
    baseline_cost_est: float
    savings_pct: float
    latency_ms: float
    cached: bool
    triage: Triage
    steps: list[dict] = field(default_factory=list)


_JUDGE_SYS = (
    "You are a strict answer-quality judge. Given a user request and a candidate "
    "answer, rate how fully and correctly the answer satisfies the request. "
    'Respond ONLY with JSON: {"score": <0.0-1.0>, "reason": "<short>"}. '
    "0.0 = wrong/empty/evasive, 1.0 = complete and correct."
)

_SYNTH_SYS = (
    "You are an expert synthesizer in a Mixture-of-Agents system. You are given a "
    "user request and several candidate answers from different models. The "
    "candidates may be uneven or partly wrong. Critically compare them, discard "
    "errors, and produce a single best answer that is more accurate and complete "
    "than any individual candidate. Do not mention the candidates or the process; "
    "just give the final answer to the user."
)


def _user_request_text(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role")
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        if role in ("user", "system") and c:
            parts.append(f"{role}: {c}")
    return "\n".join(parts)[:6000]


class Engine:
    def __init__(self, gateway: Gateway | None = None, cache: ResponseCache | None = None):
        self.gw = gateway or Gateway()
        self.cache = cache if cache is not None else ResponseCache()

    async def aclose(self) -> None:
        await self.gw.aclose()

    # ---- primitives ------------------------------------------------------- #
    async def _single(self, model: str, messages: list[dict], max_tokens: int) -> Completion:
        return await self.gw.complete(model, messages, max_tokens=max_tokens)

    async def _judge(self, messages: list[dict], answer: str) -> tuple[float, Completion]:
        req = _user_request_text(messages)
        comp = await self.gw.complete(
            config.JUDGE_MODEL,
            [
                {"role": "system", "content": _JUDGE_SYS},
                {"role": "user", "content": f"REQUEST:\n{req}\n\nANSWER:\n{answer[:4000]}"},
            ],
            max_tokens=60,
        )
        score = 0.5
        if comp.ok:
            try:
                m = re.search(r"\{.*\}", comp.content, re.S)
                score = float(json.loads(m.group(0)).get("score", 0.5)) if m else 0.5
            except Exception:  # noqa: BLE001
                score = 0.5
        return max(0.0, min(score, 1.0)), comp

    async def _moa(
        self, messages: list[dict], proposers: list[str], synthesizer: str, max_tokens: int
    ) -> tuple[Completion, list[Completion]]:
        # Fan out proposers concurrently — this is the parallelism that keeps MoA
        # from being N times slower; wall-clock ≈ slowest proposer + synthesizer.
        proposals = await asyncio.gather(*(self.gw.complete(m, messages, max_tokens=max_tokens) for m in proposers))
        good = [p for p in proposals if p.ok]
        if not good:
            # everything failed; fall back to a single mid model
            fb = await self._single(config.DEFAULT_MID, messages, max_tokens)
            return fb, list(proposals)

        req = _user_request_text(messages)
        bundle = "\n\n".join(f"--- Candidate {i + 1} (model {p.model}) ---\n{p.content}" for i, p in enumerate(good))
        synth = await self.gw.complete(
            synthesizer,
            [
                {"role": "system", "content": _SYNTH_SYS},
                {"role": "user", "content": f"USER REQUEST:\n{req}\n\nCANDIDATE ANSWERS:\n{bundle}"},
            ],
            max_tokens=max_tokens,
        )
        return synth, list(proposals)

    # ---- top-level -------------------------------------------------------- #
    async def answer(
        self,
        messages: list[dict],
        *,
        mode: str = "balanced",
        max_tokens: int = 1024,
        use_llm_triage: bool = True,
        use_cache: bool = True,
    ) -> RouteResult:
        loop = asyncio.get_event_loop()
        t_start = loop.time()

        if use_cache:
            cached = self.cache.get(messages, mode)
            if cached is not None:
                c: RouteResult = cached
                return RouteResult(**{**c.__dict__, "cached": True, "latency_ms": 0.0})

        tri = await triage(self.gw, messages, use_llm=use_llm_triage)
        steps: list[dict] = [{"stage": "triage", **_triage_dict(tri)}]

        spent = tri.classifier_cost
        models_used: list[str] = []

        # ------------------------------------------------------------------ #
        # Policy
        # ------------------------------------------------------------------ #
        if mode == "cost":
            final, extra_cost, used, route = await self._cascade(messages, tri, max_tokens, steps)
            spent += extra_cost
            models_used += used
        else:
            tier = tri.tier
            if mode == "quality":  # bias one tier up
                tier = {"cheap": "mid", "mid": "hard", "hard": "hard"}[tier]

            if tier == "cheap":
                comp = await self._single(config.DEFAULT_CHEAP, messages, max_tokens)
                final, route, used = comp, f"single:{config.DEFAULT_CHEAP}", [config.DEFAULT_CHEAP]
                spent += comp.cost
                steps.append(_step("single", comp))
            elif tier == "mid":
                comp = await self._single(config.DEFAULT_MID, messages, max_tokens)
                final, route, used = comp, f"single:{config.DEFAULT_MID}", [config.DEFAULT_MID]
                spent += comp.cost
                steps.append(_step("single", comp))
            else:  # hard -> Mixture-of-Agents
                synth, props = await self._moa(messages, config.MOA_PROPOSERS, config.MOA_SYNTHESIZER, max_tokens)
                used = [p.model for p in props] + [config.MOA_SYNTHESIZER]
                route = f"moa[{'+'.join(config.MOA_PROPOSERS)}]->{config.MOA_SYNTHESIZER}"
                spent += sum(p.cost for p in props) + synth.cost
                for p in props:
                    steps.append(_step("moa-proposer", p))
                steps.append(_step("moa-synth", synth))
                final = synth
            models_used += used

        # ------------------------------------------------------------------ #
        # Telemetry + baseline comparison
        # ------------------------------------------------------------------ #
        rep_pt = final.prompt_tokens or _est_tokens(messages)
        rep_ct = final.completion_tokens or _est_tokens_text(final.content)
        baseline = config.cost_usd(config.PRIMARY_BASELINE, rep_pt, rep_ct)
        savings = (1 - spent / baseline) * 100 if baseline > 0 else 0.0

        result = RouteResult(
            content=final.content,
            route=route,
            tier=tri.tier,
            models_used=models_used,
            prompt_tokens=rep_pt,
            completion_tokens=rep_ct,
            cost=spent,
            baseline_cost_est=baseline,
            savings_pct=savings,
            latency_ms=(loop.time() - t_start) * 1000,
            cached=False,
            triage=tri,
            steps=steps,
        )
        if use_cache and final.ok:
            self.cache.put(messages, mode, result)
        return result

    async def _cascade(
        self, messages: list[dict], tri: Triage, max_tokens: int, steps: list[dict]
    ) -> tuple[Completion, float, list[str], str]:
        """FrugalGPT cascade: cheap -> judge -> mid -> judge -> MoA."""
        spent = 0.0
        used: list[str] = []

        c1 = await self._single(config.DEFAULT_CHEAP, messages, max_tokens)
        spent += c1.cost
        used.append(config.DEFAULT_CHEAP)
        steps.append(_step("cascade-cheap", c1))
        s1, j1 = await self._judge(messages, c1.content)
        spent += j1.cost
        steps.append({"stage": "cascade-judge", "score": round(s1, 3), "cost": j1.cost})
        if c1.ok and s1 >= 0.70:
            return c1, spent, used, f"cascade:single:{config.DEFAULT_CHEAP}(score={s1:.2f})"

        c2 = await self._single(config.DEFAULT_MID, messages, max_tokens)
        spent += c2.cost
        used.append(config.DEFAULT_MID)
        steps.append(_step("cascade-mid", c2))
        s2, j2 = await self._judge(messages, c2.content)
        spent += j2.cost
        steps.append({"stage": "cascade-judge", "score": round(s2, 3), "cost": j2.cost})
        if c2.ok and s2 >= 0.65:
            return c2, spent, used, f"cascade:single:{config.DEFAULT_MID}(score={s2:.2f})"

        synth, props = await self._moa(messages, config.MOA_PROPOSERS, config.MOA_SYNTHESIZER, max_tokens)
        spent += sum(p.cost for p in props) + synth.cost
        used += [p.model for p in props] + [config.MOA_SYNTHESIZER]
        for p in props:
            steps.append(_step("cascade-moa-proposer", p))
        steps.append(_step("cascade-moa-synth", synth))
        return synth, spent, used, f"cascade:moa->{config.MOA_SYNTHESIZER}"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _est_tokens(messages: list[dict]) -> int:
    txt = _user_request_text(messages)
    return max(1, len(txt) // 4)


def _est_tokens_text(text: str) -> int:
    return max(1, len(text) // 4)


def _triage_dict(t: Triage) -> dict:
    return {
        "difficulty": round(t.difficulty, 2),
        "domain": t.domain,
        "tier": t.tier,
        "source": t.source,
        "cost": round(t.classifier_cost, 8),
    }


def _step(stage: str, c: Completion) -> dict:
    return {
        "stage": stage,
        "model": c.model,
        "ok": c.ok,
        "prompt_tokens": c.prompt_tokens,
        "completion_tokens": c.completion_tokens,
        "cost": c.cost,
        "latency_ms": round(c.latency_ms, 1),
        "error": c.error,
    }
