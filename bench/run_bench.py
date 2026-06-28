"""Run the benchmark: baselines vs. router, accuracy vs. cost.

Usage:
    .venv/bin/python -m bench.run_bench               # full set, all configs
    .venv/bin/python -m bench.run_bench --n 8         # first 8 items (quick)
    .venv/bin/python -m bench.run_bench --configs opus,router-balanced

Prints a Pareto table: accuracy, total cost, $/correct, mean latency.
The headline number is whether a router config lands cheaper-than-Opus at
equal-or-better accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys

from switchboard import config
from switchboard.engine import Engine
from switchboard.gateway import Gateway

sys.path.insert(0, ".")
from bench.dataset import DATASET, Item  # noqa: E402

# Independent grader for open-ended items — a strong, live model that is NOT a
# member of the MoA pool, so it never grades its own work.
GRADER_MODEL = "gemini-3.1-pro-preview"

GRADER_SYS = (
    "You are a rigorous grader. Given a QUESTION, a REFERENCE describing what a "
    "correct answer must contain, and a candidate ANSWER, decide if the answer is "
    'correct. Respond ONLY with JSON: {"correct": true|false}.'
)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9/.]", "", s.lower())


def _numbers(s: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", s)


def grade_exact(item: Item, answer: str) -> bool:
    na = _norm(answer)
    for acc in item.answers:
        nacc = _norm(acc)
        if nacc and nacc in na:
            return True
    # numeric fallback: does any acceptable number appear as a token in the answer?
    ans_nums = set(_numbers(answer))
    for acc in item.answers:
        for x in _numbers(acc):
            if x in ans_nums:
                return True
    return False


async def grade_judge(gw: Gateway, item: Item, answer: str) -> bool:
    comp = await gw.complete(
        GRADER_MODEL,
        [
            {"role": "system", "content": GRADER_SYS},
            {
                "role": "user",
                "content": f"QUESTION:\n{item.prompt}\n\nREFERENCE:\n{item.reference}\n\nANSWER:\n{answer[:4000]}",
            },
        ],
        max_tokens=256,  # grader is a thinking model; needs room past hidden reasoning
    )
    c = comp.content.lower()
    return '"correct": true' in c or '"correct":true' in c or ("true" in c and "false" not in c)


async def grade(gw: Gateway, item: Item, answer: str) -> bool:
    if not answer.strip():
        return False
    if item.type == "exact":
        return grade_exact(item, answer)
    return await grade_judge(gw, item, answer)


async def run_config(engine: Engine, name: str, item: Item, max_tokens: int):
    """Return (answer_text, cost_usd, latency_ms)."""
    msgs = [{"role": "user", "content": item.prompt}]
    modes = {"router-balanced": "balanced", "router-cost": "cost", "router-quality": "quality"}
    if name in modes:  # router config
        rr = await engine.answer(msgs, mode=modes[name], max_tokens=max_tokens, use_cache=False)
        return rr.content, rr.cost, rr.latency_ms
    # any other name is treated as a plain single-model baseline
    c = await engine.gw.complete(name, msgs, max_tokens=max_tokens)
    return c.content, c.cost, c.latency_ms


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(DATASET))
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--configs", type=str, default="claude-opus-4-8,router-balanced,router-cost,router-quality")
    args = ap.parse_args()

    items = DATASET[: args.n]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    engine = Engine()

    # results[config] = list of (item, correct, cost, latency)
    results: dict[str, list] = {c: [] for c in configs}

    print(f"Running {len(items)} items x {len(configs)} configs ({len(items) * len(configs)} evals)...\n")

    for item in items:
        # Run all configs for this item concurrently.
        outs = await asyncio.gather(*(run_config(engine, c, item, args.max_tokens) for c in configs))
        # Grade (judge calls may hit the network) — also concurrent.
        graded = await asyncio.gather(*(grade(engine.gw, item, out[0]) for out in outs))
        line = [f"{item.id:<12} {item.type:<6}"]
        for c, out, ok in zip(configs, outs, graded, strict=True):
            results[c].append((item, ok, out[1], out[2]))
            line.append(f"{c.split('-')[-1][:5]:>6}:{'OK ' if ok else 'XX '}")
        print("  ".join(line))

    # ---- aggregate ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'config':<20}{'acc':>7}{'total$':>11}{'$/correct':>12}{'mean_ms':>10}")
    print("-" * 78)
    rows = []
    for c in configs:
        rs = results[c]
        n_correct = sum(1 for r in rs if r[1])
        acc = n_correct / len(rs) if rs else 0
        total = sum(r[2] for r in rs)
        per_correct = total / n_correct if n_correct else float("inf")
        mean_ms = sum(r[3] for r in rs) / len(rs) if rs else 0
        rows.append((c, acc, total, per_correct, mean_ms))
        print(f"{c:<20}{acc * 100:>6.1f}%{total:>11.5f}{per_correct:>12.5f}{mean_ms:>10.0f}")

    # ---- headline ----------------------------------------------------------
    print("=" * 78)
    base = next((r for r in rows if r[0] == config.PRIMARY_BASELINE), None)
    if base:
        b_acc, b_cost = base[1], base[2]
        print(f"\nBaseline = always {config.PRIMARY_BASELINE}: acc={b_acc * 100:.1f}%, cost=${b_cost:.5f}")
        for c, acc, total, _, _ in rows:
            if c.startswith("router"):
                cost_x = (b_cost / total) if total else float("inf")
                verdict = (
                    "PARETO-WIN"
                    if (acc >= b_acc and total < b_cost)
                    else ("cheaper, lower acc" if total < b_cost else "not cheaper")
                )
                print(
                    f"  {c:<18} acc {acc * 100:5.1f}% "
                    f"({'+' if acc >= b_acc else ''}{(acc - b_acc) * 100:.1f}pt)  "
                    f"cost ${total:.5f}  ({cost_x:.1f}x cheaper)  -> {verdict}"
                )

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
