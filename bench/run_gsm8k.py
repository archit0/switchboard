"""Run the router against GSM8K — the canonical grade-school-math benchmark.

Standard protocol: zero-shot chain-of-thought, exact numeric answer match
(gold is the number after '####' in the official answer). No LLM judge, so the
accuracy numbers are clean and reproducible.

This compares the router modes against an always-Opus baseline on the SAME
fixed-seed subset, so it is apples-to-apples. (Absolute scores won't exactly
match published leaderboard numbers — those use specific few-shot/eval-harness
setups — but the *relative* router-vs-Opus comparison is what proves the
cost/quality thesis.)

    .venv/bin/python -m bench.run_gsm8k --n 50 --seed 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import urllib.request

from switchboard import config
from switchboard.engine import Engine

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "gsm8k_test.jsonl")

SYS = (
    "Solve the math word problem. Show concise step-by-step reasoning. "
    "On the FINAL line, output exactly '#### <answer>' where <answer> is just "
    "the final number (no units, no commas, no text)."
)

_NUM = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_HASH = re.compile(r"####\s*(-?\$?[\d,]+(?:\.\d+)?)")


def _to_num(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gold_of(answer: str) -> float | None:
    m = _HASH.search(answer)
    return _to_num(m.group(1)) if m else None


def pred_of(text: str) -> float | None:
    m = _HASH.findall(text)
    if m:
        return _to_num(m[-1])
    m = re.findall(r"(?:final answer|answer is|=)\D{0,8}(-?\$?\d[\d,]*(?:\.\d+)?)", text, re.I)
    if m:
        return _to_num(m[-1])
    nums = _NUM.findall(text)
    return _to_num(nums[-1]) if nums else None


def correct(pred: float | None, gold: float | None) -> bool:
    return pred is not None and gold is not None and abs(pred - gold) < 1e-4


def ensure_data() -> list[dict]:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading GSM8K test set -> {DATA_PATH}")
        urllib.request.urlretrieve(GSM8K_URL, DATA_PATH)
    with open(DATA_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_one(engine: Engine, cfg: str, question: str, max_tokens: int):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    modes = {"router-balanced": "balanced", "router-cost": "cost", "router-quality": "quality"}
    if cfg in modes:
        rr = await engine.answer(msgs, mode=modes[cfg], max_tokens=max_tokens, use_cache=False)
        return rr.content, rr.cost, rr.latency_ms
    c = await engine.gw.complete(cfg, msgs, max_tokens=max_tokens)
    return c.content, c.cost, c.latency_ms


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--configs", type=str, default="claude-opus-4-8,router-balanced,router-cost,router-quality")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    rows = ensure_data()
    random.seed(args.seed)
    sample = random.sample(rows, min(args.n, len(rows)))
    items = [(r["question"], gold_of(r["answer"])) for r in sample]

    engine = Engine()
    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, list] = {c: [] for c in configs}
    done = {"n": 0}

    print(f"GSM8K: {len(items)} items (seed={args.seed}) x {len(configs)} configs\n")

    async def eval_item(q: str, gold: float | None):
        async with sem:
            outs = await asyncio.gather(*(run_one(engine, c, q, args.max_tokens) for c in configs))
        flags = []
        for c, (ans, cost, ms) in zip(configs, outs, strict=True):
            ok = correct(pred_of(ans), gold)
            results[c].append((ok, cost, ms))
            flags.append(f"{c.split('-')[-1][:5]:>5}:{'OK' if ok else 'XX'}")
        done["n"] += 1
        print(f"[{done['n']:>3}/{len(items)}] gold={str(gold):>8}  " + "  ".join(flags))

    await asyncio.gather(*(eval_item(q, g) for q, g in items))

    # ---- aggregate ---------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'config':<20}{'acc':>8}{'total$':>11}{'$/correct':>12}{'mean_ms':>10}")
    print("-" * 78)
    rows_out = []
    for c in configs:
        rs = results[c]
        nc = sum(1 for r in rs if r[0])
        acc = nc / len(rs) if rs else 0.0
        total = sum(r[1] for r in rs)
        per = total / nc if nc else float("inf")
        mean_ms = sum(r[2] for r in rs) / len(rs) if rs else 0.0
        rows_out.append((c, acc, total, mean_ms))
        print(f"{c:<20}{acc * 100:>7.1f}%{total:>11.5f}{per:>12.6f}{mean_ms:>10.0f}")
    print("=" * 78)

    base = next((r for r in rows_out if r[0] == config.PRIMARY_BASELINE), None)
    if base:
        b_acc, b_cost = base[1], base[2]
        print(f"\nBaseline = always {config.PRIMARY_BASELINE}: acc={b_acc * 100:.1f}%, cost=${b_cost:.5f}")
        for c, acc, total, _ in rows_out:
            if c.startswith("router"):
                x = (b_cost / total) if total else float("inf")
                verdict = (
                    "PARETO-WIN"
                    if (acc >= b_acc - 1e-9 and total < b_cost)
                    else "cheaper, lower acc"
                    if total < b_cost
                    else "not cheaper"
                )
                print(
                    f"  {c:<18} acc {acc * 100:5.1f}% "
                    f"({'+' if acc >= b_acc else ''}{(acc - b_acc) * 100:.1f}pt)  "
                    f"cost ${total:.5f}  ({x:.1f}x cheaper)  -> {verdict}"
                )

    await engine.aclose()


if __name__ == "__main__":
    asyncio.run(main())
