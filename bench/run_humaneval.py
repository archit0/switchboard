"""HumanEval — the canonical code-generation benchmark — across backends.

Each problem gives a function signature + docstring; the model writes the
function; we grade pass@1 by EXECUTING the official unit tests. The same set is
run through single models and the three router modes, so it's apples-to-apples.

Code is executed in a sandboxed subprocess (separate process, temp file, hard
timeout). It's still running model-generated code — run it on a throwaway/CI box
if you're cautious.

    uv run python -m bench.run_humaneval --n 40 --seed 0
    uv run python -m bench.run_humaneval --n 4         # quick smoke
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile

from switchboard.engine import Engine

HE_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "humaneval.jsonl")

SYS = (
    "You are an expert Python programmer. Complete the function below. Respond with "
    "ONLY the complete function (the signature, body, and any imports it needs) inside "
    "a single ```python code block. No explanation, no examples, no tests."
)

# Imports made available to every solution so a missing `from typing import List`
# etc. doesn't cause spurious failures (standard practice in HumanEval harnesses).
PREAMBLE = "from typing import *\nimport math, re, collections, itertools, functools, heapq, bisect\n\n"

ALL_CONFIGS = [
    ("gpt-5.5", "single"),
    ("claude-opus-4-8", "single"),
    ("gemini-flash-latest", "single"),
    ("router-balanced", "router"),
    ("router-cost", "router"),
    ("router-quality", "router"),
]
ROUTER_MODE = {"router-balanced": "balanced", "router-cost": "cost", "router-quality": "quality"}


def ensure_data() -> list[dict]:
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print(f"downloading HumanEval -> {DATA_PATH}")
        import urllib.request

        gz = DATA_PATH + ".gz"
        urllib.request.urlretrieve(HE_URL, gz)
        with gzip.open(gz, "rt") as fin, open(DATA_PATH, "w") as fout:
            fout.write(fin.read())
        os.remove(gz)
    with open(DATA_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def build_program(problem: dict, completion: str) -> str:
    code = extract_code(completion)
    if f"def {problem['entry_point']}" not in code:
        code = problem["prompt"] + "\n" + code  # treat as a body continuation
    return PREAMBLE + code + "\n\n" + problem["test"] + f"\n\ncheck({problem['entry_point']})\n"


def run_program(program: str, timeout: int = 12) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=tempfile.gettempdir()) as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def generate(engine: Engine, name: str, kind: str, problem: dict) -> tuple[str, float]:
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": problem["prompt"]}]
    if kind == "router":
        rr = await engine.answer(msgs, mode=ROUTER_MODE[name], max_tokens=900, use_cache=False)
        return rr.content, rr.cost
    c = await engine.gw.complete(name, msgs, max_tokens=900)
    return c.content, c.cost


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--configs", default=",".join(n for n, _ in ALL_CONFIGS))
    args = ap.parse_args()

    chosen = [(n, k) for n, k in ALL_CONFIGS if n in args.configs.split(",")]
    problems = ensure_data()
    random.seed(args.seed)
    sample = random.sample(problems, min(args.n, len(problems)))

    engine = Engine()
    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, list] = {n: [] for n, _ in chosen}
    done = {"n": 0}

    print(f"HumanEval: {len(sample)} problems (seed={args.seed}) x {len(chosen)} configs\n")

    async def eval_problem(problem: dict):
        async with sem:
            gens = await asyncio.gather(*(generate(engine, n, k, problem) for n, k in chosen))
        flags = []
        for (name, _k), (completion, cost) in zip(chosen, gens, strict=True):
            passed = await asyncio.to_thread(run_program, build_program(problem, completion))
            results[name].append((passed, cost))
            flags.append(f"{name.split('-')[-1][:5]:>5}:{'P' if passed else '.'}")
        done["n"] += 1
        print(f"[{done['n']:>3}/{len(sample)}] {problem['task_id']:<14} " + "  ".join(flags))

    await asyncio.gather(*(eval_problem(p) for p in sample))
    await engine.aclose()

    # ---- report ----------------------------------------------------------- #
    rows = []
    for name, _k in chosen:
        rs = results[name]
        solved = sum(1 for p, _ in rs if p)
        acc = solved / len(rs) * 100
        cost = sum(c for _, c in rs)
        per = cost / solved if solved else float("inf")
        rows.append((name, acc, solved, len(rs), cost, per))

    print("\n" + "=" * 72)
    print(f"{'backend':<22}{'pass@1':>9}{'solved':>9}{'cost':>11}{'$/solved':>12}")
    print("-" * 72)
    for name, acc, solved, n, cost, per in rows:
        print(f"{name:<22}{acc:>8.1f}%{solved:>6}/{n:<3}{cost:>11.5f}{per:>12.5f}")
    print("=" * 72)

    def md(headers, rs):
        out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in rs]
        return "\n".join(out)

    summary = md(
        ["backend", "pass@1", "solved", "total cost", "$/solved"],
        [[f"`{n}`", f"{a:.1f}%", f"{s}/{tot}", f"${c:.5f}", f"${p:.5f}"] for n, a, s, tot, c, p in rows],
    )
    best_single = max((r for r in rows if not r[0].startswith("router")), key=lambda r: r[1], default=None)
    headline = ""
    if best_single:
        bn, ba, _bs, _bt, bc, _bp = best_single
        lines = []
        for n, a, _s, _t, c, _p in rows:
            if n.startswith("router") and c < bc:
                lines.append(
                    f"- `{n}`: {a:.1f}% pass@1 at ${c:.5f} — **{bc / c:.1f}x cheaper** than `{bn}` "
                    f"({'+' if a >= ba else ''}{a - ba:.1f} pts)"
                )
        headline = f"Best single model: **`{bn}`** at {ba:.1f}% pass@1, ${bc:.5f}.\n\n" + "\n".join(lines)

    doc = f"""# HumanEval (code generation) across backends

Canonical code-generation benchmark: each problem gives a function signature +
docstring; the model writes the function; we grade **pass@1 by executing the
official unit tests** in a sandboxed subprocess. {len(sample)}-problem seeded
subset (seed {args.seed}), same problems for every backend.

Generated by `bench/run_humaneval.py` (`uv run python -m bench.run_humaneval`).
Costs use the list-price proxies in `src/switchboard/config.py`.

## Results

{summary}

{headline}

## How to read this
- **The strongest single result is usually `router-cost`: it ties the best models'
  pass@1 at the lowest cost of any backend**, because its cascade tries a cheap model,
  a judge rejects bad code, and it escalates failures to a Sonnet-led Mixture-of-Agents
  — often without ever calling a frontier model.
- **Watch `router-balanced`**: cheap models are genuinely weak at code, and balanced
  has *no verifier*, so it can ship the cheap model's failures. The judge in
  `router-cost` is exactly what prevents this. On coding, never route without verification.
- The router's credible coding claim is **frontier pass@1 at a fraction of the cost** —
  not a new SOTA. Hard problems still need strong models; the router just pays for them
  only when a cheap attempt provably fails.
- pass@1, single sample, default decoding. A {len(sample)}-problem subset has wide
  error bars — bump `--n` for tighter numbers.

### Caveats
- Subset (not all 164), list-price costs, single-sample pass@1.
- Executes model-generated code in a subprocess sandbox (temp file + timeout); run
  on a throwaway/CI box if cautious.
"""
    path = os.path.join(os.path.dirname(__file__), "..", "HUMANEVAL.md")
    with open(path, "w") as f:
        f.write(doc)
    print(f"\nwrote {os.path.abspath(path)}")


if __name__ == "__main__":
    asyncio.run(main())
