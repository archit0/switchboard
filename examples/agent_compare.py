"""A tiny agent, one task, three router settings — compared by a simple LLM.

  Agent:    a one-shot "careful reasoning agent" (a system prompt + the router).
  Task:     a classic trap question with a known correct answer.
  Compare:  run the agent under router-balanced / router-cost / router-quality,
            then have a small, cheap LLM judge each output against the expected
            answer and pick the best.

Run:  uv run python examples/agent_compare.py
"""
from __future__ import annotations

import asyncio
import json
import re

from switchboard.engine import Engine

# --- the agent -------------------------------------------------------------- #
AGENT_SYSTEM = (
    "You are a careful reasoning agent. Think before answering. "
    "End with a line: 'Final answer: <answer>'."
)

# --- the task (a trap: the naive answer is 100 cats; the correct answer is 3) -#
TASK = (
    "If 3 cats can catch 3 mice in 3 minutes, "
    "how many cats are needed to catch 100 mice in 100 minutes?"
)
EXPECTED = "3 cats (each cat catches 1 mouse per 3 minutes; in 100 minutes one cat catches ~33, so 3 cats suffice)"

MODES = ["balanced", "cost", "quality"]

# --- the judge: a deliberately simple / cheap LLM --------------------------- #
JUDGE_MODEL = "gemini-3.1-flash-lite"
JUDGE_SYS = (
    "You are a grader comparing answers from three AI configurations to a known "
    "correct answer. For each configuration, decide if its final answer matches the "
    "expected answer, with a short note. Then pick the best one overall.\n"
    "Respond ONLY with JSON:\n"
    '{"balanced": {"correct": true|false, "note": "..."}, '
    '"cost": {"correct": true|false, "note": "..."}, '
    '"quality": {"correct": true|false, "note": "..."}, '
    '"best": "balanced|cost|quality", "summary": "one sentence"}'
)


async def main() -> None:
    eng = Engine()
    print(f"AGENT:    careful reasoning agent\nTASK:     {TASK}\nEXPECTED: {EXPECTED}\n")

    # 1) run the same agent under each router setting
    results = {}
    for mode in MODES:
        rr = await eng.answer(
            [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": TASK}],
            mode=mode,
            max_tokens=500,
            use_cache=False,
        )
        results[mode] = rr
        print(f"--- router-{mode} ---")
        print(rr.content.strip())
        print(f"[route={rr.route} | cost=${rr.cost:.6f} | "
              f"savings={rr.savings_pct:.0f}% vs Opus | {rr.latency_ms:.0f}ms]\n")

    # 2) a simple LLM judges the three outputs against the expected answer
    bundle = "\n\n".join(
        f"### Configuration: {m}\n{results[m].content.strip()}" for m in MODES
    )
    judge = await eng.gw.complete(
        JUDGE_MODEL,
        [
            {"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content":
                f"TASK:\n{TASK}\n\nEXPECTED ANSWER:\n{EXPECTED}\n\nOUTPUTS:\n{bundle}"},
        ],
        max_tokens=400,
    )

    print(f"=== JUDGE ({JUDGE_MODEL}) ===")
    verdict = None
    try:
        m = re.search(r"\{.*\}", judge.content, re.S)
        verdict = json.loads(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        verdict = None

    if verdict:
        for mode in MODES:
            v = verdict.get(mode, {})
            mark = "✓ CORRECT" if v.get("correct") else "✗ WRONG"
            rr = results[mode]
            print(f"  router-{mode:<9} {mark:<10} ${rr.cost:.6f}  — {v.get('note', '')}")
        print(f"\n  Best per judge: router-{verdict.get('best', '?')}")
        print(f"  Summary: {verdict.get('summary', '')}")
    else:
        print(judge.content.strip())

    # 3) the punchline: cost vs. correctness
    print("\n=== takeaway ===")
    cheapest_correct = None
    if verdict:
        correct_modes = [m for m in MODES if verdict.get(m, {}).get("correct")]
        if correct_modes:
            cheapest_correct = min(correct_modes, key=lambda m: results[m].cost)
            rr = results[cheapest_correct]
            print(f"  Cheapest correct setting: router-{cheapest_correct} "
                  f"(${rr.cost:.6f}, {rr.savings_pct:.0f}% cheaper than always-Opus).")
        else:
            print("  No setting matched the expected answer on this run.")
    await eng.aclose()


if __name__ == "__main__":
    asyncio.run(main())
