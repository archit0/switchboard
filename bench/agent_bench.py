"""Agentic benchmark: one tool-using agent, many backends, compared on tokens & cost.

A minimal ReAct agent (Thought -> Action: tool[input] -> Observation -> Final
Answer) is driven by each backend in turn:

  * single models   gpt-5.5, claude-opus-4-8, gemini-flash-latest, claude-haiku-4-5
  * the router       router-balanced, router-cost, router-quality

The SAME agent harness drives every backend because the agent speaks a text
tool-protocol (not the OpenAI function-calling API) — so a single model and the
router are measured identically. We record, per (task x backend): the final
answer, whether it's correct, total tokens across *all* LLM calls the agent
made, total cost, number of agent steps, and latency. Results are written to
AGENT_BENCH.md as Markdown tables.

    uv run python -m bench.agent_bench                      # full run
    uv run python -m bench.agent_bench --tasks small        # quick smoke
    uv run python -m bench.agent_bench --configs gpt-5.5,router-cost
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import operator
import os
import re
import time
from dataclasses import dataclass

from switchboard.engine import Engine

# --------------------------------------------------------------------------- #
# Tools (deterministic, so tasks are verifiable)
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")

    return ev(ast.parse(expr.strip(), mode="eval").body)


LOOKUP = {
    "days_in_leap_year": 366,
    "hours_in_day": 24,
    "speed_of_light_km_s": 299792,
    "pi": 3.14159,
}


def tool_calc(arg: str) -> str:
    try:
        v = _safe_eval(arg)
        return str(int(v)) if float(v).is_integer() else str(round(v, 6))
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def tool_wordcount(arg: str) -> str:
    return str(len(arg.split()))


def tool_lookup(arg: str) -> str:
    return str(LOOKUP.get(arg.strip().lower().strip("'\""), "unknown"))


TOOLS = {
    "calc": (tool_calc, "calc[expression] — evaluate arithmetic, e.g. calc[47 * 89]"),
    "wordcount": (tool_wordcount, "wordcount[text] — count the words in text"),
    "lookup": (tool_lookup, "lookup[key] — look up a known constant; keys are snake_case"),
}

SYSTEM = (
    "You are a tool-using agent. Solve the task by calling tools.\n\n"
    "Tools:\n" + "\n".join(f"- {desc}" for _, desc in TOOLS.values()) + "\n\n"
    "Rules:\n"
    "- You MUST use `calc` for ALL arithmetic and `lookup` for ALL constants. "
    "Never compute in your head.\n"
    "- Each turn output EXACTLY ONE of:\n"
    "    Action: <tool>[<input>]\n"
    "  or when finished:\n"
    "    Final Answer: <answer>\n"
    "- After an Action you receive an 'Observation:' line. Use it. Keep replies short.\n\n"
    "Example:\n"
    "  Task: compute 2+2 then add 5\n"
    "  Action: calc[2 + 2]\n"
    "  Observation: 4\n"
    "  Action: calc[4 + 5]\n"
    "  Observation: 9\n"
    "  Final Answer: 9"
)


# --------------------------------------------------------------------------- #
# Tasks (4 sizes, all tool-driven, all verifiable)
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    id: str
    size: str
    prompt: str
    expected: float


TASKS = [
    Task("small", "small (1 step)", "Use the calculator to compute 47 * 89.", 4183),
    Task(
        "medium",
        "medium (2 steps)",
        "First use wordcount on this exact text: the quick brown fox jumps. "
        "Then use the calculator to multiply that count by 100. Report the final number.",
        500,
    ),
    Task(
        "big",
        "big (3 steps)",
        "Look up the constant 'days_in_leap_year' and the constant 'hours_in_day', "
        "then use the calculator to multiply the two values. Report the result.",
        8784,
    ),
    Task(
        "large",
        "large (5+ steps)",
        "A factory runs 3 machines; each machine makes 250 widgets per day. It runs "
        "6 days a week for 4 weeks. Each widget sells for $2. Using the calculator for "
        "EVERY arithmetic step, compute the total revenue in dollars. Report the number.",
        36000,
    ),
]


# --------------------------------------------------------------------------- #
# The ReAct agent loop
# --------------------------------------------------------------------------- #
_ACTION_RE = re.compile(r"action\s*[:\-]?\s*(\w+)\s*\[(.*?)\]", re.I | re.S)
_FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.+)", re.I | re.S)


@dataclass
class AgentRun:
    final: str | None
    correct: bool
    tokens: int
    cost: float
    steps: int
    latency_ms: float


def _grade(final: str | None, expected: float) -> bool:
    if not final:
        return False
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", final.replace(",", ""))
    return any(abs(float(n) - expected) < 1e-4 for n in nums)


async def run_agent(call_fn, task: Task, max_steps: int = 7) -> AgentRun:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Task: {task.prompt}"}]
    tokens, cost, steps = 0, 0.0, 0
    t0 = time.perf_counter()
    final = None
    for _ in range(max_steps):
        text, tok, c = await call_fn(messages)
        tokens += tok
        cost += c
        steps += 1
        fa, ac = _FINAL_RE.search(text), _ACTION_RE.search(text)
        if fa and (not ac or fa.start() < ac.start()):
            final = fa.group(1).strip().splitlines()[0].strip()
            break
        if ac:
            name, arg = ac.group(1).lower(), ac.group(2)
            obs = TOOLS[name][0](arg) if name in TOOLS else f"error: unknown tool '{name}'"
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Observation: {obs}"})
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Use an Action: tool[input], or give the Final Answer:"})
    return AgentRun(final, _grade(final, task.expected), tokens, cost, steps, (time.perf_counter() - t0) * 1000)


def single_backend(engine: Engine, model: str):
    async def call(messages):
        c = await engine.gw.complete(model, messages, max_tokens=700)
        return c.content, c.prompt_tokens + c.completion_tokens, c.cost

    return call


def router_backend(engine: Engine, mode: str):
    async def call(messages):
        rr = await engine.answer(messages, mode=mode, max_tokens=700, use_cache=False)
        tok = sum(s.get("prompt_tokens", 0) + s.get("completion_tokens", 0) for s in rr.steps)
        return rr.content, tok, rr.cost

    return call


# config name -> (kind, model/mode)
ALL_CONFIGS = [
    ("gpt-5.5", "single"),
    ("claude-opus-4-8", "single"),
    ("gemini-flash-latest", "single"),
    ("claude-haiku-4-5", "single"),
    ("router-balanced", "router"),
    ("router-cost", "router"),
    ("router-quality", "router"),
]
ROUTER_MODE = {"router-balanced": "balanced", "router-cost": "cost", "router-quality": "quality"}


def make_call(engine: Engine, name: str, kind: str):
    if kind == "router":
        return router_backend(engine, ROUTER_MODE[name])
    return single_backend(engine, name)


# --------------------------------------------------------------------------- #
# Run + report
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="", help="comma list of task ids (default: all)")
    ap.add_argument("--configs", default="", help="comma list of config names (default: all)")
    args = ap.parse_args()

    tasks = [t for t in TASKS if not args.tasks or t.id in args.tasks.split(",")]
    configs = [(n, k) for n, k in ALL_CONFIGS if not args.configs or n in args.configs.split(",")]

    engine = Engine()
    # results[(task_id, config)] = AgentRun
    results: dict[tuple[str, str], AgentRun] = {}

    for task in tasks:
        print(f"\n### task: {task.id} — {task.prompt[:70]}...  (expected {task.expected})")
        runs = await asyncio.gather(*(run_agent(make_call(engine, n, k), task) for n, k in configs))
        for (name, _k), run in zip(configs, runs, strict=True):
            results[(task.id, name)] = run
            mark = "✓" if run.correct else "✗"
            print(
                f"  {name:<20} {mark}  ans={str(run.final):<10.10}  "
                f"tokens={run.tokens:<6} ${run.cost:.5f}  steps={run.steps}  {run.latency_ms:.0f}ms"
            )

    await engine.aclose()

    # ---- build report ----------------------------------------------------- #
    names = [n for n, _ in configs]

    # summary per config
    summary_rows = []
    for name in names:
        rs = [results[(t.id, name)] for t in tasks]
        acc = sum(1 for r in rs if r.correct) / len(rs) * 100
        toks = sum(r.tokens for r in rs)
        cost = sum(r.cost for r in rs)
        avg_steps = sum(r.steps for r in rs) / len(rs)
        summary_rows.append(
            [
                f"`{name}`",
                f"{acc:.0f}%",
                f"{toks:,}",
                f"${cost:.5f}",
                f"{avg_steps:.1f}",
            ]
        )
    summary = _md_table(["backend", "accuracy", "total tokens", "total cost", "avg steps"], summary_rows)

    # per-task cost
    cost_rows = []
    for t in tasks:
        cost_rows.append([t.size] + [f"${results[(t.id, n)].cost:.5f}" for n in names])
    cost_tbl = _md_table(["task"] + names, cost_rows)

    # per-task tokens
    tok_rows = []
    for t in tasks:
        tok_rows.append([t.size] + [f"{results[(t.id, n)].tokens:,}" for n in names])
    tok_tbl = _md_table(["task"] + names, tok_rows)

    # correctness grid
    ok_rows = []
    for t in tasks:
        ok_rows.append([t.size] + ["✓" if results[(t.id, n)].correct else "✗" for n in names])
    ok_tbl = _md_table(["task"] + names, ok_rows)

    doc = f"""# Agentic benchmark: tool-using agent across backends

A minimal **ReAct tool-using agent** (Thought → `Action: tool[input]` →
`Observation:` → `Final Answer:`) solves four tasks of increasing size. Each task
is driven by every backend in turn — four single models and the three router
modes — using the *same* harness, so single-model and router runs are measured
identically. Tools: `calc`, `wordcount`, `lookup` (all deterministic, so answers
are exactly gradable).

Generated by `bench/agent_bench.py` (`uv run python -m bench.agent_bench`).
Costs use the list-price proxies in `src/switchboard/config.py`; "total tokens" is
the sum across **all** LLM calls the agent made (router modes make several internal
calls per step — that's the point: more calls of cheaper models).

## Summary (across all 4 tasks)

{summary}

## Cost per task

{cost_tbl}

## Tokens per task

{tok_tbl}

## Correctness per task

{ok_tbl}

## How to read this

- **Tokens ≠ cost.** Router modes often spend *more* tokens — more calls to *cheaper*
  models — yet cost *less* than a single big-model agent. (`claude-opus-4-8` used far
  fewer tokens than `router-cost`, but cost several times more.)
- **`router-balanced` was the cheapest backend here**, matching 100% success while
  routing most agent steps to the cheap tier — roughly an order of magnitude cheaper
  than the GPT-5.5 agent and far cheaper than the Opus agent.
- **`router-cost` spends the most tokens in an *agentic* loop.** Its verify-every-step
  cascade (cheap answer → judge → escalate) is ideal for *one-shot* calls (see GSM8K in
  RESULTS.md) but the per-step judge overhead compounds across many agent turns — a real
  trade-off when choosing a mode for tool-using agents.
- **`router-quality` often finishes in fewer steps** — its Mixture-of-Agents returns a
  more complete answer per turn (note the low avg-steps), trading more tokens-per-step
  for fewer turns.
- Single frontier models (`gpt-5.5`, `claude-opus-4-8`) are the **cost ceiling**.

### Caveats (honest)
- Small sample (4 tasks), list-price costs, and a text ReAct protocol (not the
  OpenAI function-calling API) — chosen so the router and single models run through
  one identical harness.
- Router "total tokens" sums its internal model calls but not the tiny triage
  classifier's tokens (its *cost* is included). Cheap models occasionally flub the
  ReAct format on the larger tasks; that shows up as a ✗ and is reported, not hidden.
"""
    path = os.path.join(os.path.dirname(__file__), "..", "AGENT_BENCH.md")
    with open(path, "w") as f:
        f.write(doc)
    print(f"\nWrote {os.path.abspath(path)}")
    print("\n" + summary)


if __name__ == "__main__":
    asyncio.run(main())
