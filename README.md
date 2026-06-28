# switchboard

[![CI](https://github.com/archit0/switchboard/actions/workflows/ci.yml/badge.svg)](https://github.com/archit0/switchboard/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/switchboard-llm.svg)](https://pypi.org/project/switchboard-llm/)

A **smart, OpenAI-compatible LLM router** that saves cost without losing quality.
More than a proxy — it's a tiny **agent**: a *triage → verify → escalate* loop with
a controller model, a judge, and a Mixture-of-Agents. Point any OpenAI client at it
and it routes each request to the cheapest model that can handle it — easy prompts to
a small model, hard ones to a parallel Mixture-of-Agents — trading a little latency
for large savings while holding (or beating) frontier-model quality on a
representative workload.

The model pool defaults to **auto-updating `-latest` aliases** and is fully
config-driven — point it at new models via `models.json` without touching code (see
[Configuring the model pool](#configuring-the-model-pool)).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
client.chat.completions.create(model="router-cost", messages=[{"role": "user", "content": "..."}])
```

It works on top of **any OpenAI-compatible gateway that fronts multiple providers**
behind one key (e.g. a LiteLLM proxy) — so one client can reach OpenAI, Anthropic,
and Google models just by changing the `model` field. The router is a thin policy
on top of that.

---

## Install

```bash
pip install switchboard-llm        # or: uv add switchboard-llm
```

Configure your gateway (any OpenAI-compatible endpoint):

```bash
export OPENAI_API_KEY=...                  # your gateway key
export OPENAI_BASE_URL=https://.../v1      # your endpoint
```

## Use it

**As a server** (drop-in for any OpenAI client):

```bash
switchboard serve                          # http://localhost:8000/v1  (use --port to change)
```
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
r = client.chat.completions.create(model="router", messages=[{"role": "user", "content": "Hi"}])
print(r.model_extra["switchboard"])        # route, cost, savings telemetry
```

**As a library:**

```python
import asyncio
from switchboard import Engine

async def main():
    eng = Engine()
    rr = await eng.answer([{"role": "user", "content": "What is 17 * 23?"}], mode="cost")
    print(rr.content, f"${rr.cost:.6f}", f"{rr.savings_pct:.0f}% cheaper than Opus")
    await eng.aclose()

asyncio.run(main())
```

**From the CLI:**

```bash
switchboard ask "Prove sqrt(2) is irrational" --mode quality
switchboard models                         # probe which gateway models are actually live
```

---

## The honest thesis (read this first)

The goal is a router that is **cheaper than a frontier model (e.g. Opus) and
matches-or-beats it on benchmarks**. That is achievable — but only as a
**portfolio result over a realistic workload**, not a per-query miracle. The iron
law:

> On a *single hard query*, you cannot both beat the frontier model **and** be
> cheaper than it on that same query.

What you *can* do, and what this does:

| Traffic | What the router does | Outcome |
|---|---|---|
| **Easy queries** (most real traffic) | route to a cheap model | quality ties Opus, **5–50× cheaper** |
| **Hard queries** (the minority) | **Mixture-of-Agents**: several cheap/mid models answer in parallel, a synthesizer fuses them | quality can **match or exceed** a single Opus call, still **< Opus cost** |
| **Repeats** | exact-match cache | **free** |

Averaged over the workload, total spend is well below always-Opus and mean
accuracy is **equal-or-better**. Grounded in **RouteLLM**, **FrugalGPT** (cascade
with a judge), and **Mixture-of-Agents**.

---

## Modes

Pick the strategy via the `model` field:

| `model` | strategy |
|---|---|
| `router` / `router-balanced` | triage → single cheap (easy) / single mid (moderate) / Mixture-of-Agents (hard) |
| `router-cost` | **FrugalGPT cascade** — answer cheap, a judge scores it, escalate only if low |
| `router-quality` | bias one tier up — best quality while staying under Opus cost |

Any **real** model id (`claude-opus-4-8`, `gpt-5.5`, …) passes straight through, so
this also works as a plain multi-provider proxy.

## How it works

```
request ─► [cache] ─► [triage: how hard?] ─► [policy] ──► single cheap model      (easy)
                                                      └─► single mid model        (moderate)
                                                      └─► Mixture-of-Agents        (hard)
                                                            proposers ∥ ─► synthesizer
```

- **Triage** (`src/switchboard/classify.py`) — free heuristics (length, code/math
  markers, multi-step verbs) decide obvious cases; a tiny LLM classifier scores the
  ambiguous middle. Output: difficulty 1–5 → tier.
- **Policy / execution** (`src/switchboard/engine.py`) — `single`, `moa` (parallel
  proposers + synthesizer), or `cascade` (cheap → judge → escalate).
- **Cost accounting** — every response carries its internal cost, an estimate of
  what always-Opus would have cost, and the savings %, under a `switchboard` key.

## Configuring the model pool

The pool is **config-driven** so it never goes stale. Defaults use auto-updating
`-latest` aliases for the cheap/mid tiers (and the latest flagships for the strong
tier / baseline). To point switchboard at different models **without touching code**,
drop a `models.json` in the repo root (see `models.json.example`) — any key overrides
the default:

```json
{
  "default_cheap": "gemini-flash-lite-latest",
  "default_mid":   "gemini-flash-latest",
  "strong":        ["claude-opus-4-8", "gpt-5.5", "gemini-3.1-pro-preview"],
  "moa_proposers": ["gemini-flash-latest", "gpt-5.4-mini", "claude-haiku-4-5"],
  "moa_synthesizer": "claude-sonnet-4-6",
  "classifier": "gemini-flash-lite-latest",
  "judge": "gemini-flash-lite-latest"
}
```

Pricing works the same way via `pricing.json` (`{"model": [in_per_1M, out_per_1M]}`).
After changing the pool, verify everything is live with `switchboard models` — the
gateway's `/v1/models` list is stale and can advertise ids that 404.

---

## Results

On **GSM8K (50 items, exact numeric grading)**, baseline = always `claude-opus-4-8`:

| config | accuracy | total cost | vs Opus |
|---|---|---|---|
| always-Opus | 100.0% | $0.3674 | baseline |
| `router-cost` | **100.0%** | $0.0064 | **57× cheaper — Pareto win** |
| `router-quality` | 100.0% | $0.2781 | 1.3× cheaper |
| `router-balanced` | 92.0% | $0.0611 | 6× cheaper but lost accuracy |

Reproduce: `python -m bench.run_gsm8k --n 50 --seed 0`. Full write-up and honest
caveats in [`RESULTS.md`](RESULTS.md). (The verifier is what makes routing safe —
`router-balanced` has none and lost 8 points; `router-cost`'s judge is the fix.)

### Agentic benchmark (tool-using agent)

A minimal ReAct **tool-using agent** (tools: `calc`, `wordcount`, `lookup`) solving 4
tasks of increasing size, driven by each backend through the *same* harness. All
backends hit 100% on these deterministic tasks, so the story is **tokens vs. cost**:

| backend | accuracy | total tokens | total cost |
|---|---|---|---|
| `claude-opus-4-8` (single) | 100% | 5,885 | $0.10202 |
| `gpt-5.5` (single) | 100% | 4,846 | $0.01335 |
| `claude-haiku-4-5` (single) | 100% | 3,646 | $0.00577 |
| `gemini-flash-latest` (single) | 100% | 4,645 | $0.00289 |
| `router-quality` | 100% | 6,205 | $0.01073 |
| `router-cost` | 100% | 10,769 | $0.01264 |
| **`router-balanced`** | **100%** | 4,472 | **$0.00153** |

`router-balanced` is **~9× cheaper than the GPT-5.5 agent and ~67× cheaper than the
Opus agent** at the same success rate. Note the twist: `router-cost` spends the *most*
tokens here — its verify-every-step cascade is ideal for one-shot calls but the judge
overhead compounds across agent turns. Full tables + per-task breakdown in
[`AGENT_BENCH.md`](AGENT_BENCH.md). Reproduce: `python -m bench.agent_bench`.

---

## Limitations & next steps

- **Pricing is a list-price proxy** (`src/switchboard/config.py`). Drop your real
  rate card into `pricing.json` (`{"model": [in_per_1M, out_per_1M]}`) to override.
- **Triage under-detects "deceptively simple" trap questions** — `router-cost`/
  `router-quality` compensate via the judge/MoA.
- **Streaming is simulated** (full answer computed, then chunked) — MoA can't
  token-stream; only the single-model path could truly stream.
- **Semantic cache** (embed prompt → nearest neighbour) is not yet wired.
- **The gateway's `/v1/models` list may be stale** — trust `switchboard models`.

## License

MIT — see [LICENSE](LICENSE).
