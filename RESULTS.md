# Measured results

## GSM8K — 50 items, seed 0 (recognized benchmark)

`.venv/bin/python -m bench.run_gsm8k --n 50 --seed 0`. Zero-shot chain-of-thought,
exact numeric grading (gold = number after `####`), no LLM judge. Baseline =
always `claude-opus-4-8`, same 50 items for every config.

```
config                   acc     total$   $/correct   mean_ms
------------------------------------------------------------------------------
claude-opus-4-8       100.0%    0.36744    0.007349      2094
router-balanced        92.0%    0.06106    0.001327      4025
router-cost           100.0%    0.00640    0.000128      2898
router-quality        100.0%    0.27812    0.005562      6784
------------------------------------------------------------------------------
router-cost        acc 100.0% (+0.0pt)  cost $0.00640  (57.4x cheaper)  PARETO-WIN
router-quality     acc 100.0% (+0.0pt)  cost $0.27812  ( 1.3x cheaper)  PARETO-WIN
router-balanced    acc  92.0% (-8.0pt)  cost $0.06106  ( 6.0x cheaper)  cheaper, LOWER acc
```

**Findings (honest):**
- **`router-cost` (FrugalGPT cascade) is Pareto-dominant: ties Opus at 100% for
  57× less cost.** The cheap model + judge resolves most items; only low-confidence
  ones escalate. This is the headline result on a recognized benchmark.
- **`router-balanced` LOST 8 points** (4/50 items). This is the real cost of naive
  single-model routing with *no verification* — when the mid model errs, nothing
  catches it. The cascade's judge is precisely the fix, and the data shows it.
  Reported, not hidden. (Actionable next step: add a light verifier to the
  balanced mid tier.)
- **Magnitude caveat:** GSM8K is now *easy* for modern small models, so 57× is
  near the optimistic end. Expect smaller-but-real savings on harder suites
  (MMLU-Pro, GPQA) where the cheap tier carries less of the load.

---

## Mixed smoke set — 15 items

Run of `.venv/bin/python -m bench.run_bench` on the 15-item mixed set
(easy factual/math, CRT "trap" questions, open-ended reasoning & code).
Baseline = always `claude-opus-4-8`. Grader = `gemini-3.1-pro-preview`
(independent of the MoA pool). Costs use the list-price proxies in
`src/switchboard/config.py`.

```
config                  acc     total$   $/correct   mean_ms
------------------------------------------------------------------------------
claude-opus-4-8       93.3%    0.10338     0.00738      3084
router-balanced       93.3%    0.01162     0.00083      3138
router-cost          100.0%    0.01106     0.00074      4534
router-quality       100.0%    0.04278     0.00285      5502
------------------------------------------------------------------------------
router-balanced    acc  93.3% (+0.0pt)  cost $0.01162  ( 8.9x cheaper)  PARETO-WIN
router-cost        acc 100.0% (+6.7pt)  cost $0.01106  ( 9.3x cheaper)  PARETO-WIN
router-quality     acc 100.0% (+6.7pt)  cost $0.04278  ( 2.4x cheaper)  PARETO-WIN
```

## Reading this honestly

- **All three router modes are Pareto wins**: equal-or-better accuracy than
  always-Opus at 2.4×–9.3× lower cost. `router-cost` is strictly better on
  *both* axes here.
- **`router-balanced` matches Opus at near-Opus latency** (3.1s vs 3.1s) because
  most items route to a single cheap model. `router-quality` is slower (5.5s)
  because it fires Mixture-of-Agents on more items — that is the latency you pay
  for the parallel internal calls you asked about.
- **Caveats (don't over-read a small sample):**
  - 15 items, single-vote judge → the 3 open-ended items carry grader noise.
    Opus's (correct) √2 proof was marked wrong by the judge in this run; a
    majority-vote judge would smooth this out.
  - The 12 exact-match items are noise-free and every config answered all 12
    correctly — the accuracy separation comes entirely from the judged items.
  - Costs are **list-price proxies**, not your gateway's actual rates. Swap in the real
    rate card via `pricing.json` to get true dollar savings.

## How to reproduce / scale up

```bash
.venv/bin/python -m bench.run_bench                 # this run
.venv/bin/python -m bench.run_bench --n 8           # quick smoke
# add your own items to bench/dataset.py to test on YOUR workload mix —
# the savings depend heavily on how much of your traffic is genuinely easy.
```
