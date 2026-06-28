"""Verify every model the router relies on actually answers.

The gateway's /v1/models list is stale (advertises ids that 404 at call time),
so never trust it — call each model with a 1-token probe and report.

    .venv/bin/python -m tools.validate_models
"""

from __future__ import annotations

import asyncio

from switchboard import config
from switchboard.gateway import Gateway


async def _probe(gw: Gateway, m: str) -> tuple[str, bool, str]:
    c = await gw.complete(m, [{"role": "user", "content": "Reply with one word: ok"}], max_tokens=300)
    return m, c.ok, "" if c.ok else (c.error or "")[:90]


async def main() -> None:
    pool = sorted(
        set(
            config.CHEAP
            + config.MID
            + config.STRONG
            + [config.CLASSIFIER_MODEL, config.JUDGE_MODEL, config.DEFAULT_CHEAP, config.DEFAULT_MID]
            + config.MOA_PROPOSERS
            + [config.MOA_SYNTHESIZER]
            + config.BASELINE_MODELS
        )
    )
    gw = Gateway()
    results = await asyncio.gather(*(_probe(gw, m) for m in pool))
    bad = [r for r in results if not r[1]]
    for m, ok, err in results:
        print(f"  {'OK  ' if ok else 'FAIL'} {m:<26}{'' if ok else '  <- ' + err}")
    print(
        f"\n{len(results) - len(bad)}/{len(results)} live."
        + (f"  BROKEN: {[b[0] for b in bad]}" if bad else "  all good.")
    )
    await gw.aclose()


if __name__ == "__main__":
    asyncio.run(main())
