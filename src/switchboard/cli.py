"""Command-line interface for switchboard.

switchboard serve [--host H] [--port P]   run the OpenAI-compatible server
switchboard ask "<prompt>" [--mode cost]  route one prompt and print telemetry
switchboard models                        probe which gateway models are live
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def _cmd_serve(args: argparse.Namespace) -> int:
    from switchboard.server import run

    run(host=args.host, port=args.port)
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from switchboard.engine import Engine

    async def go() -> int:
        eng = Engine()
        try:
            rr = await eng.answer(
                [{"role": "user", "content": args.prompt}],
                mode=args.mode,
                max_tokens=args.max_tokens,
            )
        finally:
            await eng.aclose()
        print(rr.content)
        print(
            f"\n[route={rr.route} | cost=${rr.cost:.6f} "
            f"| baseline(opus)≈${rr.baseline_cost_est:.6f} "
            f"| savings={rr.savings_pct:.1f}% | {rr.latency_ms:.0f}ms]",
            file=sys.stderr,
        )
        return 0

    return asyncio.run(go())


def _cmd_models(args: argparse.Namespace) -> int:
    from switchboard import config
    from switchboard.gateway import Gateway

    async def go() -> int:
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

        async def probe(m: str):
            c = await gw.complete(m, [{"role": "user", "content": "Reply with one word: ok"}], max_tokens=300)
            return m, c.ok, "" if c.ok else (c.error or "")[:90]

        results = await asyncio.gather(*(probe(m) for m in pool))
        await gw.aclose()
        bad = [r for r in results if not r[1]]
        for m, ok, err in results:
            print(f"  {'OK  ' if ok else 'FAIL'} {m:<26}{'' if ok else '  <- ' + err}")
        print(
            f"\n{len(results) - len(bad)}/{len(results)} live."
            + (f"  BROKEN: {[b[0] for b in bad]}" if bad else "  all good.")
        )
        return 1 if bad else 0

    return asyncio.run(go())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="switchboard", description="OpenAI-compatible LLM router.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the OpenAI-compatible server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    p_ask = sub.add_parser("ask", help="route one prompt and print the answer + telemetry")
    p_ask.add_argument("prompt")
    p_ask.add_argument("--mode", default="balanced", choices=["balanced", "cost", "quality"])
    p_ask.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    p_ask.set_defaults(func=_cmd_ask)

    p_models = sub.add_parser("models", help="probe which gateway models are actually live")
    p_models.set_defaults(func=_cmd_models)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
