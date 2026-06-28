# Contributing

Thanks for your interest in switchboard!

## Dev setup

```bash
uv sync --group dev        # install package + dev deps
make check                 # ruff lint + pytest
```

You need an OpenAI-compatible gateway to run anything that hits the network
(the router, `switchboard models`, the benchmarks). Set:

```bash
export OPENAI_API_KEY=...                 # your gateway key
export OPENAI_BASE_URL=https://.../v1     # any OpenAI-compatible endpoint
```

## Layout

```
src/switchboard/      the package (gateway client, triage, engine, server, cli)
bench/                benchmark harnesses (GSM8K + a mixed smoke set)
tools/                dev utilities (live-model probe)
tests/                offline unit tests (no network)
```

## Guidelines

- Keep the core engine dependency-light (`httpx` only); the server adds
  `fastapi`/`uvicorn`.
- Pure logic (triage heuristics, cost math, answer extraction) should have
  offline unit tests in `tests/`.
- Pricing in `src/switchboard/config.py` is a list-price proxy — don't hardcode
  any single provider's real rates; users override via `pricing.json`.
- Run `make format` before committing. CI runs ruff + pytest on 3.11–3.13.

## Releasing (maintainers)

Bump `__version__` in `src/switchboard/__init__.py`, then tag:

```bash
git tag v0.1.1 && git push origin v0.1.1
```

The release workflow builds and publishes to PyPI via Trusted Publishing.
