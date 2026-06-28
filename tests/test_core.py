"""Offline unit tests — pure logic only, no network or gateway required."""

from switchboard import cache, classify, cost_usd, gateway, price_of
from switchboard.engine import _est_tokens, _user_request_text


def test_cost_usd_is_per_million_tokens():
    # opus = (15, 75) per 1M -> 1M in + 1M out = 90.00
    assert cost_usd("claude-opus-4-8", 1_000_000, 1_000_000) == 90.0
    assert cost_usd("claude-opus-4-8", 0, 0) == 0.0


def test_price_of_falls_back_for_unknown_model():
    assert price_of("totally-made-up-model") == (1.0, 5.0)


def test_heuristic_flags_trivial_prompt_confidently():
    diff, domain, confident = classify._heuristic("Hi there!")
    assert confident is True
    assert diff <= 2.0
    assert domain == "chat"


def test_heuristic_flags_long_codey_prompt_as_hard():
    text = "Design and optimize this. " + "def f():\n    pass\n" * 200
    diff, _domain, confident = classify._heuristic(text)
    assert confident is True
    assert diff >= 4.0


def test_tier_thresholds():
    assert classify._tier_for(1.5) == "cheap"
    assert classify._tier_for(3.0) == "mid"
    assert classify._tier_for(4.5) == "hard"


def test_last_user_text_handles_str_and_parts():
    assert classify._last_user_text([{"role": "user", "content": "hello"}]) == "hello"
    parts = [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    assert "a" in classify._last_user_text(parts)


def test_reasoning_model_detection():
    assert gateway._is_reasoning_model("gpt-5-nano") is True
    assert gateway._is_reasoning_model("o3-deep-research") is True
    assert gateway._is_reasoning_model("claude-haiku-4-5") is False
    assert gateway._is_reasoning_model("gemini-3.5-flash") is False


def test_response_cache_hit_miss_and_ttl():
    c = cache.ResponseCache(max_items=8)
    msgs = [{"role": "user", "content": "x"}]
    assert c.get(msgs, "cost") is None
    assert c.misses == 1
    c.put(msgs, "cost", {"answer": 42})
    assert c.get(msgs, "cost") == {"answer": 42}
    assert c.hits == 1
    # different mode is a different key
    assert c.get(msgs, "balanced") is None


def test_engine_text_helpers():
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello world"}]
    text = _user_request_text(msgs)
    assert "hello world" in text
    assert _est_tokens(msgs) >= 1
