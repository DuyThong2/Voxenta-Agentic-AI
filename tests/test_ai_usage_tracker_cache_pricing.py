from types import SimpleNamespace

from infra.message_broker import ai_usage_tracker


def _fake_response(input_tokens: int, output_tokens: int, cache_read: int) -> SimpleNamespace:
    return SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_token_details": {"cache_read": cache_read},
        }
    )


def test_cache_read_tokens_are_priced_at_the_cached_rate_not_full_input_rate() -> None:
    answer_id = "answer-cache-test"
    ai_usage_tracker._usage_buffer.pop(answer_id, None)

    # gpt-5.4: input=2.50/Mtok, output=15.00/Mtok, cached_input=0.25/Mtok (see ai_usage_pricing.py).
    # 1,000,000 input tokens, half of them cache reads.
    response = _fake_response(input_tokens=1_000_000, output_tokens=0, cache_read=500_000)

    ai_usage_tracker.record_llm_usage(answer_id, "openai", "gpt-5.4", response)
    items = ai_usage_tracker.pop_usage(answer_id)

    assert len(items) == 1
    cost_usd = items[0].cost_usd

    # Old (buggy) formula: 1,000,000 tokens all at full input price = 2.50.
    old_formula_cost = 2.50
    # New formula: 500k uncached @ 2.50/Mtok + 500k cached @ 0.25/Mtok = 1.25 + 0.125 = 1.375.
    expected_cost = 0.5 * 2.50 + 0.5 * 0.25

    assert cost_usd < old_formula_cost
    assert cost_usd == round(expected_cost, 8)


def test_no_cache_read_falls_back_to_full_input_price() -> None:
    answer_id = "answer-no-cache-test"
    ai_usage_tracker._usage_buffer.pop(answer_id, None)

    response = _fake_response(input_tokens=1_000_000, output_tokens=0, cache_read=0)

    ai_usage_tracker.record_llm_usage(answer_id, "openai", "gpt-5.4", response)
    items = ai_usage_tracker.pop_usage(answer_id)

    assert items[0].cost_usd == round(2.50, 8)
