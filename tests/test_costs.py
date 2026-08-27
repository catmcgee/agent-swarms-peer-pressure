from swarmstop.costs import TokenPrice


def test_token_cost_separates_cached_input() -> None:
    price = TokenPrice(
        provider="test",
        model="model",
        input_per_million=2.0,
        cached_input_per_million=0.5,
        output_per_million=10.0,
        observed_at="2026-08-27",
        source="https://example.invalid",
    )

    cost = price.cost(input_tokens=1_000_000, cached_input_tokens=250_000, output_tokens=100_000)

    assert cost == 2.625
