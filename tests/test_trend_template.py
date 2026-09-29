import math

from screener.trend_template import evaluate_trend_template, relative_strength_score, rs_ratings


def test_leader_in_stage_two_passes(vcp_setup):
    result = evaluate_trend_template(vcp_setup, rs_rating=95)
    assert result.passed, result.criteria
    assert all(result.criteria.values())
    assert result.metrics["sma50"] > result.metrics["sma150"] > result.metrics["sma200"]


def test_weak_relative_strength_fails(vcp_setup):
    result = evaluate_trend_template(vcp_setup, rs_rating=55)
    assert not result.passed
    assert [k for k, ok in result.criteria.items() if not ok] == ["rs_rating_strong"]


def test_missing_rs_rating_fails(vcp_setup):
    assert not evaluate_trend_template(vcp_setup, rs_rating=None).passed


def test_downtrend_fails(downtrend):
    result = evaluate_trend_template(downtrend, rs_rating=95)
    assert not result.passed
    assert not result.criteria["price_above_sma150_and_sma200"]
    assert not result.criteria["sma200_rising"]


def test_short_history_is_rejected(vcp_setup):
    result = evaluate_trend_template(vcp_setup.iloc[-100:], rs_rating=95)
    assert not result.passed
    assert "bars" in result.reason


def test_rs_ratings_rank_the_universe(vcp_setup, downtrend, sideways):
    ratings = rs_ratings({"LEADER": vcp_setup, "LAGGARD": downtrend, "FLAT": sideways, "NEW": vcp_setup.iloc[-100:]})
    assert ratings["LEADER"] == 99
    assert ratings["LEADER"] > ratings["FLAT"] > ratings["LAGGARD"]
    assert "NEW" not in ratings  # under a year of history: cannot be rated
    assert math.isnan(relative_strength_score(vcp_setup["close"].iloc[-100:]))
