from dataclasses import replace

import numpy as np
import pytest

from screener.config import TradePlanConfig, VCPConfig
from screener.trade_plan import build_trade_plan
from screener.vcp import (
    ENTRY_CHEAT,
    ENTRY_PIVOT,
    STATUS_BREAKOUT,
    STATUS_SETUP,
    detect_vcp,
    find_contractions,
)

from .conftest import LAST_LOW, PIVOT, VCP_ANCHORS, VCP_VOLUMES, make_ohlcv


def test_find_contractions_walks_high_low_high_low():
    high = np.array([10, 12, 11, 9, 10, 11.5, 10.5, 11, 11.2, 10.9, 11.0], dtype=float)
    low = high - 0.5
    pairs = find_contractions(high, low, 0, min_depth_pct=2.0)
    assert pairs == [(1, 3), (5, 6), (8, 9)]


def test_detects_setup_below_pivot(vcp_setup):
    vcp = detect_vcp(vcp_setup)
    assert vcp is not None
    assert vcp.status == STATUS_SETUP
    assert vcp.pivot == pytest.approx(PIVOT)
    assert vcp.last_contraction_low == pytest.approx(LAST_LOW)
    assert [round(c.depth_pct) for c in vcp.contractions] == [21, 10, 5]
    assert vcp.footprint == "12W 21/10/5 3T"
    assert vcp.volume_dry_up_ratio < 0.7
    assert vcp.quietest_day_ratio < 0.6
    assert vcp.entry_type == ENTRY_PIVOT
    assert vcp.pivot_position == pytest.approx((PIVOT - 80 * 0.995) / (100 * 1.005 - 80 * 0.995))
    assert -5 < vcp.pct_from_pivot < 0


def test_detects_fresh_breakout_with_volume(vcp_breakout):
    vcp = detect_vcp(vcp_breakout)
    assert vcp is not None
    assert vcp.status == STATUS_BREAKOUT
    # Must anchor on the tight final pivot, not the wider one above it.
    assert vcp.pivot == pytest.approx(PIVOT)
    assert vcp.breakout_bars_ago == 1
    assert vcp.breakout_volume_confirmed is True
    assert 0 < vcp.pct_from_pivot < 5


def test_extended_stock_is_not_actionable(vcp_extended):
    assert detect_vcp(vcp_extended) is None


def test_no_pattern_in_downtrend_or_chop(downtrend, sideways):
    assert detect_vcp(downtrend) is None
    assert detect_vcp(sideways) is None


def test_rejects_base_without_volume_dry_up():
    rising_volume = [(0, 1_000_000), (340, 1_000_000), (360, 2_500_000)]
    assert detect_vcp(make_ohlcv(VCP_ANCHORS, rising_volume)) is None


def test_rejects_wide_and_loose_final_contraction():
    loose = VCP_ANCHORS[:5] + [(350, 97), (356, 84), (360, 93)]  # last pullback -13%
    assert detect_vcp(make_ohlcv(loose, VCP_VOLUMES)) is None


def test_rejects_setup_far_below_pivot():
    far = VCP_ANCHORS[:-1] + [(360, 93.5)]
    assert detect_vcp(make_ohlcv(far, VCP_VOLUMES), replace(VCPConfig(), max_pct_below_pivot=2.0)) is None


def test_failed_breakout_that_undercuts_base_is_rejected():
    failed = VCP_ANCHORS + [(361, 98.5), (363, 92), (365, 98)]
    vcp = detect_vcp(make_ohlcv(failed, VCP_VOLUMES + [(365, 400_000)]))
    # The 2-bar plunge and V-recovery must not be read as a fresh tight contraction either.
    assert vcp is None


def test_rejects_volume_that_is_only_marginally_below_average():
    barely = [(0, 1_000_000), (349, 1_000_000), (350, 970_000), (360, 970_000)]
    assert detect_vcp(make_ohlcv(VCP_ANCHORS, barely)) is None


def test_rejects_quiet_average_without_a_truly_quiet_session():
    # 10-bar average is 0.69x normal, but no single session drops below 0.6x.
    flat_low = [(0, 1_000_000), (350, 1_000_000), (351, 640_000), (360, 640_000)]
    df = make_ohlcv(VCP_ANCHORS, flat_low)
    assert detect_vcp(df) is None
    assert detect_vcp(df, replace(VCPConfig(), max_quietest_day_ratio=0.7)) is not None


DEEP_PIVOT = [(0, 40), (300, 100), (315, 68), (330, 82), (340, 74), (350, 80.5), (356, 77.5), (360, 79.8)]


def test_pivot_deep_inside_the_base_is_not_a_standard_entry():
    df = make_ohlcv(DEEP_PIVOT, VCP_VOLUMES)
    assert detect_vcp(df) is None

    cheat = detect_vcp(df, replace(VCPConfig(), allow_cheat_entries=True))
    assert cheat is not None
    assert cheat.entry_type == ENTRY_CHEAT
    assert 1 / 3 <= cheat.pivot_position < 2 / 3


def test_breakout_is_the_first_close_above_the_pivot_not_an_intraday_poke():
    # Bar 361 wicks through the pivot on thin volume but closes under it; the real
    # breakout is bar 364, closing above the pivot on a volume surge.
    df = make_ohlcv(
        VCP_ANCHORS + [(361, 96.8), (363, 96.9), (364, 99.5)],
        VCP_VOLUMES + [(361, 300_000), (363, 300_000), (364, 4_000_000)],
    )
    df.iloc[361, df.columns.get_loc("high")] = 97.9
    vcp = detect_vcp(df)
    assert vcp is not None
    assert vcp.status == STATUS_BREAKOUT
    assert vcp.pivot == pytest.approx(PIVOT)
    assert vcp.breakout_bars_ago == 0
    assert vcp.breakout_volume_confirmed is True


def test_poke_without_a_close_above_the_pivot_is_not_a_breakout():
    df = make_ohlcv(VCP_ANCHORS + [(361, 96.8), (362, 96.9)], VCP_VOLUMES + [(362, 350_000)])
    df.iloc[361, df.columns.get_loc("high")] = 97.9
    vcp = detect_vcp(df)
    assert vcp is None or vcp.status == STATUS_SETUP


class TestTradePlan:
    STOP = LAST_LOW * 0.995

    def test_setup_is_planned_from_the_pivot(self, vcp_setup):
        plan = build_trade_plan(detect_vcp(vcp_setup))
        assert plan.entry_low == round(PIVOT, 2)
        assert plan.reference_entry == plan.entry_low
        assert plan.stop_loss == round(self.STOP, 2)
        assert plan.target == round(PIVOT + 3 * (PIVOT - self.STOP), 2)
        assert plan.stop_basis == "below low of last contraction"
        assert plan.stop_loss < plan.entry_low < plan.entry_high < plan.target

    def test_entry_range_never_breaks_the_max_risk_rule(self, vcp_setup):
        plan = build_trade_plan(detect_vcp(vcp_setup))
        # 5% above the pivot would be 102.36, but the stop is 8% away already at 100.29.
        assert plan.entry_high == round(self.STOP / 0.92, 2)
        assert plan.entry_high < round(PIVOT * 1.05, 2)
        assert plan.max_risk_pct == pytest.approx(8.0, abs=0.01)

        roomy = build_trade_plan(detect_vcp(vcp_setup), TradePlanConfig(max_stop_pct=12.0))
        assert roomy.entry_high == round(PIVOT * 1.05, 2)

    def test_breakout_risk_is_measured_from_the_price_paid(self, vcp_breakout):
        vcp = detect_vcp(vcp_breakout)
        plan = build_trade_plan(vcp)
        assert plan.reference_entry == 99.5
        assert plan.risk_pct == pytest.approx((99.5 - self.STOP) / 99.5 * 100, abs=0.01)
        assert plan.risk_pct > (PIVOT - self.STOP) / PIVOT * 100  # more than the pivot-based figure
        assert plan.risk_pct <= 8.0
        assert plan.target == round(99.5 + 3 * (99.5 - self.STOP), 2)

    def test_breakout_too_far_above_the_stop_is_not_buyable(self):
        # 4.6% past the pivot is inside the classic 5% buy zone, but 9.5% above the stop.
        df = make_ohlcv(VCP_ANCHORS + [(362, 102.0)], VCP_VOLUMES + [(361, 3_000_000), (362, 3_000_000)])
        vcp = detect_vcp(df)
        assert vcp is not None and vcp.status == STATUS_BREAKOUT
        assert build_trade_plan(vcp) is None

    def test_setup_needing_a_wider_stop_than_allowed_is_skipped(self, vcp_setup):
        # Logical stop is 5.35% under the pivot: fine at 8%, not a trade at 4%.
        assert build_trade_plan(detect_vcp(vcp_setup), TradePlanConfig(max_stop_pct=4.0)) is None

    def test_stop_is_not_tighter_than_minimum(self, vcp_setup):
        plan = build_trade_plan(detect_vcp(vcp_setup), TradePlanConfig(min_stop_pct=6.0))
        assert plan.risk_pct == pytest.approx(6.0)
        assert "minimum" in plan.stop_basis
