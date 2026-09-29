"""Turn a detected VCP into an entry range, stop-loss and target."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ScreenerConfig, TradePlanConfig
from .vcp import STATUS_BREAKOUT, VCPResult, find_vcp_candidates


@dataclass
class TradePlan:
    entry_low: float  # the pivot
    entry_high: float  # highest price at which the trade still fits the risk limit
    reference_entry: float  # price the risk/target figures assume you pay
    stop_loss: float
    target: float
    risk_pct: float  # from reference_entry
    max_risk_pct: float  # if bought at entry_high
    reward_pct: float
    reward_to_risk: float
    stop_basis: str


def build_trade_plan(
    vcp: VCPResult, cfg: TradePlanConfig = TradePlanConfig(), buy_zone_pct: float = 5.0
) -> TradePlan | None:
    """Buy through the pivot, stop under the last contraction, target a multiple of risk.

    The stop goes just below the low of the final contraction: if price returns
    there the pattern has failed. Minervini never risks more than 7-8% *from his
    purchase price*, which shapes the plan in three ways:

    - if that logical stop is more than ``max_stop_pct`` below the pivot the setup
      is rejected (``None``) - the final contraction is too loose for a low-risk entry;
    - the top of the entry range is the lower of the buy zone and the price at which
      the stop would be ``max_stop_pct`` away, so no price in the range breaks the rule;
    - risk and target are measured from the pivot for a setup, but from the current
      price for a stock that has already broken out, because that is what you would pay.
    """
    pivot = vcp.pivot
    stop = vcp.last_contraction_low * (1 - cfg.stop_buffer_pct / 100)
    basis = "below low of last contraction"
    if stop < pivot * (1 - cfg.max_stop_pct / 100):
        return None
    tightest = pivot * (1 - cfg.min_stop_pct / 100)
    if stop > tightest:
        stop, basis = tightest, f"{cfg.min_stop_pct:g}% minimum stop distance"

    entry_high = min(pivot * (1 + buy_zone_pct / 100), stop / (1 - cfg.max_stop_pct / 100))
    reference = vcp.price if vcp.status == STATUS_BREAKOUT else pivot
    if reference > entry_high:
        return None  # already too far above the stop to buy within the risk limit

    risk = reference - stop
    target = reference + cfg.reward_to_risk * risk
    return TradePlan(
        entry_low=round(pivot, 2),
        entry_high=round(entry_high, 2),
        reference_entry=round(reference, 2),
        stop_loss=round(stop, 2),
        target=round(target, 2),
        risk_pct=round(risk / reference * 100, 2),
        max_risk_pct=round((entry_high - stop) / entry_high * 100, 2),
        reward_pct=round((target / reference - 1) * 100, 2),
        reward_to_risk=cfg.reward_to_risk,
        stop_basis=basis,
    )


def plan_best_candidate(df: pd.DataFrame, config: ScreenerConfig) -> tuple[VCPResult, TradePlan] | None:
    """Best reading of the chart that can also be bought within the risk limit."""
    for candidate in find_vcp_candidates(df, config.vcp):
        plan = build_trade_plan(candidate, config.plan, config.vcp.buy_zone_pct)
        if plan is not None:
            return candidate, plan
    return None
