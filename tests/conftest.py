"""Synthetic price series with a known shape, so expected pivots can be computed by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

WICK = 0.005  # high/low sit 0.5% above/below the close


def make_ohlcv(anchors: list[tuple[int, float]], volumes: list[tuple[int, float]]) -> pd.DataFrame:
    """Piecewise-linear closes through ``anchors`` [(bar, price)], volumes likewise."""
    n = anchors[-1][0] + 1
    bars = np.arange(n)
    close = np.interp(bars, [a[0] for a in anchors], [a[1] for a in anchors])
    volume = np.interp(bars, [v[0] for v in volumes], [v[1] for v in volumes])
    index = pd.bdate_range(end="2026-09-16", periods=n)
    return pd.DataFrame(
        {"open": close, "high": close * (1 + WICK), "low": close * (1 - WICK), "close": close, "volume": volume},
        index=index,
    )


# 300-bar advance from 40 to 100, then a 12-week base: -20% / -8.7% / -3.9%, volume drying up.
VCP_ANCHORS = [(0, 40), (300, 100), (315, 80), (330, 98), (340, 89.5), (350, 97), (356, 93.2), (360, 96)]
VCP_VOLUMES = [(0, 1_000_000), (300, 1_200_000), (330, 900_000), (350, 600_000), (360, 400_000)]
PIVOT = 97 * (1 + WICK)
LAST_LOW = 93.2 * (1 - WICK)


@pytest.fixture
def vcp_setup() -> pd.DataFrame:
    return make_ohlcv(VCP_ANCHORS, VCP_VOLUMES)


@pytest.fixture
def vcp_breakout() -> pd.DataFrame:
    """Same base, then two bars through the pivot on a volume surge."""
    return make_ohlcv(VCP_ANCHORS + [(362, 99.5)], VCP_VOLUMES + [(361, 3_000_000), (362, 3_000_000)])


@pytest.fixture
def vcp_extended() -> pd.DataFrame:
    """Broke out and ran 13% past the pivot: too late to buy."""
    return make_ohlcv(VCP_ANCHORS + [(368, 110)], VCP_VOLUMES + [(361, 3_000_000), (368, 2_000_000)])


@pytest.fixture
def downtrend() -> pd.DataFrame:
    return make_ohlcv([(0, 100), (360, 60)], [(0, 1_000_000), (360, 1_000_000)])


@pytest.fixture
def sideways() -> pd.DataFrame:
    return make_ohlcv([(0, 50), (120, 54), (240, 49), (360, 52)], [(0, 1_000_000), (360, 1_000_000)])
