"""
ashare-vectorbt-engine 独立自动化验证套件
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal

from engine.contracts import ExecutionAssumptions
from engine.costs import AShareCostModel, AShareCostConfig
from engine.sizing import round_order_qty_nb, allocate_target_shares_nb
from engine.ashare_sim import EnhancedVectorBTEngine
from engine.grid_optimizer import GridOptimizer
from engine.bridge import DualEngineBridge


def test_statutory_costs():
    model = AShareCostModel(AShareCostConfig())
    fee, eff_px = model.calculate_cost("BUY", price=10.0, qty=100.0)
    assert abs(fee - 5.01) < 1e-3


def test_sizing_and_odd_lot():
    b_qty = round_order_qty_nb(target_shares=285.0, current_shares=0.0)
    assert b_qty == 200.0
    s_full = round_order_qty_nb(target_shares=0.0, current_shares=250.0, allow_odd_lot_sell=True)
    assert s_full == -250.0


def test_dual_engine_consistency():
    dates = pd.date_range("2025-01-01", periods=20, freq="B").strftime("%Y-%m-%d").tolist()
    bars = []
    np.random.seed(42)
    p1, p2 = 20.0, 100.0
    for d in dates:
        p1 *= (1.0 + np.random.normal(0.001, 0.015))
        p2 *= (1.0 + np.random.normal(0.002, 0.015))
        bars.append({"trade_date": d, "symbol": "000001.SZ", "open": p1, "high": p1*1.01, "low": p1*0.99, "close": p1, "volume": 1000000})
        bars.append({"trade_date": d, "symbol": "600519.SH", "open": p2, "high": p2*1.01, "low": p2*0.99, "close": p2, "volume": 1000000})
    bars_df = pd.DataFrame(bars)

    weights_df = pd.DataFrame({
        "000001.SZ": [0.5] * len(dates),
        "600519.SH": [0.5] * len(dates),
    }, index=dates)

    bridge = DualEngineBridge(ExecutionAssumptions(initial_cash=Decimal("10000000.0")))
    res = bridge.run_dual(bars_df=bars_df, target_weights_df=weights_df)

    assert res.attribution.consistency_verdict in ("EXCELLENT", "ACCEPTABLE")
    assert abs(res.attribution.total_delta_return) < 0.02
