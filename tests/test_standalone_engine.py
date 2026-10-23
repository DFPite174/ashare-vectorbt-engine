"""
ashare-vectorbt-engine 独立自动化验证套件
"""

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal

from engine.contracts import ExecutionAssumptions, ExecutionPriceMode
from engine.costs import AShareCostModel, AShareCostConfig, calc_market_impact_nb
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


def test_vwap_and_impact_cost():
    # 验证冲击函数
    imp = calc_market_impact_nb(400000.0, 10000000.0, 0.02, 0.10)
    assert abs(imp - 0.0004) < 1e-6

    # 验证引擎端到端 VWAP 与 冲击成本
    bars = [
        {"trade_date": "2025-01-02", "symbol": "000001.SZ", "open": 10.0, "high": 11.0, "low": 9.5, "close": 10.0, "volume": 10000, "amount": 105000.0},
        {"trade_date": "2025-01-03", "symbol": "000001.SZ", "open": 10.5, "high": 12.0, "low": 10.0, "close": 11.0, "volume": 10000, "amount": 112000.0},
    ]
    bars_df = pd.DataFrame(bars)
    target_weights = pd.DataFrame({"000001.SZ": [0.5, 0.5]}, index=["2025-01-02", "2025-01-03"])

    assumptions = ExecutionAssumptions(
        initial_cash=Decimal("1000000.0"),
        price_mode=ExecutionPriceMode.VWAP,
        enable_impact_cost=True,
        impact_cost_coef=Decimal("0.10"),
    )
    engine = EnhancedVectorBTEngine(assumptions)
    res = engine.run(bars_df=bars_df, target_weights_df=target_weights)

    assert "total_market_impact_paid" in res.metrics
    assert res.metrics["total_market_impact_paid"] > 0.0
    assert "impact_cost" in res.orders_df.columns
