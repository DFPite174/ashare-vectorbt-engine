"""
ashare-vectorbt-engine 极速动量超参数扫描与双引擎一致性快速示例
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from decimal import Decimal
import numpy as np
import pandas as pd

from engine import GridOptimizer, DualEngineBridge, ExecutionAssumptions


def main():
    print("[INIT] 启动 ashare-vectorbt-engine 极速扫描示例...")

    # 1. 构造演示行情 (2025年 60 个交易日)
    dates = pd.date_range("2025-01-01", periods=60, freq="B").strftime("%Y-%m-%d").tolist()
    bars = []
    np.random.seed(42)
    p1, p2 = 12.0, 1500.0
    for d in dates:
        p1 *= (1.0 + np.random.normal(0.001, 0.02))
        p2 *= (1.0 + np.random.normal(0.0015, 0.015))
        bars.append({"trade_date": d, "symbol": "000001.SZ", "open": p1, "high": p1*1.02, "low": p1*0.98, "close": p1, "volume": 1000000})
        bars.append({"trade_date": d, "symbol": "600519.SH", "open": p2, "high": p2*1.02, "low": p2*0.98, "close": p2, "volume": 1000000})
    bars_df = pd.DataFrame(bars)

    # 2. 运行网格优化
    optimizer = GridOptimizer(ExecutionAssumptions(initial_cash=Decimal("10000000.0")))
    opt_res = optimizer.optimize_momentum_strategy(
        bars_df=bars_df,
        lookback_windows=[5, 10, 20],
        top_k_list=[1, 2],
        rebalance_intervals=[5, 10],
    )

    print(f"[OK] 完成 {opt_res.total_combinations_tested} 组参数扫描，耗时 {opt_res.elapsed_time_sec:.3f} 秒!")
    print("\n[TOP 3] 参数排行榜:")
    print(opt_res.leaderboard_df.head(3)[["lookback_window", "top_k", "rebalance_interval", "total_return", "sharpe_ratio", "max_drawdown"]])

    # 3. 双引擎核验最优组合
    best_p = opt_res.best_params
    print(f"\n[TARGET] 提交最优参数进行双引擎事件驱动核验: {best_p}")
    pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
    mom = pivoted_close.pct_change(best_p["lookback_window"]).fillna(-999.0)
    target_weights = pd.DataFrame(0.0, index=dates, columns=["000001.SZ", "600519.SH"])
    rebalance_mask = pd.Series(False, index=dates)

    w_val = 1.0 / best_p["top_k"]
    for i in range(best_p["lookback_window"], len(dates), best_p["rebalance_interval"]):
        dt = dates[i]
        top_syms = mom.loc[dt].nlargest(best_p["top_k"]).index.tolist()
        for s in top_syms:
            target_weights.loc[dt, s] = w_val
        rebalance_mask.iloc[i] = True

    bridge = DualEngineBridge(ExecutionAssumptions(initial_cash=Decimal("10000000.0")))
    run_res = bridge.run_dual(
        bars_df=bars_df,
        target_weights_df=target_weights,
        strategy_name="Quickstart_Momentum",
        rebalance_mask=rebalance_mask,
    )

    print("[OK] 双引擎核验报告摘要:")
    print(f"   - VectorBT 累计收益:      {run_res.attribution.vbt_total_return*100:+.2f}%")
    print(f"   - NautilusTrader 累计收益: {run_res.attribution.nt_total_return*100:+.2f}%")
    print(f"   - 绝对偏差:              {run_res.attribution.total_delta_return*100:+.2f}%")
    print(f"   - 一致性裁定:            {run_res.attribution.consistency_verdict}")


if __name__ == "__main__":
    main()
