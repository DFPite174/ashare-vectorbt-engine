"""
A 股高通量超维网格优化器 (Grid Optimizer)
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Sequence
import numpy as np
import pandas as pd

from engine.contracts import ExecutionAssumptions
from engine.ashare_sim import EnhancedVectorBTEngine, VectorBTResult


@dataclass
class GridOptimizationResult:
    leaderboard_df: pd.DataFrame
    best_params: Dict[str, Any]
    best_result: VectorBTResult
    total_combinations_tested: int
    elapsed_time_sec: float


class GridOptimizer:
    def __init__(self, assumptions: ExecutionAssumptions | None = None):
        self.assumptions = assumptions or ExecutionAssumptions()
        self.engine = EnhancedVectorBTEngine(self.assumptions)

    def optimize_momentum_strategy(
        self,
        bars_df: pd.DataFrame,
        lookback_windows: Sequence[int] = (5, 10, 20, 30, 60),
        top_k_list: Sequence[int] = (1, 2, 3),
        rebalance_intervals: Sequence[int] = (1, 5, 10, 20),
        rank_metric: str = "sharpe_ratio",
        corporate_actions_df: Optional[pd.DataFrame] = None,
    ) -> GridOptimizationResult:
        import time
        start_t = time.perf_counter()

        pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
        timeline = sorted(list(pivoted_close.index))
        symbols = sorted(list(pivoted_close.columns))
        n_days = len(timeline)

        combinations = list(itertools.product(lookback_windows, top_k_list, rebalance_intervals))
        records = []
        best_res: Optional[VectorBTResult] = None
        best_metric_val = -1e9
        best_param_dict: Dict[str, Any] = {}

        for w_lookback, top_k, reb_step in combinations:
            mom_df = pivoted_close.pct_change(periods=w_lookback).fillna(-999.0)

            weights_df = pd.DataFrame(0.0, index=timeline, columns=symbols)
            for i in range(w_lookback, n_days, reb_step):
                dt = timeline[i]
                row_mom = mom_df.loc[dt]
                top_syms = row_mom.nlargest(top_k).index.tolist()
                w_per_sym = 1.0 / top_k
                for s in top_syms:
                    weights_df.loc[dt, s] = w_per_sym

            rebalance_mask = pd.Series(False, index=timeline)
            for i in range(w_lookback, n_days, reb_step):
                rebalance_mask.iloc[i] = True

            res = self.engine.run(
                bars_df=bars_df,
                target_weights_df=weights_df,
                rebalance_mask=rebalance_mask,
                corporate_actions_df=corporate_actions_df,
            )

            rec = {
                "lookback_window": w_lookback,
                "top_k": top_k,
                "rebalance_interval": reb_step,
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "calmar_ratio": res.calmar_ratio,
                "win_rate": res.metrics["win_rate"],
                "annualized_turnover": res.metrics["annualized_turnover"],
                "total_fees_paid": res.metrics["total_fees_paid"],
                "filled_orders": res.metrics["filled_orders_count"],
                "blocked_orders": res.metrics["blocked_orders_count"],
            }
            records.append(rec)

            val = res.metrics.get(rank_metric, -1e9)
            if val > best_metric_val:
                best_metric_val = val
                best_res = res
                best_param_dict = {
                    "lookback_window": w_lookback,
                    "top_k": top_k,
                    "rebalance_interval": reb_step,
                }

        leaderboard = pd.DataFrame(records).sort_values(by=rank_metric, ascending=False).reset_index(drop=True)
        elapsed = time.perf_counter() - start_t

        return GridOptimizationResult(
            leaderboard_df=leaderboard,
            best_params=best_param_dict,
            best_result=best_res,  # type: ignore
            total_combinations_tested=len(combinations),
            elapsed_time_sec=elapsed,
        )
