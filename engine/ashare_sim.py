"""
A 股高性能 Numba 向量化仿真内核 (Enhanced VectorBT Simulation Kernel)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import numba as nb

from engine.contracts import ExecutionAssumptions, TargetWeights
from engine.costs import calc_trade_fee_nb, calc_effective_price_nb
from engine.sizing import allocate_target_shares_nb


@nb.njit(fastmath=True)
def simulate_ashare_core_nb(
    close_mat: np.ndarray,
    high_mat: np.ndarray,
    low_mat: np.ndarray,
    weights_mat: np.ndarray,
    up_limit_mat: np.ndarray,
    down_limit_mat: np.ndarray,
    dividend_mat: np.ndarray,
    rebalance_mask: np.ndarray,
    initial_cash: float,
    commission_rate: float,
    min_commission: float,
    stamp_duty_rate: float,
    transfer_fee_rate: float,
    slippage_rate: float,
    lot_size: int = 100,
    min_buy_qty: int = 100,
    cash_buffer_ratio: float = 0.005,
):
    n_bars, n_symbols = close_mat.shape

    cash = initial_cash
    current_shares = np.zeros(n_symbols, dtype=np.float64)
    tradable_shares = np.zeros(n_symbols, dtype=np.float64)
    locked_shares_today = np.zeros(n_symbols, dtype=np.float64)

    daily_equity = np.zeros(n_bars, dtype=np.float64)
    daily_cash = np.zeros(n_bars, dtype=np.float64)
    daily_positions = np.zeros((n_bars, n_symbols), dtype=np.float64)

    max_orders_est = n_bars * n_symbols * 2
    order_records = np.zeros((max_orders_est, 7), dtype=np.float64)
    order_count = 0

    for t in range(n_bars):
        # 1. 推进 T+1 与现金分红
        for s in range(n_symbols):
            tradable_shares[s] += locked_shares_today[s]
            locked_shares_today[s] = 0.0

            div_val = dividend_mat[t, s]
            if div_val > 0.0 and current_shares[s] > 0.0:
                cash += current_shares[s] * div_val

        # 2. 检查调仓
        if rebalance_mask[t]:
            target_w = weights_mat[t]
            curr_px = close_mat[t]

            order_deltas, _ = allocate_target_shares_nb(
                weights=target_w,
                prices=curr_px,
                current_shares=current_shares,
                tradable_shares=tradable_shares,
                current_cash=cash,
                cash_buffer_ratio=cash_buffer_ratio,
                lot_size=lot_size,
                min_buy_qty=min_buy_qty,
            )

            # 卖单
            for s in range(n_symbols):
                delta = order_deltas[s]
                if delta < 0.0:
                    px = curr_px[s]
                    down_limit = down_limit_mat[t, s]
                    wanted_qty = abs(delta)

                    if down_limit > 0.0 and px <= down_limit + 1e-4:
                        if order_count < max_orders_est:
                            order_records[order_count] = [t, s, -1, wanted_qty, px, 0.0, 2.0]
                            order_count += 1
                        continue

                    fill_qty = min(wanted_qty, tradable_shares[s])
                    if fill_qty <= 0.0:
                        continue

                    eff_px = calc_effective_price_nb(-1, px, slippage_rate)
                    fee = calc_trade_fee_nb(
                        -1, eff_px, fill_qty,
                        commission_rate, min_commission, stamp_duty_rate, transfer_fee_rate
                    )

                    trade_proceeds = fill_qty * eff_px - fee
                    cash += trade_proceeds
                    current_shares[s] -= fill_qty
                    tradable_shares[s] -= fill_qty

                    if order_count < max_orders_est:
                        order_records[order_count] = [t, s, -1, fill_qty, eff_px, fee, 0.0]
                        order_count += 1

            # 买单
            for s in range(n_symbols):
                delta = order_deltas[s]
                if delta > 0.0:
                    px = curr_px[s]
                    up_limit = up_limit_mat[t, s]
                    wanted_qty = delta

                    if up_limit > 0.0 and px >= up_limit - 1e-4:
                        if order_count < max_orders_est:
                            order_records[order_count] = [t, s, 1, wanted_qty, px, 0.0, 1.0]
                            order_count += 1
                        continue

                    eff_px = calc_effective_price_nb(1, px, slippage_rate)
                    pre_fee = calc_trade_fee_nb(
                        1, eff_px, wanted_qty,
                        commission_rate, min_commission, stamp_duty_rate, transfer_fee_rate
                    )
                    total_outflow = wanted_qty * eff_px + pre_fee

                    if total_outflow > cash:
                        avail_spend = cash - (initial_cash * 0.001)
                        if avail_spend <= 0.0:
                            if order_count < max_orders_est:
                                order_records[order_count] = [t, s, 1, wanted_qty, px, 0.0, 3.0]
                                order_count += 1
                            continue

                        reduced_shares = (avail_spend / (eff_px * 1.0004)) // lot_size * lot_size
                        if reduced_shares < min_buy_qty:
                            if order_count < max_orders_est:
                                order_records[order_count] = [t, s, 1, wanted_qty, px, 0.0, 3.0]
                                order_count += 1
                            continue
                        fill_qty = reduced_shares
                        fee = calc_trade_fee_nb(
                            1, eff_px, fill_qty,
                            commission_rate, min_commission, stamp_duty_rate, transfer_fee_rate
                        )
                        total_outflow = fill_qty * eff_px + fee
                    else:
                        fill_qty = wanted_qty
                        fee = pre_fee

                    cash -= total_outflow
                    current_shares[s] += fill_qty
                    locked_shares_today[s] += fill_qty

                    if order_count < max_orders_est:
                        order_records[order_count] = [t, s, 1, fill_qty, eff_px, fee, 0.0]
                        order_count += 1

        # 3. 估值
        pos_val = 0.0
        for s in range(n_symbols):
            daily_positions[t, s] = current_shares[s]
            if current_shares[s] > 0.0:
                pos_val += current_shares[s] * close_mat[t, s]

        equity = cash + pos_val
        daily_cash[t] = cash
        daily_equity[t] = equity

    return daily_equity, daily_cash, daily_positions, order_records[:order_count]


@dataclass
class VectorBTResult:
    timeline: List[str]
    symbols: List[str]
    equity_series: pd.Series
    cash_series: pd.Series
    positions_df: pd.DataFrame
    orders_df: pd.DataFrame
    metrics: Dict[str, Any]

    @property
    def total_return(self) -> float:
        return self.metrics.get("total_return", 0.0)

    @property
    def annualized_return(self) -> float:
        return self.metrics.get("annualized_return", 0.0)

    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown", 0.0)

    @property
    def calmar_ratio(self) -> float:
        return self.metrics.get("calmar_ratio", 0.0)


class EnhancedVectorBTEngine:
    def __init__(self, assumptions: ExecutionAssumptions | None = None):
        self.assumptions = assumptions or ExecutionAssumptions()

    def run(
        self,
        bars_df: pd.DataFrame,
        target_weights_df: pd.DataFrame,
        rebalance_mask: Optional[pd.Series] = None,
        price_limits_df: Optional[pd.DataFrame] = None,
        corporate_actions_df: Optional[pd.DataFrame] = None,
    ) -> VectorBTResult:
        pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
        pivoted_high = bars_df.pivot(index="trade_date", columns="symbol", values="high").ffill().bfill()
        pivoted_low = bars_df.pivot(index="trade_date", columns="symbol", values="low").ffill().bfill()

        timeline = sorted(list(pivoted_close.index))
        symbols = sorted(list(pivoted_close.columns))

        aligned_weights = target_weights_df.reindex(index=timeline, columns=symbols).fillna(0.0)

        if rebalance_mask is None:
            weight_diff = (aligned_weights.diff().abs().sum(axis=1) > 1e-4)
            weight_diff.iloc[0] = (aligned_weights.iloc[0].sum() > 1e-4)
            rebalance_bool = weight_diff.values
        else:
            rebalance_bool = rebalance_mask.reindex(timeline).fillna(False).values

        if price_limits_df is not None and "up_limit" in price_limits_df.columns:
            up_lim_piv = price_limits_df.pivot(index="trade_date", columns="symbol", values="up_limit").reindex(index=timeline, columns=symbols)
            down_lim_piv = price_limits_df.pivot(index="trade_date", columns="symbol", values="down_limit").reindex(index=timeline, columns=symbols)
            up_limit_mat = up_lim_piv.fillna(999999.0).values
            down_limit_mat = down_lim_piv.fillna(0.0).values
        else:
            prev_close = pivoted_close.shift(1).fillna(pivoted_close)
            limit_ratios = np.array([0.20 if s.startswith("688") or s.startswith("300") else 0.10 for s in symbols])
            up_limit_mat = np.round(prev_close.values * (1.0 + limit_ratios), 2)
            down_limit_mat = np.round(prev_close.values * (1.0 - limit_ratios), 2)

        dividend_mat = np.zeros((len(timeline), len(symbols)), dtype=np.float64)
        if corporate_actions_df is not None and not corporate_actions_df.empty:
            time_idx = {dt: i for i, dt in enumerate(timeline)}
            sym_idx = {s: j for j, s in enumerate(symbols)}
            for _, row in corporate_actions_df.iterrows():
                ex_d = str(row.get("ex_date", ""))
                s = str(row.get("symbol", ""))
                cash_div = float(row.get("cash_div", 0.0))
                if ex_d in time_idx and s in sym_idx and cash_div > 0.0:
                    dividend_mat[time_idx[ex_d], sym_idx[s]] += cash_div

        close_mat = np.ascontiguousarray(pivoted_close.values, dtype=np.float64)
        high_mat = np.ascontiguousarray(pivoted_high.values, dtype=np.float64)
        low_mat = np.ascontiguousarray(pivoted_low.values, dtype=np.float64)
        weights_mat = np.ascontiguousarray(aligned_weights.values, dtype=np.float64)
        up_limit_mat = np.ascontiguousarray(up_limit_mat, dtype=np.float64)
        down_limit_mat = np.ascontiguousarray(down_limit_mat, dtype=np.float64)
        dividend_mat = np.ascontiguousarray(dividend_mat, dtype=np.float64)
        rebalance_mask_arr = np.ascontiguousarray(rebalance_bool, dtype=np.bool_)

        equity_arr, cash_arr, pos_arr, orders_rec = simulate_ashare_core_nb(
            close_mat=close_mat,
            high_mat=high_mat,
            low_mat=low_mat,
            weights_mat=weights_mat,
            up_limit_mat=up_limit_mat,
            down_limit_mat=down_limit_mat,
            dividend_mat=dividend_mat,
            rebalance_mask=rebalance_mask_arr,
            initial_cash=float(self.assumptions.initial_cash),
            commission_rate=float(self.assumptions.commission_rate),
            min_commission=float(self.assumptions.min_commission),
            stamp_duty_rate=float(self.assumptions.stamp_duty_rate),
            transfer_fee_rate=float(self.assumptions.transfer_fee_rate),
            slippage_rate=float(self.assumptions.slippage) * 0.0001,
            lot_size=100,
            min_buy_qty=100,
            cash_buffer_ratio=0.005,
        )

        equity_series = pd.Series(equity_arr, index=timeline, name="equity")
        cash_series = pd.Series(cash_arr, index=timeline, name="cash")
        positions_df = pd.DataFrame(pos_arr, index=timeline, columns=symbols)

        daily_returns = equity_series.pct_change().fillna(0.0)
        total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1.0

        n_days = len(timeline)
        annualized_ret = (1.0 + total_ret) ** (252.0 / max(n_days, 1)) - 1.0

        ret_std = daily_returns.std()
        sharpe = (daily_returns.mean() / ret_std * np.sqrt(252.0)) if ret_std > 1e-8 else 0.0

        downside_returns = daily_returns[daily_returns < 0.0]
        sortino = (daily_returns.mean() / downside_returns.std() * np.sqrt(252.0)) if len(downside_returns) > 0 and downside_returns.std() > 1e-8 else 0.0

        cummax = equity_series.cummax()
        drawdown_series = (equity_series - cummax) / cummax
        max_dd = abs(drawdown_series.min())
        calmar = (annualized_ret / max_dd) if max_dd > 1e-6 else 0.0

        win_rate = float((daily_returns > 0.0).sum() / max((daily_returns != 0.0).sum(), 1))

        status_map = {0.0: "FILLED", 1.0: "LIMIT_UP_BLOCKED", 2.0: "LIMIT_DOWN_BLOCKED", 3.0: "CASH_DEFICIT"}
        orders_list = []
        total_turnover_val = 0.0
        total_fees_paid = 0.0

        for row in orders_rec:
            t_idx, s_idx, side, qty, px, fee, code = row
            status_str = status_map.get(code, "UNKNOWN")
            dt = timeline[int(t_idx)]
            sym = symbols[int(s_idx)]
            side_str = "BUY" if side > 0 else "SELL"
            orders_list.append({
                "trade_date": dt,
                "symbol": sym,
                "side": side_str,
                "quantity": qty,
                "price": px,
                "fee": fee,
                "status": status_str,
            })
            if status_str == "FILLED":
                total_turnover_val += qty * px
                total_fees_paid += fee

        orders_df = pd.DataFrame(orders_list) if orders_list else pd.DataFrame(columns=[
            "trade_date", "symbol", "side", "quantity", "price", "fee", "status"
        ])

        avg_equity = equity_series.mean()
        annualized_turnover = (total_turnover_val / max(avg_equity, 1.0)) * (252.0 / max(n_days, 1))

        metrics = {
            "initial_cash": float(self.assumptions.initial_cash),
            "final_equity": float(equity_series.iloc[-1]),
            "total_return": float(total_ret),
            "annualized_return": float(annualized_ret),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "annualized_turnover": float(annualized_turnover),
            "total_fees_paid": float(total_fees_paid),
            "filled_orders_count": int((orders_df["status"] == "FILLED").sum()) if not orders_df.empty else 0,
            "blocked_orders_count": int((orders_df["status"] != "FILLED").sum()) if not orders_df.empty else 0,
        }

        return VectorBTResult(
            timeline=timeline,
            symbols=symbols,
            equity_series=equity_series,
            cash_series=cash_series,
            positions_df=positions_df,
            orders_df=orders_df,
            metrics=metrics,
        )
