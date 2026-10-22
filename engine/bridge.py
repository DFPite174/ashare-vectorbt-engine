"""
双引擎协同桥接器 (Dual-Engine Bridge)
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from engine.contracts import ExecutionAssumptions, TargetWeights, DailyNAV, Fill, OrderSide
from engine.market_rules import round_order, block_reason
from engine.golden_ledger import GoldenLedger
from engine.ashare_sim import EnhancedVectorBTEngine, VectorBTResult
from engine.attribution import DualEngineAttributionReporter, AttributionMetrics


@dataclass
class DualEngineRunResult:
    vbt_result: VectorBTResult
    nt_equity_series: pd.Series
    nt_orders_df: pd.DataFrame
    nt_metrics: Dict[str, Any]
    attribution: AttributionMetrics
    report_md: str


class DualEngineBridge:
    def __init__(self, assumptions: ExecutionAssumptions | None = None):
        self.assumptions = assumptions or ExecutionAssumptions()
        self.vbt_engine = EnhancedVectorBTEngine(self.assumptions)
        self.reporter = DualEngineAttributionReporter()

    def run_nautilus_ledger(
        self,
        bars_df: pd.DataFrame,
        target_weights_df: pd.DataFrame,
        rebalance_mask: Optional[pd.Series] = None,
        corporate_actions_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.Series, pd.DataFrame, Dict[str, Any]]:
        pivoted_close = bars_df.pivot(index="trade_date", columns="symbol", values="close").ffill().bfill()
        timeline = sorted(list(pivoted_close.index))
        symbols = sorted(list(pivoted_close.columns))

        aligned_weights = target_weights_df.reindex(index=timeline, columns=symbols).fillna(0.0)
        if rebalance_mask is None:
            weight_diff = (aligned_weights.diff().abs().sum(axis=1) > 1e-4)
            weight_diff.iloc[0] = (aligned_weights.iloc[0].sum() > 1e-4)
            rebalance_bool = weight_diff
        else:
            rebalance_bool = rebalance_mask.reindex(timeline).fillna(False)

        init_cash = Decimal(str(self.assumptions.initial_cash))
        ledger = GoldenLedger(initial_cash=init_cash)

        nav_history: List[float] = []
        orders_history: List[Dict[str, Any]] = []
        total_fees = Decimal("0.0")

        ca_dict: Dict[str, List[Dict[str, Any]]] = {}
        if corporate_actions_df is not None and not corporate_actions_df.empty:
            for _, row in corporate_actions_df.iterrows():
                ex_dt = str(row["ex_date"])
                if ex_dt not in ca_dict:
                    ca_dict[ex_dt] = []
                ca_dict[ex_dt].append(row.to_dict())

        prev_close = pivoted_close.shift(1).fillna(pivoted_close)

        for dt in timeline:
            ledger.on_trade_day_start(dt)

            if dt in ca_dict:
                for ca in ca_dict[dt]:
                    sym = ca["symbol"]
                    cash_div = Decimal(str(ca.get("cash_div", 0.0)))
                    if cash_div > 0:
                        ledger.apply_dividend(sym, cash_div)

            if rebalance_bool.loc[dt]:
                weights_today = aligned_weights.loc[dt]
                close_prices_today = pivoted_close.loc[dt]

                curr_nav = ledger.cash
                for s, q in ledger.positions.items():
                    if q > 0 and s in close_prices_today:
                        curr_nav += q * Decimal(str(close_prices_today[s]))

                investable = curr_nav * Decimal("0.995")

                sell_orders = []
                buy_orders = []

                for s in symbols:
                    w = Decimal(str(weights_today[s]))
                    px = Decimal(str(close_prices_today[s]))
                    held = ledger.positions.get(s, Decimal("0.0"))

                    if w > 0:
                        target_val = investable * w
                        target_qty = (target_val / px) // Decimal("100") * Decimal("100")
                    else:
                        target_qty = Decimal("0.0")

                    delta = target_qty - held
                    rule = {
                        "max_order": 1000000,
                        "buy_step": 100,
                        "buy_min": 100,
                        "sell_step": 100,
                        "sell_min": 100,
                        "trade_status": "OPEN",
                        "limit_state": "BOUNDED",
                        "up_limit": float(round(float(prev_close.loc[dt, s]) * (1.20 if s.startswith("688") or s.startswith("300") else 1.10), 2)),
                        "down_limit": float(round(float(prev_close.loc[dt, s]) * (0.80 if s.startswith("688") or s.startswith("300") else 0.90), 2)),
                    }
                    rounded = round_order(float(delta), float(held), rule)
                    if rounded < 0:
                        sell_orders.append((s, abs(rounded), px, rule))
                    elif rounded > 0:
                        buy_orders.append((s, rounded, px, rule))

                # 卖单
                for s, qty_int, px, rule in sell_orders:
                    bar_dict = {"close": float(px), "volume": 1000000}
                    reason = block_reason(rule, bar_dict, "sell", float(px))
                    if reason:
                        orders_history.append({"trade_date": dt, "symbol": s, "side": "SELL", "quantity": qty_int, "price": float(px), "fee": 0.0, "status": reason})
                        continue

                    tradable = ledger.tradable_positions.get(s, Decimal("0.0"))
                    fill_qty = min(Decimal(str(qty_int)), tradable)
                    if fill_qty <= 0:
                        continue

                    eff_px = px * (Decimal("1.0") - Decimal(str(self.assumptions.slippage)) * Decimal("0.0001"))
                    trade_val = fill_qty * eff_px
                    comm = max(Decimal(str(self.assumptions.min_commission)), trade_val * Decimal(str(self.assumptions.commission_rate)))
                    stamp = trade_val * Decimal(str(self.assumptions.stamp_duty_rate))
                    transfer = trade_val * Decimal(str(self.assumptions.transfer_fee_rate))
                    fee = (comm + stamp + transfer).quantize(Decimal("0.01"))
                    total_fees += fee

                    fill = Fill(
                        fill_id=f"fill_{dt}_{s}_sell",
                        symbol=s,
                        side=OrderSide.SELL,
                        quantity=fill_qty,
                        price=eff_px,
                        fee=fee,
                        timestamp=datetime.strptime(dt, "%Y-%m-%d"),
                    )
                    ledger.apply_fill(fill, dt)
                    orders_history.append({"trade_date": dt, "symbol": s, "side": "SELL", "quantity": float(fill_qty), "price": float(eff_px), "fee": float(fee), "status": "FILLED"})

                # 买单
                for s, qty_int, px, rule in buy_orders:
                    bar_dict = {"close": float(px), "volume": 1000000}
                    reason = block_reason(rule, bar_dict, "buy", float(px))
                    if reason:
                        orders_history.append({"trade_date": dt, "symbol": s, "side": "BUY", "quantity": qty_int, "price": float(px), "fee": 0.0, "status": reason})
                        continue

                    eff_px = px * (Decimal("1.0") + Decimal(str(self.assumptions.slippage)) * Decimal("0.0001"))
                    want_qty = Decimal(str(qty_int))
                    pre_trade_val = want_qty * eff_px
                    comm = max(Decimal(str(self.assumptions.min_commission)), pre_trade_val * Decimal(str(self.assumptions.commission_rate)))
                    transfer = pre_trade_val * Decimal(str(self.assumptions.transfer_fee_rate))
                    pre_fee = (comm + transfer).quantize(Decimal("0.01"))

                    total_need = pre_trade_val + pre_fee
                    if total_need > ledger.cash:
                        avail = ledger.cash - Decimal("1000.0")
                        if avail <= 0:
                            orders_history.append({"trade_date": dt, "symbol": s, "side": "BUY", "quantity": qty_int, "price": float(eff_px), "fee": 0.0, "status": "CASH_DEFICIT"})
                            continue
                        adj_qty = (avail / (eff_px * Decimal("1.0004"))) // Decimal("100") * Decimal("100")
                        if adj_qty < Decimal("100"):
                            orders_history.append({"trade_date": dt, "symbol": s, "side": "BUY", "quantity": qty_int, "price": float(eff_px), "fee": 0.0, "status": "CASH_DEFICIT"})
                            continue
                        fill_qty = adj_qty
                        val = fill_qty * eff_px
                        comm = max(Decimal(str(self.assumptions.min_commission)), val * Decimal(str(self.assumptions.commission_rate)))
                        transfer = val * Decimal(str(self.assumptions.transfer_fee_rate))
                        fee = (comm + transfer).quantize(Decimal("0.01"))
                    else:
                        fill_qty = want_qty
                        fee = pre_fee

                    total_fees += fee
                    fill = Fill(
                        fill_id=f"fill_{dt}_{s}_buy",
                        symbol=s,
                        side=OrderSide.BUY,
                        quantity=fill_qty,
                        price=eff_px,
                        fee=fee,
                        timestamp=datetime.strptime(dt, "%Y-%m-%d"),
                    )
                    ledger.apply_fill(fill, dt)
                    orders_history.append({"trade_date": dt, "symbol": s, "side": "BUY", "quantity": float(fill_qty), "price": float(eff_px), "fee": float(fee), "status": "FILLED"})

            px_dict = {s: Decimal(str(pivoted_close.loc[dt, s])) for s in symbols}
            daily_nav_obj = ledger.mark_to_market(dt, px_dict)
            nav_history.append(float(daily_nav_obj.total_equity))

        nt_equity = pd.Series(nav_history, index=timeline, name="nt_equity")
        orders_df = pd.DataFrame(orders_history) if orders_history else pd.DataFrame(columns=["trade_date", "symbol", "side", "quantity", "price", "fee", "status"])

        metrics = {
            "initial_cash": float(init_cash),
            "final_equity": float(nav_history[-1]),
            "total_return": (nav_history[-1] / nav_history[0]) - 1.0,
            "total_fees_paid": float(total_fees),
            "filled_orders_count": int((orders_df["status"] == "FILLED").sum()) if not orders_df.empty else 0,
        }

        return nt_equity, orders_df, metrics

    def run_dual(
        self,
        bars_df: pd.DataFrame,
        target_weights_df: pd.DataFrame,
        strategy_name: str = "A_Share_Strategy",
        rebalance_mask: Optional[pd.Series] = None,
        corporate_actions_df: Optional[pd.DataFrame] = None,
        report_output_path: Optional[str] = None,
    ) -> DualEngineRunResult:
        vbt_res = self.vbt_engine.run(
            bars_df=bars_df,
            target_weights_df=target_weights_df,
            rebalance_mask=rebalance_mask,
            corporate_actions_df=corporate_actions_df,
        )

        nt_equity, nt_orders, nt_metrics = self.run_nautilus_ledger(
            bars_df=bars_df,
            target_weights_df=target_weights_df,
            rebalance_mask=rebalance_mask,
            corporate_actions_df=corporate_actions_df,
        )

        attribution = self.reporter.attribute(
            vbt_result=vbt_res,
            nt_equity_series=nt_equity,
            nt_orders_df=nt_orders,
            nt_metrics=nt_metrics,
        )

        report_md = self.reporter.generate_markdown_report(
            metrics=attribution,
            strategy_name=strategy_name,
            output_path=report_output_path,
        )

        return DualEngineRunResult(
            vbt_result=vbt_res,
            nt_equity_series=nt_equity,
            nt_orders_df=nt_orders,
            nt_metrics=nt_metrics,
            attribution=attribution,
            report_md=report_md,
        )
