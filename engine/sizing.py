"""
A 股动态仓位与整手规整器 (Sizing & Lot Rounding)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import numba as nb


@dataclass(frozen=True)
class SizingConfig:
    cash_buffer_ratio: float = 0.005
    lot_size: int = 100
    min_buy_qty: int = 100
    max_order_qty: int = 1000000
    allow_odd_lot_sell: bool = True
    min_trade_amount: float = 1000.0


@nb.njit(fastmath=True)
def round_order_qty_nb(
    target_shares: float,
    current_shares: float,
    lot_size: int = 100,
    min_buy_qty: int = 100,
    max_order_qty: int = 1000000,
    allow_odd_lot_sell: bool = True,
    price: float = 0.0,
    min_trade_amount: float = 0.0,
) -> float:
    delta = target_shares - current_shares

    if delta > 0.0:
        lots = int(delta // lot_size)
        qty = lots * lot_size
        if qty < min_buy_qty:
            return 0.0
        if min_trade_amount > 0.0 and price > 0.0 and (qty * price < min_trade_amount):
            return 0.0
        if qty > max_order_qty:
            qty = max_order_qty
        return float(qty)

    elif delta < 0.0:
        wanted_sell = abs(delta)
        available = current_shares
        actual_sell = min(wanted_sell, available)

        if actual_sell <= 0.0:
            return 0.0

        if allow_odd_lot_sell and abs(actual_sell - current_shares) < 1e-4:
            qty = min(int(actual_sell), max_order_qty)
            return float(-qty)

        lots = int(actual_sell // lot_size)
        qty = lots * lot_size
        if qty < lot_size:
            return 0.0
        if min_trade_amount > 0.0 and price > 0.0 and (qty * price < min_trade_amount):
            return 0.0
        if qty > max_order_qty:
            qty = max_order_qty
        return float(-qty)

    return 0.0


@nb.njit(fastmath=True)
def allocate_target_shares_nb(
    weights: np.ndarray,
    prices: np.ndarray,
    current_shares: np.ndarray,
    tradable_shares: np.ndarray,
    current_cash: float,
    cash_buffer_ratio: float = 0.005,
    lot_size: int = 100,
    min_buy_qty: int = 100,
    max_order_qty: int = 1000000,
    min_trade_amount: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_symbols = len(weights)
    order_deltas = np.zeros(n_symbols, dtype=np.float64)
    target_shares = np.zeros(n_symbols, dtype=np.float64)

    pos_val = 0.0
    for i in range(n_symbols):
        if prices[i] > 0.0:
            pos_val += current_shares[i] * prices[i]
    total_equity = current_cash + pos_val

    if total_equity <= 0.0:
        return order_deltas, target_shares

    investable_equity = total_equity * (1.0 - cash_buffer_ratio)

    raw_deltas = np.zeros(n_symbols, dtype=np.float64)
    for i in range(n_symbols):
        w = weights[i]
        px = prices[i]
        if px > 0.0 and w > 0.0:
            desired_val = investable_equity * w
            ideal_shares = desired_val / px
            t_shares = float(int(ideal_shares // lot_size) * lot_size)
            target_shares[i] = t_shares
            raw_deltas[i] = t_shares - current_shares[i]
        elif w <= 0.0:
            target_shares[i] = 0.0
            raw_deltas[i] = -current_shares[i]

    # 先卖
    simulated_cash = current_cash
    for i in range(n_symbols):
        if raw_deltas[i] < 0.0 and prices[i] > 0.0:
            max_sellable = tradable_shares[i]
            wanted_sell = abs(raw_deltas[i])
            actual_sell = min(wanted_sell, max_sellable)

            if actual_sell > 0.0:
                if abs(actual_sell - current_shares[i]) < 1e-4:
                    order_qty = float(min(int(actual_sell), max_order_qty))
                else:
                    lots = int(actual_sell // lot_size)
                    order_qty = float(min(lots * lot_size, max_order_qty))

                if order_qty > 0.0:
                    order_deltas[i] = -order_qty
                    sell_proceeds = order_qty * prices[i] * (1.0 - 0.0008)
                    simulated_cash += sell_proceeds

    # 后买
    for i in range(n_symbols):
        if raw_deltas[i] > 0.0 and prices[i] > 0.0:
            px = prices[i]
            unit_cost = px * (1.0 + 0.0005 + 0.0003)
            max_spend = simulated_cash - (total_equity * cash_buffer_ratio * 0.5)
            if max_spend <= 0.0:
                continue

            max_affordable_shares = max_spend / unit_cost
            wanted_buy = raw_deltas[i]
            can_buy = min(wanted_buy, max_affordable_shares)

            lots = int(can_buy // lot_size)
            buy_qty = float(min(lots * lot_size, max_order_qty))

            if buy_qty >= min_buy_qty:
                cost = buy_qty * unit_cost
                if cost <= simulated_cash:
                    order_deltas[i] = buy_qty
                    simulated_cash -= cost

    return order_deltas, target_shares


class DynamicSizer:
    def __init__(self, config: SizingConfig | None = None):
        self.config = config or SizingConfig()

    def calculate_orders(
        self,
        weights: Dict[str, float],
        prices: Dict[str, float],
        current_shares: Dict[str, float],
        tradable_shares: Dict[str, float],
        current_cash: float,
    ) -> Dict[str, float]:
        symbols = sorted(list(set(list(weights.keys()) + list(prices.keys()) + list(current_shares.keys()))))
        n = len(symbols)

        w_arr = np.array([weights.get(s, 0.0) for s in symbols], dtype=np.float64)
        px_arr = np.array([prices.get(s, 0.0) for s in symbols], dtype=np.float64)
        cur_arr = np.array([current_shares.get(s, 0.0) for s in symbols], dtype=np.float64)
        tra_arr = np.array([tradable_shares.get(s, 0.0) for s in symbols], dtype=np.float64)

        deltas, _ = allocate_target_shares_nb(
            weights=w_arr,
            prices=px_arr,
            current_shares=cur_arr,
            tradable_shares=tra_arr,
            current_cash=float(current_cash),
            cash_buffer_ratio=self.config.cash_buffer_ratio,
            lot_size=self.config.lot_size,
            min_buy_qty=self.config.min_buy_qty,
            max_order_qty=self.config.max_order_qty,
            min_trade_amount=self.config.min_trade_amount,
        )

        return {symbols[i]: float(deltas[i]) for i in range(n) if abs(deltas[i]) > 0.0}
