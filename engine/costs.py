"""
A 股交易费税与滑点模型 (AShare Cost & Slippage Model)
支持 Numba JIT 高性能编译与向量化计算：
1. 券商佣金: 万三 (单笔最低 5 元)
2. 印花税: 千分之 0.5 (卖方单边)
3. 过户费: 十万分之一 (双边)
4. 滑点模型: 基点滑点 (BPS)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import numba as nb


@dataclass(frozen=True)
class AShareCostConfig:
    commission_rate: float = 0.0003
    min_commission: float = 5.0
    stamp_duty_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_bps: float = 5.0

    @property
    def slippage_rate(self) -> float:
        return self.slippage_bps * 1e-4


@nb.njit(fastmath=True)
def calc_trade_fee_nb(
    side: int,  # 1 for BUY, -1 for SELL
    price: float,
    qty: float,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
    stamp_duty_rate: float = 0.0005,
    transfer_fee_rate: float = 0.00001,
) -> float:
    if qty <= 0 or price <= 0:
        return 0.0

    trade_value = price * qty
    comm = trade_value * commission_rate
    if comm < min_commission:
        comm = min_commission

    transfer = trade_value * transfer_fee_rate
    stamp = 0.0
    if side == -1:
        stamp = trade_value * stamp_duty_rate

    return comm + transfer + stamp


@nb.njit(fastmath=True)
def calc_effective_price_nb(
    side: int,  # 1 for BUY, -1 for SELL
    price: float,
    slippage_rate: float = 0.0005,
) -> float:
    if side == 1:
        return price * (1.0 + slippage_rate)
    elif side == -1:
        return price * (1.0 - slippage_rate)
    return price


@nb.njit(fastmath=True)
def calc_market_impact_nb(
    trade_amount: float,
    adv: float,
    volatility: float,
    eta: float = 0.10,
) -> float:
    """
    平方根流动性冲击成本模型:
    Impact Cost = eta * sigma * sqrt(Trade Amount / ADV)
    """
    if adv <= 1000.0 or trade_amount <= 0.0 or volatility <= 0.0:
        return 0.0
    participation = trade_amount / adv
    if participation > 1.0:
        participation = 1.0
    return eta * volatility * np.sqrt(participation)


@nb.njit(fastmath=True)
def calc_effective_price_with_impact_nb(
    side: int,  # 1 for BUY, -1 for SELL
    price: float,
    slippage_rate: float = 0.0005,
    impact_rate: float = 0.0,
) -> float:
    """综合固定滑点与非线性流动性冲击的最终成交价"""
    total_friction = slippage_rate + impact_rate
    if side == 1:
        return price * (1.0 + total_friction)
    elif side == -1:
        return price * (1.0 - total_friction)
    return price


class AShareCostModel:
    def __init__(self, config: AShareCostConfig | None = None):
        self.config = config or AShareCostConfig()

    def calculate_cost(self, side: str | int, price: float, qty: float) -> tuple[float, float]:
        s = 1 if side in ("BUY", "buy", 1) else -1
        eff_px = calc_effective_price_nb(s, float(price), self.config.slippage_rate)
        fee = calc_trade_fee_nb(
            side=s,
            price=eff_px,
            qty=float(qty),
            commission_rate=self.config.commission_rate,
            min_commission=self.config.min_commission,
            stamp_duty_rate=self.config.stamp_duty_rate,
            transfer_fee_rate=self.config.transfer_fee_rate,
        )
        return fee, eff_px
