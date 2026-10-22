"""
ashare-vectorbt-engine - 基础契约模型 (Contracts)
"""

from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExecutionAssumptions(BaseModel):
    """交易制度假设"""
    model_config = ConfigDict(frozen=True)

    initial_cash: Decimal = Field(default=Decimal("10000000.0"), description="初始资金")
    commission_rate: Decimal = Field(default=Decimal("0.0003"), description="佣金费率 (万三)")
    min_commission: Decimal = Field(default=Decimal("5.0"), description="最低单笔佣金 (5元)")
    stamp_duty_rate: Decimal = Field(default=Decimal("0.0005"), description="卖方单边印花税 (千分之0.5)")
    transfer_fee_rate: Decimal = Field(default=Decimal("0.00001"), description="双边过户费 (十万分之一)")
    slippage: Decimal = Field(default=Decimal("0.01"), description="固定滑点")
    t_plus_one: bool = Field(default=True, description="A 股 T+1 约束")


class TargetWeights(BaseModel):
    """目标权重矩阵"""
    model_config = ConfigDict(frozen=True)

    signal_date: date
    execution_date: date
    weights: Dict[str, float]
    strategy_id: str = "default_strategy"


class Fill(BaseModel):
    """订单成交实体"""
    model_config = ConfigDict(frozen=True)

    fill_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    fee: Decimal
    timestamp: datetime


class DailyNAV(BaseModel):
    """日度盯市估值记录"""
    model_config = ConfigDict(frozen=True)

    date: date
    cash: Decimal
    positions_value: Decimal
    total_equity: Decimal
    holdings: Dict[str, Decimal] = Field(default_factory=dict)
    tradable_holdings: Dict[str, Decimal] = Field(default_factory=dict)
