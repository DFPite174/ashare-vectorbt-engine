"""
A 股市场交易制度与规整规则 (Market Rules)
"""

from __future__ import annotations
from typing import Dict, Any, Optional


def round_order(delta: float, held: float, rule: Dict[str, Any]) -> int:
    max_order = rule.get("max_order", 1000000)
    buy_step = rule.get("buy_step", 100)
    buy_min = rule.get("buy_min", 100)
    sell_step = rule.get("sell_step", 100)
    sell_min = rule.get("sell_min", 100)

    if delta > 0:
        qty = min(int(delta // buy_step) * buy_step, max_order)
        return qty if qty >= buy_min else 0

    wanted = min(int(abs(delta)), int(held), max_order)
    if not wanted:
        return 0

    if wanted == int(held):
        return -wanted

    qty = int(wanted // sell_step) * sell_step
    return -qty if qty >= sell_min else 0


def block_reason(
    rule: Dict[str, Any],
    bar: Dict[str, Any],
    side: str,
    execution_price: Optional[float] = None
) -> Optional[str]:
    status = rule.get("trade_status")
    if status != "OPEN":
        return status
    if bar.get("volume", 0) <= 0:
        return "ZERO_VOLUME"

    if rule.get("limit_state") == "BOUNDED":
        lo, hi = rule["down_limit"], rule["up_limit"]
        close_p = bar.get("close", 0.0)
        if side.lower() == "buy" and close_p >= hi - 1e-8:
            return "LIMIT_UP_CLOSE"
        if side.lower() == "sell" and close_p <= lo + 1e-8:
            return "LIMIT_DOWN_CLOSE"

    return None
