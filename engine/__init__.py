"""
ashare-vectorbt-engine
A 股工业级增强型 VectorBT 极速回测与双引擎协同系统
"""

from engine.contracts import ExecutionAssumptions, TargetWeights, DailyNAV, Fill, OrderSide
from engine.costs import AShareCostConfig, AShareCostModel, calc_trade_fee_nb, calc_effective_price_nb
from engine.sizing import SizingConfig, DynamicSizer, round_order_qty_nb, allocate_target_shares_nb
from engine.ashare_sim import EnhancedVectorBTEngine, VectorBTResult
from engine.grid_optimizer import GridOptimizer, GridOptimizationResult
from engine.attribution import DualEngineAttributionReporter, AttributionMetrics
from engine.bridge import DualEngineBridge, DualEngineRunResult
from engine.golden_ledger import GoldenLedger
from engine.market_rules import round_order, block_reason

__version__ = "1.0.0"

__all__ = [
    "ExecutionAssumptions",
    "TargetWeights",
    "DailyNAV",
    "Fill",
    "OrderSide",
    "AShareCostConfig",
    "AShareCostModel",
    "calc_trade_fee_nb",
    "calc_effective_price_nb",
    "SizingConfig",
    "DynamicSizer",
    "round_order_qty_nb",
    "allocate_target_shares_nb",
    "EnhancedVectorBTEngine",
    "VectorBTResult",
    "GridOptimizer",
    "GridOptimizationResult",
    "DualEngineAttributionReporter",
    "AttributionMetrics",
    "DualEngineBridge",
    "DualEngineRunResult",
    "GoldenLedger",
    "round_order",
    "block_reason",
]
