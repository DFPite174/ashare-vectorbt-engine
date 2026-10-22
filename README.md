# ashare-vectorbt-engine

> **A 股工业级增强型 VectorBT 极速回测与双引擎协同系统**  
> *A-Share Enhanced High-Throughput Vectorized Backtesting Engine with Numba JIT Microstructural Constraints & NautilusTrader Dual-Engine Bridge.*

---

## 🌟 核心亮点 (Core Highlights)

1. **A 股本土微观制度 Numba 纳秒级 JIT 仿真**：
   - **T+1 动态库存追踪**：原生隔离昨日可卖仓位与当日新购锁定仓位，严防虚假同日高频来回套利；
   - **涨跌停板挂单熔断**：涨停日自动阻断买入 (`LIMIT_UP_BLOCKED`)，跌停日自动阻断卖出 (`LIMIT_DOWN_BLOCKED`)；
   - **法定阶梯费税精准扣费**：单笔最低 5 元券商佣金、卖方单边 0.05% 印花税、双边过户费与价格滑点；
   - **整手与零股规整**：主板 100 股整数倍买入截断，清仓零股全卖，动态 0.5% 现金缓冲池杜绝穿仓。
2. **商用 VectorBT Pro 级超维网格优化 (Grid Optimizer)**：
   - 笛卡尔积多维超参数并行扫描（动量周期、持仓 Top-K、调仓频率），单组策略回测平均仅需 **~170ms**，3 秒内完成数十组参数扫描并输出全景排行榜。
3. **双引擎微观结构差异归因中枢 (Dual-Engine Attribution Bridge)**：
   - 以不可变契约 `TargetWeights` 为唯一输入，无缝桥接 VectorBT 高速粗筛与 NautilusTrader / Golden Ledger 事件驱动高保真核验；
   - 定量分解规整差 (`Δ_lot`)、费税差 (`Δ_fee`)、涨跌停差 (`Δ_limit`) 与 T+1 时滞差 (`Δ_t1`)，偏差收敛在 **0.00%**。

---

## 🚀 极速上手 (Quickstart)

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行快速示例
```bash
python examples/quickstart_momentum_sweep.py
```

### 3. 代码示例
```python
from decimal import Decimal
import pandas as pd
from engine import GridOptimizer, DualEngineBridge, ExecutionAssumptions

# 1. 初始化优化器
optimizer = GridOptimizer(ExecutionAssumptions(initial_cash=Decimal("10000000.0")))

# 2. 运行高通量动量策略超参数扫描
opt_res = optimizer.optimize_momentum_strategy(
    bars_df=bars_df,
    lookback_windows=[10, 20, 30],
    top_k_list=[1, 2],
    rebalance_intervals=[5, 10, 20],
)
print("Top 1 参数组合:", opt_res.best_params)

# 3. 双引擎事件驱动协同核验与归因
bridge = DualEngineBridge()
run_res = bridge.run_dual(
    bars_df=bars_df,
    target_weights_df=target_weights,
    strategy_name="A_Share_Momentum",
)
print("一致性裁定:", run_res.attribution.consistency_verdict)
```

---

## 📂 项目结构 (Repository Structure)

```text
ashare-vectorbt-engine/
├── engine/                       # 核心引擎源码
│   ├── contracts.py              # 不可变业务实体契约 (Pydantic v2)
│   ├── costs.py                  # A 股法定费税与滑点 Numba JIT 模型
│   ├── sizing.py                 # 动态整手截断、零股清仓与先卖后买
│   ├── ashare_sim.py             # 高性能 A 股 Numba 向量化仿真内核
│   ├── grid_optimizer.py         # 超维超参数网格扫描优化器
│   ├── golden_ledger.py          # 确定性会计黄金账本
│   ├── market_rules.py           # A 股交易规则与价格边界拦截
│   ├── attribution.py            # 双引擎微观结构差异归因分析器
│   └── bridge.py                 # 双引擎一体化协同桥梁
├── examples/                     # 开箱即用实战用例
│   └── quickstart_momentum_sweep.py
├── tests/                        # 单元测试与一致性测试套件
│   └── test_standalone_engine.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## ⚖️ 开源协议 (License)
MIT License.
