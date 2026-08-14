"""Overnight-only strategy backtest with transaction-cost sensitivity.

策略: 每日收盘 MOC 买入, 次日开盘 MOO 卖出, 持有隔夜段。
成本: 每边 cost_bp 基点 (滑点+点差+佣金), 每天两边。
对比: 日内策略(开盘买收盘卖)、买入持有。
"""
import json
import numpy as np
import pandas as pd

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
COSTS_BP = [0, 0.5, 1, 2, 5]

def max_drawdown(cum):
    return (cum / cum.cummax() - 1).min()

out = {}
for ticker in ["SPY", "QQQ"]:
    r = pd.read_csv(f"{OUT}/{ticker}_legs.csv", index_col=0, parse_dates=True)
    n_years = len(r) / 252
    rows = []
    for cost_bp in COSTS_BP:
        c = cost_bp / 1e4
        net = (1 + r["overnight"]) * (1 - c) ** 2 - 1
        cum = (1 + net).cumprod()
        rows.append({
            "cost_bp_per_side": cost_bp,
            "ann_ret_pct": round((cum.iloc[-1] ** (1 / n_years) - 1) * 100, 2),
            "sharpe": round(net.mean() / net.std() * np.sqrt(252), 2),
            "max_dd_pct": round(max_drawdown(cum) * 100, 1),
            "final_multiple": round(cum.iloc[-1], 2),
        })
    bh = (1 + r["total"]).cumprod()
    intra = (1 + r["intraday"]).cumprod()
    out[ticker] = {
        "overnight_cost_table": rows,
        "buy_hold": {
            "ann_ret_pct": round((bh.iloc[-1] ** (1 / n_years) - 1) * 100, 2),
            "sharpe": round(r["total"].mean() / r["total"].std() * np.sqrt(252), 2),
            "max_dd_pct": round(max_drawdown(bh) * 100, 1),
            "final_multiple": round(bh.iloc[-1], 2),
        },
        "intraday_only": {
            "ann_ret_pct": round((intra.iloc[-1] ** (1 / n_years) - 1) * 100, 2),
            "max_dd_pct": round(max_drawdown(intra) * 100, 1),
            "final_multiple": round(intra.iloc[-1], 2),
        },
        "overnight_mdd_gross_pct": round(
            max_drawdown((1 + r["overnight"]).cumprod()) * 100, 1),
    }

with open(f"{OUT}/backtest.json", "w") as f:
    json.dump(out, f, indent=1)

for ticker, res in out.items():
    print(f"\n===== {ticker} 隔夜策略 成本敏感性 (2000-2026) =====")
    print(pd.DataFrame(res["overnight_cost_table"]).to_string(index=False))
    print("买入持有:", res["buy_hold"])
    print("纯日内:  ", res["intraday_only"])
