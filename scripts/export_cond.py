"""Conditional analysis chart data + conditional strategy backtest.

策略: 仅当前一夜隔夜收益<0时, 当日收盘买入持有至次日开盘。
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
COSTS_BP = [0, 0.5, 1, 2, 5]

def bstats(x):
    x = x.dropna()
    t, _ = stats.ttest_1samp(x, 0)
    return {"n": len(x), "win": round((x > 0).mean() * 100, 1),
            "bp": round(x.mean() * 1e4, 2), "t": round(t, 2)}

def max_dd(cum):
    return round((cum / cum.cummax() - 1).min() * 100, 1)

data = {}
for ticker in ["QQQ", "SPY"]:
    r = pd.read_csv(f"{OUT}/{ticker}_legs.csv", index_col=0, parse_dates=True)
    df = pd.DataFrame({
        "on": r["overnight"],
        "intra_prev": r["intraday"].shift(1),
        "total_prev": r["total"].shift(1),
        "on_prev": r["overnight"].shift(1),
        "ret5_prev": (1 + r["total"]).rolling(5).apply(np.prod, raw=True).shift(1) - 1,
    }).dropna()

    signconds = [
        ("无条件基准", pd.Series(True, index=df.index)),
        ("前夜隔夜下跌后", df.on_prev <= 0), ("前夜隔夜上涨后", df.on_prev > 0),
        ("前日全日下跌后", df.total_prev <= 0), ("前日全日上涨后", df.total_prev > 0),
        ("当日日内下跌后", df.intra_prev <= 0), ("当日日内上涨后", df.intra_prev > 0),
        ("前一周下跌后", df.ret5_prev <= 0), ("前一周上涨后", df.ret5_prev > 0),
    ]
    periods = {"full": df, "h1": df.loc[:"2012"], "h2": df.loc["2013":]}
    conds = {p: [dict(name=n, **bstats(sub.on[m.reindex(sub.index, fill_value=False)]))
                 for n, m in signconds]
             for p, sub in periods.items()}

    quint = {}
    for key, col in [("intra", "intra_prev"), ("week", "ret5_prev")]:
        q = pd.qcut(df[col], 5, labels=False)
        quint[key] = [dict(name=f"Q{k+1}", **bstats(df.on[q == k])) for k in range(5)]

    # conditional strategy: hold overnight only if prev overnight <= 0
    mask = (df.on_prev <= 0)
    strat = df.on.where(mask, 0.0)
    n_years = len(df) / 252
    trades_per_year = round(mask.sum() / n_years)
    rows = []
    for cbp in COSTS_BP:
        c = cbp / 1e4
        net = np.where(mask, (1 + df.on) * (1 - c) ** 2 - 1, 0.0)
        net = pd.Series(net, index=df.index)
        cum = (1 + net).cumprod()
        rows.append({"cost": cbp,
                     "ann": round((cum.iloc[-1] ** (1 / n_years) - 1) * 100, 2),
                     "sharpe": round(net.mean() / net.std() * np.sqrt(252), 2),
                     "mdd": max_dd(cum), "mult": round(cum.iloc[-1], 2)})

    cum3 = pd.DataFrame({
        "cond": (1 + strat).cumprod(),
        "on": (1 + df.on).cumprod(),
        "bh": (1 + r["total"].reindex(df.index)).cumprod(),
    }).resample("ME").last()

    # share of total overnight log-return captured by conditional nights
    share = round(float(np.log1p(df.on[mask]).sum() / np.log1p(df.on).sum()) * 100)

    data[ticker] = {
        "conds": conds, "quint": quint,
        "strat": {"rows": rows, "trades_per_year": trades_per_year,
                  "pct_nights": round(mask.mean() * 100), "share": share,
                  "gross_mdd": max_dd((1 + strat).cumprod())},
        "curves": {"dates": [d.strftime("%Y-%m") for d in cum3.index],
                   **{k: [round(v, 4) for v in cum3[k]] for k in ["cond", "on", "bh"]}},
    }

with open(f"{OUT}/conddata.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)

for t in ["QQQ", "SPY"]:
    s = data[t]["strat"]
    print(f"\n{t}: 交易{s['pct_nights']}%的夜晚({s['trades_per_year']}次/年), "
          f"捕获{s['share']}%的隔夜对数收益, 毛回撤{s['gross_mdd']}%")
    print(pd.DataFrame(s["rows"]).to_string(index=False))
