"""Overnight vs intraday return decomposition for SPY/QQQ.

overnight = Open_t / Close_{t-1} - 1   (夜盘+盘前, 即隔夜段)
intraday  = Close_t / Open_t - 1       (正式开盘到收盘)
total     = Close_t / Close_{t-1} - 1
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)

def decompose(ticker):
    df = pd.read_csv(f"{OUT}/{ticker}.csv", index_col=0, parse_dates=True)
    r = pd.DataFrame(index=df.index)
    r["overnight"] = df["Open"] / df["Close"].shift(1) - 1
    r["intraday"] = df["Close"] / df["Open"] - 1
    r["total"] = df["Close"] / df["Close"].shift(1) - 1
    return r.dropna()

def summarize(x: pd.Series):
    t, p = stats.ttest_1samp(x, 0)
    ann = (1 + x).prod() ** (252 / len(x)) - 1
    return {
        "mean_bp": round(x.mean() * 1e4, 2),          # 日均, 基点
        "ann_ret_pct": round(ann * 100, 2),           # 年化复利
        "vol_bp": round(x.std() * 1e4, 1),
        "win_rate_pct": round((x > 0).mean() * 100, 1),
        "t_stat": round(t, 2),
        "p_value": round(p, 5),
        "sharpe": round(x.mean() / x.std() * np.sqrt(252), 2),
        "worst_day_pct": round(x.min() * 100, 2),
        "best_day_pct": round(x.max() * 100, 2),
    }

results = {}
for ticker in ["SPY", "QQQ"]:
    r = decompose(ticker)
    res = {"full": {}, "recent": {}, "yearly": {}}
    for leg in ["overnight", "intraday", "total"]:
        res["full"][leg] = summarize(r[leg])
        res["recent"][leg] = summarize(r[leg].loc["2020":])
    # yearly cumulative return per leg
    for year, grp in r.groupby(r.index.year):
        res["yearly"][int(year)] = {
            leg: round(((1 + grp[leg]).prod() - 1) * 100, 1)
            for leg in ["overnight", "intraday"]
        }
    # cumulative curves (for charting)
    cum = (1 + r).cumprod()
    cum.to_csv(f"{OUT}/{ticker}_cum.csv")
    r.to_csv(f"{OUT}/{ticker}_legs.csv")
    results[ticker] = res

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1)

for ticker, res in results.items():
    print(f"\n===== {ticker} 2000-2026 全样本 =====")
    print(pd.DataFrame(res["full"]).T.to_string())
    print(f"----- {ticker} 2020 至今 -----")
    print(pd.DataFrame(res["recent"]).T.to_string())
