"""Conditional overnight-return analysis.

对隔夜收益 overnight[t] (Close_{t-1} -> Open_t), 决策时点是 t-1 收盘,
条件变量都用 t-1 收盘时已知的信息:
  - 当日日内收益 intraday[t-1] (开盘->收盘, 即你说的"日间盘涨跌")
  - 前一日全日收益 total[t-1]
  - 前一周收益 ret5[t-1] (过去5个交易日累计)
  - 前一夜隔夜收益 overnight[t-1]
"""
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
pd.set_option("display.width", 200)

def bucket_stats(overnight, mask, name):
    x = overnight[mask].dropna()
    if len(x) < 30:
        return None
    t, _ = stats.ttest_1samp(x, 0)
    return {"bucket": name, "n": len(x),
            "win%": round((x > 0).mean() * 100, 1),
            "mean_bp": round(x.mean() * 1e4, 2),
            "ann%": round(((1 + x.mean()) ** 252 - 1) * 100, 1),
            "t": round(t, 2)}

def run(ticker):
    r = pd.read_csv(f"{OUT}/{ticker}_legs.csv", index_col=0, parse_dates=True)
    df = pd.DataFrame({
        "on": r["overnight"],                       # 目标: 今晚的隔夜收益
        "intra_prev": r["intraday"].shift(1),       # 当日日内 (t-1)
        "total_prev": r["total"].shift(1),          # 前一日全日
        "on_prev": r["overnight"].shift(1),         # 前一夜隔夜
        "ret5_prev": (1 + r["total"]).rolling(5).apply(np.prod, raw=True).shift(1) - 1,
    }).dropna()

    conds = []
    conds.append(("无条件基准", pd.Series(True, index=df.index)))
    conds.append(("当日日内上涨", df.intra_prev > 0))
    conds.append(("当日日内下跌", df.intra_prev <= 0))
    conds.append(("前一日全日上涨", df.total_prev > 0))
    conds.append(("前一日全日下跌", df.total_prev <= 0))
    conds.append(("前一夜隔夜上涨", df.on_prev > 0))
    conds.append(("前一夜隔夜下跌", df.on_prev <= 0))
    conds.append(("前一周上涨", df.ret5_prev > 0))
    conds.append(("前一周下跌", df.ret5_prev <= 0))
    q = pd.qcut(df.intra_prev, 5, labels=False)
    for k in range(5):
        conds.append((f"当日日内五分位Q{k+1}" + ("(最跌)" if k == 0 else "(最涨)" if k == 4 else ""), q == k))
    q5 = pd.qcut(df.ret5_prev, 5, labels=False)
    for k in range(5):
        conds.append((f"前一周五分位Q{k+1}" + ("(最跌)" if k == 0 else "(最涨)" if k == 4 else ""), q5 == k))

    def table(sub):
        rows = [bucket_stats(sub.on, m.reindex(sub.index, fill_value=False), n)
                for n, m in conds]
        return pd.DataFrame([r for r in rows if r])

    print(f"\n================ {ticker} 全样本 2000-2026 ================")
    print(table(df).to_string(index=False))
    a, b = df.loc[:"2012"], df.loc["2013":]
    print(f"\n---- {ticker} 前半样本 2000-2012 ----")
    print(table(a).to_string(index=False))
    print(f"\n---- {ticker} 后半样本 2013-2026 ----")
    print(table(b).to_string(index=False))

for t in ["QQQ", "SPY"]:
    run(t)
