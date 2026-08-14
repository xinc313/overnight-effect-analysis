"""Additional factors for the overnight signal: VIX level, weekday, earnings months.

所有条件均使用决策时点(t-1 收盘)已知信息:
  VIX 用 t-1 收盘值; 星期几指 t 日(明早开盘那天); 财报月为 1/4/7/10。
"""
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
pd.set_option("display.width", 220)

r = pd.read_csv(f"{OUT}/QQQ_legs.csv", index_col=0, parse_dates=True)
on = r["overnight"]
prev = on.shift(1)
sig = prev <= 0

vix = pd.read_csv(f"{OUT}/VIX.csv", index_col=0, parse_dates=True)["Close"]
vix_prev = vix.reindex(on.index).ffill().shift(1)

def bstats(x):
    x = pd.Series(x).dropna()
    if len(x) < 30: return None
    t, _ = stats.ttest_1samp(x, 0)
    return {"n": int(len(x)), "win%": round((x > 0).mean() * 100, 1),
            "mean_bp": round(x.mean() * 1e4, 2), "t": round(t, 2)}

def table(conds, base_mask=None, label=""):
    rows = {}
    for name, m in conds:
        mm = m if base_mask is None else (m & base_mask)
        s = bstats(on[mm])
        if s: rows[name] = s
    print(f"\n--- {label} ---")
    print(pd.DataFrame(rows).T.to_string())

# ========== 1. VIX ==========
vix_conds = [("VIX<15", vix_prev < 15), ("VIX 15-20", (vix_prev >= 15) & (vix_prev < 20)),
             ("VIX 20-30", (vix_prev >= 20) & (vix_prev < 30)), ("VIX>30", vix_prev >= 30)]
print("===== 1. VIX 水平 (前日收盘) =====")
table(vix_conds, None, "全部夜晚")
table(vix_conds, sig, "仅信号夜(前夜隔夜下跌后)")

# ========== 2. 星期几 ==========
wd = pd.Series(on.index.dayofweek, index=on.index)
names = ["周一(含周末)", "周二", "周三", "周四", "周五"]
wd_conds = [(names[i], wd == i) for i in range(5)]
print("\n===== 2. 星期几 (隔夜结束的那天早晨) =====")
table(wd_conds, None, "全部夜晚")
table(wd_conds, sig, "仅信号夜")

# ========== 3. 财报季 ==========
mo = pd.Series(on.index.month, index=on.index)
er = mo.isin([1, 4, 7, 10])
er_conds = [("财报月 1/4/7/10", er), ("非财报月", ~er)]
print("\n===== 3. 财报季(粗略按月份) =====")
table(er_conds, None, "全部夜晚")
table(er_conds, sig, "仅信号夜")

# ========== 4. 对显著候选做前后半样本检验 ==========
print("\n===== 4. 稳定性检验: 信号夜内部, 前后半样本 =====")
h1 = on.index <= "2012-12-31"
h2 = on.index > "2012-12-31"
cands = [("VIX>30 信号夜", (vix_prev >= 30) & sig),
         ("VIX<15 信号夜", (vix_prev < 15) & sig),
         ("周一 信号夜", (wd == 0) & sig),
         ("周三 信号夜", (wd == 2) & sig),
         ("财报月 信号夜", er & sig),
         ("非财报月 信号夜", (~er) & sig)]
rows = {}
for name, m in cands:
    a, b = bstats(on[m & h1]), bstats(on[m & h2])
    if a and b:
        rows[name] = {"00-12 win%": a["win%"], "00-12 bp": a["mean_bp"], "00-12 t": a["t"],
                      "13-26 win%": b["win%"], "13-26 bp": b["mean_bp"], "13-26 t": b["t"]}
print(pd.DataFrame(rows).T.to_string())
