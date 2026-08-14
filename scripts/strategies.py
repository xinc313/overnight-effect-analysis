"""QQQ conditional-overnight strategy variants, focused on the losing-entry case.

所有变体: 信号 = 前夜隔夜<=0; 收盘买入次日开盘卖出; 单边成本1bp*仓位。
"""
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
pd.set_option("display.width", 220)
C = 1e-4  # 1bp per side

r = pd.read_csv(f"{OUT}/QQQ_legs.csv", index_col=0, parse_dates=True)
on = r["overnight"]
prev = on.shift(1)

# streak of consecutive down overnights known at decision time
k = np.zeros(len(on), dtype=int)
for i in range(1, len(on)):
    k[i] = k[i-1] + 1 if on.iloc[i-1] <= 0 else 0
k = pd.Series(k, index=on.index)

# realized vol of overnight leg, past 60 nights, known at decision time
vol = on.rolling(60).std().shift(1) * np.sqrt(252)

def bstats(x):
    x = pd.Series(x).dropna()
    t, _ = stats.ttest_1samp(x, 0)
    return {"n": int(len(x)), "win%": round((x > 0).mean() * 100, 1),
            "mean_bp": round(x.mean() * 1e4, 2), "t": round(t, 2)}

# ---------- A. 前夜亏损幅度 -> 下一夜 ----------
print("===== A. 前夜隔夜亏损幅度 与 下一夜隔夜 (QQQ 2000-2026) =====")
buckets = [("小跌 0~-0.25%", (prev <= 0) & (prev > -0.0025)),
           ("中跌 -0.25~-0.5%", (prev <= -0.0025) & (prev > -0.005)),
           ("大跌 -0.5~-1%", (prev <= -0.005) & (prev > -0.01)),
           ("深跌 <-1%", prev <= -0.01)]
print(pd.DataFrame({n: bstats(on[m]) for n, m in buckets}).T.to_string())

# ---------- B. 策略变体 ----------
def perf(w, label, sub=None):
    """w: 仓位序列(0=空仓). 收益 = w*on - 2*C*w (双边成本按仓位)."""
    net = w * on - 2 * C * w
    if sub is not None: net = net.loc[sub]
    net = net.dropna()
    n_years = len(net) / 252
    cum = (1 + net).cumprod()
    ann = (cum.iloc[-1] ** (1 / n_years) - 1) * 100
    sh = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
    mdd = (cum / cum.cummax() - 1).min() * 100
    ww = w.loc[net.index]
    return {"strategy": label, "ann%": round(ann, 2), "sharpe": round(sh, 2),
            "maxDD%": round(mdd, 1), "trades/yr": round((ww > 0).sum() / n_years),
            "avg_size": round(ww[ww > 0].mean(), 2) if (ww > 0).any() else 0}

sig = (prev <= 0).astype(float)

variants = {}
variants["S0 基准: 固定1x"] = sig
variants["S1 深跌门槛: 前夜<-0.25%才进"] = ((prev <= -0.0025)).astype(float)
w = pd.Series(0.0, index=on.index)
w[(k == 1)] = 1.0; w[(k == 2)] = 1.5; w[(k >= 3)] = 2.0
variants["S2 连跌加仓 1x/1.5x/2x"] = w
tgt = vol.median()
wv = (tgt / vol).clip(0.5, 2.0) * sig
variants["S3 波动率目标仓位"] = wv.fillna(0)
variants["S4 连跌1-3夜才做(剔除>=4)"] = ((k >= 1) & (k <= 3)).astype(float)
variants["S5 = S2+S3 组合"] = (w * (tgt / vol).clip(0.5, 2.0)).fillna(0)

rows_full, rows_h1, rows_h2 = [], [], []
for lab, w_ in variants.items():
    rows_full.append(perf(w_, lab))
    rows_h1.append(perf(w_, lab, sub=slice("2000", "2015")))
    rows_h2.append(perf(w_, lab, sub=slice("2016", "2026")))

print("\n===== B. 策略变体对比 (QQQ, 单边成本1bp) — 全样本 2000-2026 =====")
print(pd.DataFrame(rows_full).to_string(index=False))
print("\n----- 训练段 2000-2015 -----")
print(pd.DataFrame(rows_h1).to_string(index=False))
print("\n----- 样本外 2016-2026 -----")
print(pd.DataFrame(rows_h2).to_string(index=False))

# ---------- C. 连跌加仓的尾部代价 ----------
print("\n===== C. 加仓策略(S2)的尾部: 最坏10个单夜(仓位加权收益%) =====")
s2net = (variants["S2 连跌加仓 1x/1.5x/2x"] * on).dropna()
worst = s2net.nsmallest(10)
for d, v in worst.items():
    print(f"  {d.date()}  {round(v*100,2)}%  (streak={k.loc[d]}, 当夜on={round(on.loc[d]*100,2)}%)")

# S0 worst for comparison
s0net = (sig * on).dropna()
print("  对照 S0 最坏单夜:", round(s0net.min()*100, 2), "%")
