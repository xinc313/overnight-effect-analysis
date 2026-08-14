"""FINAL model backtest: signal(prev overnight<=0) AND not-into-Monday,
streak scale-in (1x/1.5x/2x) x vol-target (median60/vol, clip 0.5-2), 1bp/side.
"""
import json
import numpy as np
import pandas as pd

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
C = 1e-4

r = pd.read_csv(f"{OUT}/QQQ_legs.csv", index_col=0, parse_dates=True)
on = r["overnight"]
prev = on.shift(1)
k = np.zeros(len(on), dtype=int)
for i in range(1, len(on)):
    k[i] = k[i-1] + 1 if on.iloc[i-1] <= 0 else 0
k = pd.Series(k, index=on.index)
vol = on.rolling(60).std().shift(1) * np.sqrt(252)
notmon = pd.Series(on.index.dayofweek != 0, index=on.index)

sig = (prev <= 0) & notmon
w = pd.Series(0.0, index=on.index)
w[sig & (k == 1)] = 1.0
w[sig & (k == 2)] = 1.5
w[sig & (k >= 3)] = 2.0
w = (w * (vol.median() / vol).clip(0.5, 2.0)).fillna(0).clip(upper=2.0)

def perf(w_, label, sub=None):
    net = (w_ * on - 2 * C * w_)
    if sub is not None: net = net.loc[sub]
    net = net.dropna()
    ny = len(net) / 252
    cum = (1 + net).cumprod()
    tr = net[w_.loc[net.index] > 0]
    return {"label": label,
            "ann": round((cum.iloc[-1] ** (1/ny) - 1) * 100, 2),
            "sharpe": round(net.mean() / net.std() * np.sqrt(252), 2),
            "mdd": round((cum / cum.cummax() - 1).min() * 100, 1),
            "trades_yr": round((w_.loc[net.index] > 0).sum() / ny),
            "win": round((tr > 0).mean() * 100, 1),
            "final": round(cum.iloc[-1], 2)}

rows = [perf(w, "FINAL 全样本"),
        perf(w, "FINAL 训练段", slice("2000", "2015")),
        perf(w, "FINAL 样本外", slice("2016", "2026")),
        perf(w, "FINAL 2020至今", slice("2020", "2026"))]
for row in rows: print(row)

# comparison curves: FINAL vs S0 vs U0(every night) vs buy&hold (net 1bp)
s0w = (prev <= 0).astype(float)
u0w = pd.Series(1.0, index=on.index)
curves = {}
for key, w_ in [("final", w), ("s0", s0w), ("u0", u0w)]:
    net = (w_ * on - 2 * C * w_).dropna()
    curves[key] = (1 + net).cumprod().resample("ME").last()
curves["bh"] = (1 + r["total"]).cumprod().dropna().resample("ME").last()
cum = pd.DataFrame(curves).dropna()

print("\n===== 对比表数字 (1bp/边) =====")
for key, w_ in [("FINAL", w), ("S0 信号夜1x", s0w), ("U0 每天隔夜", u0w)]:
    for per, sl in [("全样本", slice(None)), ("16-26", slice("2016", "2026")),
                    ("25-26", slice("2025", "2026"))]:
        net = (w_ * on - 2 * C * w_).loc[sl].dropna()
        ny = len(net) / 252
        cumx = (1 + net).cumprod()
        print(f"{key:12s} {per:6s} 年化 {(cumx.iloc[-1]**(1/ny)-1)*100:6.2f}% "
              f"夏普 {net.mean()/net.std()*np.sqrt(252):5.2f} "
              f"回撤 {(cumx/cumx.cummax()-1).min()*100:6.1f}%")
for per, sl in [("全样本", slice(None)), ("16-26", slice("2016", "2026")),
                ("25-26", slice("2025", "2026"))]:
    net = r["total"].loc[sl].dropna()
    ny = len(net) / 252
    cumx = (1 + net).cumprod()
    print(f"{'B&H':12s} {per:6s} 年化 {(cumx.iloc[-1]**(1/ny)-1)*100:6.2f}% "
          f"夏普 {net.mean()/net.std()*np.sqrt(252):5.2f} "
          f"回撤 {(cumx/cumx.cummax()-1).min()*100:6.1f}%")

# yearly returns of FINAL
netf = (w * on - 2 * C * w).dropna()
yearly = {int(y): round(((1 + g).prod() - 1) * 100, 1) for y, g in netf.groupby(netf.index.year)}

data = {"perf": rows,
        "curves": {"dates": [d.strftime("%Y-%m") for d in cum.index],
                   **{k2: [round(v, 4) for v in cum[k2]] for k2 in cum.columns}},
        "yearly": yearly,
        "worst_night": round((w * on).min() * 100, 2),
        "avg_size": round(w[w > 0].mean(), 2), "max_size": round(w.max(), 2)}
with open(f"{OUT}/tab5data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)
print("worst night:", data["worst_night"], "% | avg size", data["avg_size"], "| max", data["max_size"])
print("yearly:", yearly)
