"""Grid search over 96 factor-switch combinations, train (2000-2015) vs OOS (2016+)."""
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data"
C = 1e-4

r = pd.read_csv(OUT / "QQQ_legs.csv", index_col=0, parse_dates=True)
on = r["overnight"]; prev = on.shift(1)
idx = on.index
notmon = pd.Series(idx.dayofweek != 0, index=idx)
dn = (prev <= 0); up = (prev > 0)
k = np.zeros(len(on), dtype=int)
for i in range(1, len(on)):
    k[i] = k[i-1] + 1 if on.iloc[i-1] <= 0 else 0
k = pd.Series(k, index=idx)
vol = on.rolling(60).std().shift(1) * np.sqrt(252)
voladj = (vol.median() / vol).clip(0.5, 2.0).fillna(1)
er = pd.Series(pd.DatetimeIndex(idx).month.isin([1, 4, 7, 10]), index=idx)
vix = pd.read_csv(OUT / "VIX.csv", index_col=0, parse_dates=True)["Close"].reindex(idx).ffill().shift(1)

def bucket_mean(mask, K=126):
    s = on[mask]
    return s.rolling(K).mean().shift(1).reindex(idx).ffill()

adapt_mask = ((dn & (bucket_mean(dn) > 2*C)) | (up & (bucket_mean(up) > 2*C)))

def sharpe(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0

rows = []
for cond, wkend, scale, vt, erf, vixf in product(
        ["跌后", "每天", "自适应"], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]):
    m = {"跌后": dn, "每天": pd.Series(True, index=idx), "自适应": adapt_mask}[cond].copy()
    if wkend: m &= notmon
    if erf: m &= er
    if vixf: m &= (vix < 30)
    w = m.astype(float)
    if scale:
        w = w * np.where(k >= 3, 2.0, np.where(k == 2, 1.5, 1.0))
    if vt:
        w = w * voladj
    w = pd.Series(w, index=idx).clip(upper=2.0)
    net = (w * on - 2 * C * w)
    rows.append({"条件": cond, "跳周末": wkend, "加仓": scale, "波动率": vt,
                 "财报月": erf, "VIX<30": vixf,
                 "训练夏普": round(sharpe(net.loc["2000":"2015"]), 2),
                 "样本外夏普": round(sharpe(net.loc["2016":]), 2)})

df = pd.DataFrame(rows)
df["训练排名"] = df["训练夏普"].rank(ascending=False).astype(int)
df["样本外排名"] = df["样本外夏普"].rank(ascending=False).astype(int)
pd.set_option("display.width", 200)
print("训练段前10名 及其样本外表现:")
print(df.sort_values("训练夏普", ascending=False).head(10).to_string(index=False))
print("\n排名相关性(Spearman):",
      round(df[["训练夏普", "样本外夏普"]].corr(method="spearman").iloc[0, 1], 2))
