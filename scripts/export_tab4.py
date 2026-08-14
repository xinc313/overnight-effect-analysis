"""Export tab-4 data: loss-magnitude buckets + variant equity curves (net 1bp)."""
import json
import numpy as np
import pandas as pd
from scipy import stats

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
sig = (prev <= 0).astype(float)

def bstats(x):
    x = pd.Series(x).dropna()
    t, _ = stats.ttest_1samp(x, 0)
    return {"n": int(len(x)), "win": round((x > 0).mean() * 100, 1),
            "bp": round(x.mean() * 1e4, 2), "t": round(t, 2)}

mag = [("小跌 0~−0.25%", (prev <= 0) & (prev > -0.0025)),
       ("中跌 −0.25~−0.5%", (prev <= -0.0025) & (prev > -0.005)),
       ("大跌 −0.5~−1%", (prev <= -0.005) & (prev > -0.01)),
       ("深跌 <−1%", prev <= -0.01)]
data = {"mag": [dict(name=n, **bstats(on[m])) for n, m in mag],
        "sig_bp": round(on[prev <= 0].mean() * 1e4, 2)}

w2 = pd.Series(0.0, index=on.index)
w2[k == 1] = 1.0; w2[k == 2] = 1.5; w2[k >= 3] = 2.0
tgt = vol.median()
w3 = ((tgt / vol).clip(0.5, 2.0) * sig).fillna(0)
w5 = (w2 * (tgt / vol).clip(0.5, 2.0)).fillna(0)

curves = {}
for key, w in [("s0", sig), ("s2", w2), ("s3", w3), ("s5", w5)]:
    net = (w * on - 2 * C * w).dropna()
    curves[key] = (1 + net).cumprod().resample("ME").last()
cum = pd.DataFrame(curves).dropna()
data["curves"] = {"dates": [d.strftime("%Y-%m") for d in cum.index],
                  **{k2: [round(v, 4) for v in cum[k2]] for k2 in cum.columns}}

with open(f"{OUT}/tab4data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)
print("ends:", {k2: data["curves"][k2][-1] for k2 in ["s0", "s2", "s3", "s5"]})
