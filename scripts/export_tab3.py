"""Export tab-3 chart data: streak buckets (QQQ vs TQQQ, 2010+) + TQQQ curves."""
import json
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)

def legs(ticker):
    df = pd.read_csv(f"{OUT}/{ticker}.csv", index_col=0, parse_dates=True)
    r = pd.DataFrame(index=df.index)
    r["on"] = df["Open"] / df["Close"].shift(1) - 1
    r["intra"] = df["Close"] / df["Open"] - 1
    r["total"] = df["Close"] / df["Close"].shift(1) - 1
    return r.dropna()

def streaks(on):
    s = np.zeros(len(on), dtype=int)
    for i in range(1, len(on)):
        s[i] = s[i-1] + 1 if on.iloc[i-1] <= 0 else 0
    return pd.Series(s, index=on.index)

def bstats(x):
    x = pd.Series(x).dropna()
    t, _ = stats.ttest_1samp(x, 0)
    return {"n": int(len(x)), "win": round((x > 0).mean() * 100, 1),
            "bp": round(x.mean() * 1e4, 2), "t": round(t, 2)}

TQ = legs("TQQQ")
QQ = legs("QQQ").loc[TQ.index[0]:]

data = {"streak": {}, "curves": {}}
for lab, r in [("QQQ", QQ), ("TQQQ", TQ)]:
    k = streaks(r["on"])
    buckets = [("无条件", pd.Series(True, index=r.index)),
               ("连跌1夜", k == 1), ("连跌2夜", k == 2),
               ("连跌3夜", k == 3), ("连跌≥4夜", k >= 4)]
    data["streak"][lab] = [dict(name=n, **bstats(r["on"][m])) for n, m in buckets]

# TQQQ curves: P0 conditional net @1bp vs TQQQ B&H vs QQQ B&H
c = 1e-4
prev = TQ["on"].shift(1)
m = prev <= 0
net = pd.Series(np.where(m, (1 + TQ["on"]) * (1 - c) ** 2 - 1, 0.0), index=TQ.index)
cum = pd.DataFrame({
    "strat": (1 + net).cumprod(),
    "tqqq_bh": (1 + TQ["total"]).cumprod(),
    "qqq_bh": (1 + QQ["total"].reindex(TQ.index).fillna(0)).cumprod(),
}).resample("ME").last()
data["curves"] = {"dates": [d.strftime("%Y-%m") for d in cum.index],
                  **{k2: [round(v, 4) for v in cum[k2]] for k2 in cum.columns}}

with open(f"{OUT}/tab3data.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)
print(json.dumps(data["streak"], ensure_ascii=False, indent=1)[:900])
print("curve ends:", {k2: data["curves"][k2][-1] for k2 in ["strat", "tqqq_bh", "qqq_bh"]})
