"""Rebuild final-tab data: fixed model, walk-forward GBM, 50/50 ensemble, B&H.

GBM: 逐年重训 GradientBoosting, 预测>2bp 才持隔夜, 1bp/边。
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
C = 1e-4

r = pd.read_csv(f"{OUT}/QQQ_legs.csv", index_col=0, parse_dates=True)
on, intra, tot = r["overnight"], r["intraday"], r["total"]
idx = on.index
vix = pd.read_csv(f"{OUT}/VIX.csv", index_col=0, parse_dates=True)["Close"].reindex(idx).ffill()

kd = np.zeros(len(on), dtype=int); ku = np.zeros(len(on), dtype=int)
for i in range(1, len(on)):
    kd[i] = kd[i-1] + 1 if on.iloc[i-1] <= 0 else 0
    ku[i] = ku[i-1] + 1 if on.iloc[i-1] > 0 else 0
kd = pd.Series(kd, index=idx); ku = pd.Series(ku, index=idx)

# ---- 固定模型 (第五页原版, 封顶2x) ----
vol = on.rolling(60).std().shift(1) * np.sqrt(252)
notmon = pd.Series(idx.dayofweek != 0, index=idx)
sig = ((on.shift(1) <= 0) & notmon).fillna(False)
wf = pd.Series(0.0, index=idx)
wf[sig & (kd == 1)] = 1.0; wf[sig & (kd == 2)] = 1.5; wf[sig & (kd >= 3)] = 2.0
wf = (wf * (vol.median() / vol).clip(0.5, 2.0)).fillna(0).clip(upper=2.0)
fixed = (wf * on - 2 * C * wf).fillna(0)

# ---- GBM walk-forward ----
F = pd.DataFrame(index=idx)
F["prev_on"] = on.shift(1)
F["intra_y"] = intra.shift(1)
F["total_y"] = tot.shift(1)
F["ret5"] = (1 + tot).rolling(5).apply(np.prod, raw=True).shift(1) - 1
F["kd"] = kd.astype(float); F["ku"] = ku.astype(float)
F["vol60"] = on.rolling(60).std().shift(1)
F["vix"] = vix.shift(1)
F["autoc"] = on.rolling(126).corr(on.shift(1)).shift(1)
F["dow"] = idx.dayofweek.astype(float)
F["er_mo"] = pd.Series(pd.DatetimeIndex(idx).month.isin([1, 4, 7, 10]), index=idx).astype(float)
data = F.join(on.rename("y")).dropna()
feats = [c for c in data.columns if c != "y"]

preds = []
for Y in range(2008, 2027):
    tr = data[data.index < f"{Y}-01-01"]
    te = data[(data.index >= f"{Y}-01-01") & (data.index < f"{Y+1}-01-01")]
    if len(te) == 0: continue
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, subsample=0.8, random_state=0)
    gbm.fit(tr[feats], tr["y"])
    preds.append(pd.Series(gbm.predict(te[feats]), index=te.index))
pg = pd.concat(preds)
pg.to_csv(f"{OUT}/gbm_pred.csv")
wg = (pg > 2e-4).astype(float)
gbmnet = (wg * on.loc[pg.index] - 2 * C * wg)

start = pg.index[0]
fixed08 = fixed.loc[start:]
bh08 = tot.loc[start:].fillna(0)
ens = 0.5 * fixed08 + 0.5 * gbmnet.reindex(fixed08.index).fillna(0)

def stats(net):
    x = net.dropna()
    ny = len(x) / 252
    cum = (1 + x).cumprod()
    return {"ann": round((cum.iloc[-1] ** (1/ny) - 1) * 100, 2),
            "sharpe": round(x.mean() / x.std() * np.sqrt(252), 2),
            "mdd": round((cum / cum.cummax() - 1).min() * 100, 1)}

end = idx[-1]
wins = {"full08": None, "y10": 10, "y5": 5, "y3": 3, "y1": 1}
table = {}
for name, net in [("fixed", fixed08), ("gbm", gbmnet), ("ens", ens), ("bh", bh08)]:
    table[name] = {}
    for wn, yrs in wins.items():
        s = start if yrs is None else end - pd.DateOffset(years=yrs)
        table[name][wn] = stats(net.loc[s:])
        print(name, wn, table[name][wn])

curves = {}
for key, net in [("fixed", fixed08), ("gbm", gbmnet.reindex(fixed08.index).fillna(0)),
                 ("ens", ens), ("bh", bh08)]:
    curves[key] = (1 + net).cumprod().resample("ME").last()
cum = pd.DataFrame(curves).dropna()

imp = pd.Series(gbm.feature_importances_, index=feats).sort_values(ascending=False)
NAMES = {"ret5": "近5日收益", "prev_on": "昨夜隔夜", "total_y": "昨日全日", "vix": "VIX",
         "vol60": "60夜波动", "intra_y": "当日日内", "kd": "连跌计数", "autoc": "隔夜自相关",
         "dow": "星期几", "er_mo": "财报月", "ku": "连涨计数"}
ens_y = {int(y): round(((1 + g).prod() - 1) * 100, 1) for y, g in ens.groupby(ens.index.year)}

out = {"table": table,
       "curves": {"dates": [d.strftime("%Y-%m") for d in cum.index],
                  **{k: [round(v, 4) for v in cum[k]] for k in cum.columns}},
       "imp": [{"name": NAMES[f], "v": round(v * 100, 1)} for f, v in imp.items()],
       "yearly": ens_y,
       "gbm_trades_yr": round(float(wg.sum()) / (len(wg) / 252))}
with open(f"{OUT}/tab5data2.json", "w") as f:
    json.dump(out, f, ensure_ascii=False)
print("trades/yr gbm:", out["gbm_trades_yr"], "| yearly ens:", ens_y)
