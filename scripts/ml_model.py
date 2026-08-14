"""Walk-forward ML on overnight returns. All features known at close of t-1.

逐年重训: 每年只用该年之前的全部数据训练, 预测该年 — 全程样本外。
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

F = pd.DataFrame(index=idx)
F["prev_on"] = on.shift(1)                 # 昨夜隔夜
F["intra_y"] = intra.shift(1)              # 当日日内 (用户要求的因子)
F["total_y"] = tot.shift(1)
F["ret5"] = (1 + tot).rolling(5).apply(np.prod, raw=True).shift(1) - 1
F["kd"] = pd.Series(kd, index=idx).astype(float)
F["ku"] = pd.Series(ku, index=idx).astype(float)
F["vol60"] = on.rolling(60).std().shift(1)
F["vix"] = vix.shift(1)
F["autoc"] = on.rolling(126).corr(on.shift(1)).shift(1)
F["dow"] = idx.dayofweek.astype(float)     # 明早星期几
F["er_mo"] = pd.Series(pd.DatetimeIndex(idx).month.isin([1, 4, 7, 10]),
                       index=idx).astype(float)
y = on
data = F.join(y.rename("y")).dropna()
feats = [c for c in data.columns if c != "y"]

def perf(net, label):
    end = net.index[-1]
    for wname, start in [("全期08-26", pd.Timestamp("2008-01-01")),
                         ("近10年", end - pd.DateOffset(years=10)),
                         ("近5年", end - pd.DateOffset(years=5)),
                         ("近3年", end - pd.DateOffset(years=3)),
                         ("近1年", end - pd.DateOffset(years=1))]:
        x = net.loc[start:].dropna()
        if not len(x): continue
        ny = len(x) / 252
        cum = (1 + x).cumprod()
        sh = x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        print(f"  {label:22s} {wname:8s} 年化 {(cum.iloc[-1]**(1/ny)-1)*100:6.2f}%  "
              f"夏普 {sh:5.2f}  回撤 {(cum/cum.cummax()-1).min()*100:6.1f}%")

# walk-forward: 逐年重训
preds_gbm, preds_lr = [], []
years = range(2008, 2027)
for Y in years:
    tr = data[data.index < f"{Y}-01-01"]
    te = data[(data.index >= f"{Y}-01-01") & (data.index < f"{Y+1}-01-01")]
    if len(te) == 0: continue
    gbm = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                    learning_rate=0.05, subsample=0.8,
                                    random_state=0)
    gbm.fit(tr[feats], tr["y"])
    preds_gbm.append(pd.Series(gbm.predict(te[feats]), index=te.index))
    sc = StandardScaler().fit(tr[feats])
    lr = LogisticRegression(max_iter=1000, C=0.5)
    lr.fit(sc.transform(tr[feats]), (tr["y"] > 0).astype(int))
    preds_lr.append(pd.Series(lr.predict_proba(sc.transform(te[feats]))[:, 1],
                              index=te.index))

pg = pd.concat(preds_gbm); pl = pd.concat(preds_lr)
oo = on.loc[pg.index]

print("===== walk-forward ML (全程样本外, 1bp/边) =====")
for thr in [1e-4, 2e-4]:
    w = (pg > thr).astype(float)
    perf((w * oo - 2 * C * w), f"GBM 预测>{int(thr*1e4)}bp")
for thr in [0.5, 0.55]:
    w = (pl > thr).astype(float)
    perf((w * oo - 2 * C * w), f"LogReg P>{thr}")

# 特征重要性 (最后一次训练的GBM, 即用到2025年底的全部数据)
imp = pd.Series(gbm.feature_importances_, index=feats).sort_values(ascending=False)
print("\n===== GBM 特征重要性 (最终模型) =====")
for f_, v in imp.items():
    print(f"  {f_:8s} {v*100:5.1f}%")
