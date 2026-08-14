"""TQQQ overnight analysis + streak conditioning + exit-policy backtests.

streak_t = 截至 t-1 夜, 连续多少个隔夜收益<=0 (今晚决策时已知)
"""
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)
pd.set_option("display.width", 200)

def legs(ticker):
    df = pd.read_csv(f"{OUT}/{ticker}.csv", index_col=0, parse_dates=True)
    r = pd.DataFrame(index=df.index)
    r["on"] = df["Open"] / df["Close"].shift(1) - 1
    r["intra"] = df["Close"] / df["Open"] - 1
    r["total"] = df["Close"] / df["Close"].shift(1) - 1
    return r.dropna()

def bstats(x):
    x = pd.Series(x).dropna()
    if len(x) < 20: return None
    t, _ = stats.ttest_1samp(x, 0)
    return {"n": len(x), "win%": round((x > 0).mean() * 100, 1),
            "mean_bp": round(x.mean() * 1e4, 2), "t": round(t, 2)}

def streaks(on):
    s = np.zeros(len(on), dtype=int)
    for i in range(1, len(on)):
        s[i] = s[i-1] + 1 if on.iloc[i-1] <= 0 else 0
    return pd.Series(s, index=on.index)  # 连续下跌隔夜数(截至昨夜)

def summary_table(r, label):
    on, prev = r["on"], r["on"].shift(1)
    k = streaks(r["on"])
    rows = {
        "无条件": bstats(on),
        "前夜隔夜下跌后(信号)": bstats(on[prev <= 0]),
        "前夜隔夜上涨后": bstats(on[prev > 0]),
        "连跌1夜后": bstats(on[k == 1]),
        "连跌2夜后": bstats(on[k == 2]),
        "连跌3夜后": bstats(on[k == 3]),
        "连跌>=4夜后": bstats(on[k >= 4]),
    }
    print(f"\n===== {label}: 今晚隔夜收益 按条件分组 =====")
    print(pd.DataFrame({k2: v for k2, v in rows.items() if v}).T.to_string())

def losing_entry(r, label):
    prev = r["on"].shift(1)
    lost = (prev <= 0) & (r["on"] <= 0)   # 信号触发且入场后开盘亏损
    print(f"\n===== {label}: 信号夜入场后开盘亏损 (n={lost.sum()}) =====")
    print("  亏损夜的日内(扛到收盘的增量):", bstats(r["intra"][lost]))
    print("  对照-盈利夜的日内:          ", bstats(r["intra"][(prev <= 0) & (r['on'] > 0)]))

def max_dd(cum):
    return round((cum / cum.cummax() - 1).min() * 100, 1)

def perf(net, label, n_years, trades):
    net = pd.Series(net)
    cum = (1 + net).cumprod()
    ann = (cum.iloc[-1] ** (1 / n_years) - 1) * 100
    sh = net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0
    return {"policy": label, "ann%": round(ann, 2), "sharpe": round(sh, 2),
            "maxDD%": max_dd(cum), "trades/yr": round(trades / n_years)}

def policies(r, label, cost_bp=1.0):
    c = cost_bp / 1e4
    on, intra, total = r["on"], r["intra"], r["total"]
    prev = on.shift(1)
    k = streaks(on)
    n_years = len(r) / 252
    res = []

    # P0 基准: 前夜跌就持隔夜, 开盘必卖
    m = prev <= 0
    net = np.where(m, (1 + on) * (1 - c) ** 2 - 1, 0.0)
    res.append(perf(net, "P0 基准:开盘必卖,信号在就再进", n_years, m.sum()))

    # P1 扛单: 入场后若隔夜亏损, 不卖, 一直持有到首个隔夜盈利的开盘
    ret, holding, trades = [], False, 0
    vals = list(zip(on.values, intra.values, total.values, prev.values))
    for on_t, intra_t, total_t, prev_t in vals:
        if not holding:
            if prev_t <= 0 and not np.isnan(prev_t):
                holding = True; trades += 1
                # 今晚入场: 当日收盘买入 -> 明日结算, 记为下一行
                ret.append(0.0)
            else:
                ret.append(0.0)
            continue
        # holding: 从昨收持有
        if on_t > 0:   # 开盘盈利 -> 卖出
            ret.append((1 + on_t) * (1 - c) ** 2 - 1)
            holding = False
            # 当日收盘是否再入场由明日 prev 判断(自动)
        else:          # 开盘亏损 -> 扛到收盘继续持有
            ret.append(total_t)
    res.append(perf(ret, "P1 扛单:亏了不卖,持到隔夜转正", n_years, trades))

    # P2 只做连跌第1夜 (割肉后休息, 待反弹重置)
    m = k == 1
    net = np.where(m, (1 + on) * (1 - c) ** 2 - 1, 0.0)
    res.append(perf(net, "P2 只做连跌1夜(亏后休息)", n_years, m.sum()))

    # P3 只做连跌>=2夜 (越跌越买的方向)
    m = k >= 2
    net = np.where(m, (1 + on) * (1 - c) ** 2 - 1, 0.0)
    res.append(perf(net, "P3 只做连跌>=2夜", n_years, m.sum()))

    # 参照: 买入持有
    res.append(perf(total.values, "参照:买入持有", n_years, 0))

    print(f"\n===== {label}: 退出/连败政策对比 (单边成本 {cost_bp}bp) =====")
    print(pd.DataFrame(res).to_string(index=False))

def losing_streak_of_strategy(r, label):
    prev = r["on"].shift(1)
    tr = r["on"][prev <= 0]           # 每笔交易的毛收益
    lose = (tr <= 0).astype(int)
    grp = (lose != lose.shift()).cumsum()
    runs = lose.groupby(grp).agg(["sum", "size"])
    losing_runs = runs[runs["sum"] > 0]["size"]
    # 最坏连续亏损段的累计亏损
    worst_run_loss = 0
    cur = 1.0
    for v, l in zip(tr.values, lose.values):
        if l: cur *= (1 + v); worst_run_loss = min(worst_run_loss, cur - 1)
        else: cur = 1.0
    print(f"\n===== {label}: 条件策略(P0)连败特征 =====")
    print(f"  总交易 {len(tr)}, 单笔胜率 {round((tr>0).mean()*100,1)}%")
    print(f"  最长连败 {int(losing_runs.max())} 笔; 连败>=3笔出现 {int((losing_runs>=3).sum())} 次")
    print(f"  最坏单笔 {round(tr.min()*100,2)}%; 最坏连败段累亏 {round(worst_run_loss*100,2)}%")

TQ = legs("TQQQ")
QQ = legs("QQQ").loc[TQ.index[0]:]
QQfull = legs("QQQ")

for r, lab in [(TQ, "TQQQ 2010-2026"), (QQ, "QQQ 同期 2010-2026")]:
    summary_table(r, lab)
for r, lab in [(TQ, "TQQQ"), (QQfull, "QQQ 全样本")]:
    losing_entry(r, lab)
for r, lab, cb in [(TQ, "TQQQ 2010-2026", 1.0), (QQfull, "QQQ 全样本 2000-2026", 1.0)]:
    policies(r, lab, cb)
for r, lab in [(TQ, "TQQQ"), (QQfull, "QQQ 全样本")]:
    losing_streak_of_strategy(r, lab)
