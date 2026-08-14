"""Export chart-ready JSON: monthly cumulative curves + yearly bars."""
import json
import pandas as pd

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)

data = {}
for ticker in ["SPY", "QQQ"]:
    cum = pd.read_csv(f"{OUT}/{ticker}_cum.csv", index_col=0, parse_dates=True)
    m = cum.resample("ME").last()
    data[ticker] = {
        "dates": [d.strftime("%Y-%m") for d in m.index],
        "overnight": [round(v, 4) for v in m["overnight"]],
        "intraday": [round(v, 4) for v in m["intraday"]],
        "total": [round(v, 4) for v in m["total"]],
    }

with open(f"{OUT}/results.json") as f:
    results = json.load(f)
for ticker in ["SPY", "QQQ"]:
    data[ticker]["yearly"] = results[ticker]["yearly"]

with open(f"{OUT}/chartdata.json", "w") as f:
    json.dump(data, f)
print("months:", len(data["QQQ"]["dates"]), "| size:",
      len(json.dumps(data)) // 1024, "KB")
