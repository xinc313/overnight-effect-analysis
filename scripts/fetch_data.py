"""Download daily OHLC for SPY/QQQ and save to CSV."""
import yfinance as yf
import pandas as pd

from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
OUT = str(OUT)

for ticker in ["SPY", "QQQ", "TQQQ"]:
    df = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(f"{OUT}/{ticker}.csv")
    print(ticker, len(df), df.index[0].date(), "->", df.index[-1].date())
