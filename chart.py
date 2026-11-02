import sys
import yfinance as yf
import mplfinance as mpf
import warnings
warnings.filterwarnings("ignore")

symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
df = yf.download(symbol, period="1y", auto_adjust=True)
df.columns = df.columns.get_level_values(0)  # flatten columns

# Moving averages and trend signal
df["SMA20"] = df["Close"].rolling(20).mean()
df["SMA50"] = df["Close"].rolling(50).mean()
df["trend"] = (df["SMA20"] > df["SMA50"]).map({True: "UP", False: "DOWN"})

# Bullish engulfing: red candle followed by a green candle that covers it
prev = df.shift(1)
df["bull_engulf"] = (
    (prev["Close"] < prev["Open"]) & (df["Close"] > df["Open"])
    & (df["Open"] <= prev["Close"]) & (df["Close"] >= prev["Open"])
)

marks = df["Low"].where(df["bull_engulf"]) * 0.98
plots = [
    mpf.make_addplot(df["SMA20"]),
    mpf.make_addplot(df["SMA50"]),
    mpf.make_addplot(marks, type="scatter", marker="^", markersize=80),
]
mpf.plot(df, type="candle", volume=True, addplot=plots, title=symbol)

print(df[["Close", "SMA20", "SMA50", "trend"]].tail(3))