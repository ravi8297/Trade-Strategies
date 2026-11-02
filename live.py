import sys
import time
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")
symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)

print(f"Watching {symbol} every 60 seconds. Press Ctrl+C to stop.\n")
while True:
    df = yf.download(symbol, period="5d", interval="5m",
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])

    live = df.iloc[-1]    # candle still forming
    cur = df.iloc[-2]     # last closed candle (used for signals)
    prev = df.iloc[-3]

    signals = []
    if prev["SMA20"] <= prev["SMA50"] and cur["SMA20"] > cur["SMA50"]:
        signals.append("SMA20 crossed ABOVE SMA50 (bullish)")
    if prev["SMA20"] >= prev["SMA50"] and cur["SMA20"] < cur["SMA50"]:
        signals.append("SMA20 crossed BELOW SMA50 (bearish)")

    if cur["RSI"] < 30:
        signals.append("RSI below 30 (oversold)")
    if cur["RSI"] > 70:
        signals.append("RSI above 70 (overbought)")
    if (prev["Close"] < prev["Open"] and cur["Close"] > cur["Open"]
            and cur["Open"] <= prev["Close"] and cur["Close"] >= prev["Open"]):
        signals.append("Bullish engulfing candle")

    trend = "UP" if cur["SMA20"] > cur["SMA50"] else "DOWN"
    print(f"{df.index[-1]}  price {live['Close']:.2f}  RSI {cur['RSI']:.0f}  "
          f"trend {trend}  signals: {', '.join(signals) or 'none'}")
    time.sleep(60)