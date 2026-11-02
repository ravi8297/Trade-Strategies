import sys, time, csv, warnings
import yfinance as yf

warnings.filterwarnings("ignore")
symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
cash, shares, last_seen = 100000.0, 0, None   # virtual money

def rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)

def log(action, price, qty):
    with open("trades.csv", "a", newline="") as f:
        csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), symbol, action, price, qty])

print(f"Paper trading {symbol} with 100000 virtual cash. Ctrl+C to stop.\n")
while True:
    df = yf.download(symbol, period="5d", interval="5m", auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["RSI"] = rsi(df["Close"])

    price = float(df.iloc[-1]["Close"])
    cur, prev = df.iloc[-2], df.iloc[-3]
    candle = df.index[-2]

    if last_seen is None:
        last_seen = candle                      # skip the first old signal
    elif candle != last_seen:                   # a new candle just closed
        last_seen = candle
        cross_up = prev["SMA20"] <= prev["SMA50"] and cur["SMA20"] > cur["SMA50"]
        cross_dn = prev["SMA20"] >= prev["SMA50"] and cur["SMA20"] < cur["SMA50"]

        if shares == 0 and (cross_up or cur["RSI"] < 30):
            qty = int(cash * 0.5 // price)      # use 50% of cash
            if qty > 0:
                cash -= qty * price
                shares += qty
                log("BUY", round(price, 2), qty)
                print(f">>> VIRTUAL BUY {qty} @ {price:.2f}")
        elif shares > 0 and (cross_dn or cur["RSI"] > 70):
            cash += shares * price
            log("SELL", round(price, 2), shares)
            print(f">>> VIRTUAL SELL {shares} @ {price:.2f}")
            shares = 0

    value = cash + shares * price
    print(f"{df.index[-1]}  price {price:.2f}  shares {shares}  cash {cash:.0f}  total {value:.0f}")
    time.sleep(60)