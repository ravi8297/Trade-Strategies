"""
analyze.py - run the strategy engine on a stock.

  python analyze.py AAPL                  one-time report on the latest candle
  python analyze.py AAPL --closed         use the last COMPLETED candle (safer)
  python analyze.py AAPL --chart          also draw the chart with levels
  python analyze.py AAPL --watch 300      refresh every 300 seconds (live)
  python analyze.py AAPL --backtest       test the strategy on past data
  python analyze.py --backtest --symbols AAPL MSFT NVDA TCS.NS    test several stocks together
  python analyze.py --backtest --symbols AAPL MSFT --trail        trailing-stop exit (lets winners run)
  python analyze.py RELIANCE.NS --interval 1d --period 2y

Educational tool, not financial advice. Signals are probabilities, never certainties.
"""
import argparse
import time
import warnings
from collections import defaultdict

import numpy as np

import strategies as S

warnings.filterwarnings("ignore")


def load(symbol, period, interval):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def report(symbol, D, i, capital):
    r = S.evaluate(D, i)
    print("=" * 66)
    print(f"{symbol}   candle: {D.df.index[i]}")
    print(f"Price {r['price']:.2f} | Primary trend: {r['trend']} | RSI {r['rsi']:.0f} | ATR {r['atr']:.2f}")
    print(f"SCORE {r['score']:+.1f}  ->  {r['action']}")
    print("-" * 66)
    if r["signals"]:
        for name, s, msg, tgt in sorted(r["signals"], key=lambda x: -abs(x[1])):
            print(f"  {s:+.1f}  {name}: {msg}")
    else:
        print("  No pattern or indicator signals on this candle.")
    near = sorted(r["levels"], key=lambda x: abs(x[0] - r["price"]))[:4]
    if near:
        print("-" * 66)
        print("Nearest support/resistance: " +
              ", ".join(f"{lvl:.2f} ({t} touches)" for lvl, t in sorted(near)))
    risk = r["price"] - r["stop"]
    if risk > 0:
        qty = int(capital * 0.01 // risk)
        targets = [t for *_, t in r["signals"] if t]
        target = max(targets) if targets else r["price"] + 2 * risk
        print("-" * 66)
        print(f"If you buy: stop-loss {r['stop']:.2f} (2 x ATR), risk/share {risk:.2f}")
        print(f"Position size for 1% risk on {capital:,.0f}: {qty} shares")
        print(f"Target ~{target:.2f}  (reward:risk {(target - r['price']) / risk:.1f}:1)")
    print("=" * 66)


def draw_chart(symbol, D, levels):
    import mplfinance as mpf
    df = D.df.iloc[-150:]
    plots = [mpf.make_addplot(df[c]) for c in ("SMA20", "SMA50", "SMA200")]
    kw = {}
    lv = [x for x, _ in levels if df["Low"].min() * 0.9 < x < df["High"].max() * 1.1]
    if lv:
        kw["hlines"] = dict(hlines=lv, colors="gray", linestyle="--", linewidths=0.8)
    mpf.plot(df, type="candle", volume=True, addplot=plots,
             title=f"{symbol}  (SMA20/50/200, dashed = support/resistance)", **kw)


def backtest_run(D, fwd=10, hold_max=30, cost=0.001, trail=False):
    """Replay history bar by bar. Returns trades, per-signal stats and baseline moves."""
    n, start = D.n, S.WARMUP
    stats, trades, pos = defaultdict(list), [], None
    base = [D.c[i + fwd] / D.c[i] - 1 for i in range(start, n - fwd)]
    for i in range(start, n - 1):
        r = S.evaluate(D, i)
        for name, s in r["raw"]:
            if i + fwd < n:
                stats[(name, "bull" if s > 0 else "bear")].append(D.c[i + fwd] / D.c[i] - 1)
        exited = False
        if pos is not None:
            if trail:   # chandelier trailing stop: 3 x ATR below the highest high so far
                pos["peak"] = max(pos["peak"], D.h[i - 1])
                pos["stop"] = max(pos["stop"], pos["peak"] - 3 * D.atr[i - 1])
            px = None
            if D.l[i] <= pos["stop"]:
                px = min(D.o[i], pos["stop"])
            elif r["score"] <= S.SELL_AT and not (trail and r["trend"] == "UP"):
                px = D.o[i + 1]
            elif not trail and i - pos["bar"] >= hold_max:
                px = D.c[i]
            if px is not None:
                trades.append(px / pos["entry"] - 1 - 2 * cost)
                pos, exited = None, True
        if pos is None and not exited and r["score"] >= S.BUY_AT:
            e = D.o[i + 1]
            pos = {"entry": e, "bar": i + 1, "stop": e - 2 * D.atr[i], "peak": e}
    if pos is not None:
        trades.append(D.c[-1] / pos["entry"] - 1 - 2 * cost)
    return {"trades": trades, "stats": stats, "base": base,
            "bh": D.c[-1] / D.c[start] - 1,
            "span": f"{D.df.index[start].date()} to {D.df.index[-1].date()}"}


def print_backtests(runs, fwd=10, cost=0.001, trail=False):
    """runs = [(symbol, result)]; pools trades and signal stats across symbols."""
    print("=" * 70)
    print(f"BACKTEST  ({'trailing 3xATR stop' if trail else 'exit on SELL score / stop / 30 bars'}; "
          f"entry next open; initial stop 2xATR; {cost:.2%} cost per side)")
    print("-" * 70)
    all_t, pooled, base = [], defaultdict(list), []
    for sym, res in runs:
        t = np.array(res["trades"])
        all_t += list(t)
        base += res["base"]
        for k, v in res["stats"].items():
            pooled[k] += v
        if len(t):
            tot = np.prod(1 + t) - 1
            print(f"{sym:12s} {res['span']}  trades {len(t):3d}  win {(t > 0).mean():4.0%}  "
                  f"strategy {tot:+7.1%}  buy&hold {res['bh']:+7.1%}")
        else:
            print(f"{sym:12s} {res['span']}  no trades  buy&hold {res['bh']:+7.1%}")
    if all_t:
        t = np.array(all_t)
        wins, losses = t[t > 0].sum(), -t[t < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        print("-" * 70)
        print(f"ALL TRADES: {len(t)} | win rate {(t > 0).mean():.0%} | avg trade {t.mean():+.2%} | "
              f"profit factor {pf:.2f}")
        print("(profit factor = total gains / total losses; above 1.3 over many trades is decent)")
    base_up = float((np.array(base) > 0).mean()) if base else 0.5
    print("-" * 70)
    print(f"Each signal alone: result {fwd} candles later. Baseline: ANY {fwd}-candle window was "
          f"up {base_up:.0%} of the time.")
    print(f"{'signal':34s}{'dir':5s}{'count':>6s}{'right %':>9s}{'edge':>8s}{'avg move':>10s}")
    for (name, d), rets in sorted(pooled.items(), key=lambda kv: -len(kv[1])):
        a = np.array(rets)
        right = (a > 0).mean() if d == "bull" else (a < 0).mean()
        edge = right - (base_up if d == "bull" else 1 - base_up)
        star = "*" if len(a) < 20 else " "
        print(f"{name:34s}{d:5s}{len(a):6d}{right:9.0%}{edge:+8.0%}{a.mean():10.2%}{star}")
    print("=" * 70)
    print("edge = right% minus the baseline. Only a clearly positive edge with a big count")
    print("matters. * = under 20 cases, not reliable. For bear signals, 'right' means the")
    print("price fell. Beware of tuning settings until the past looks perfect (overfitting).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default="AAPL")
    ap.add_argument("--period", default=None, help="default 2y (5y for --backtest)")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--closed", action="store_true", help="use last completed candle")
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--cost", type=float, default=0.001, help="cost per side in backtest")
    ap.add_argument("--symbols", nargs="+", help="several symbols for --backtest")
    ap.add_argument("--trail", action="store_true", help="trailing-stop exit in backtest")
    a = ap.parse_args()

    sym = a.symbol.upper()
    period = a.period or ("5y" if a.backtest else "2y")

    if a.backtest:
        runs = []
        for name in [x.upper() for x in (a.symbols or [sym])]:
            try:
                df = load(name, period, a.interval)
                if len(df) < S.WARMUP + 50:
                    print(f"{name}: not enough data ({len(df)} candles), skipped")
                    continue
                print(f"Testing {name}...")
                runs.append((name, backtest_run(S.Data(df), cost=a.cost, trail=a.trail)))
            except Exception as ex:
                print(f"{name}: failed ({ex})")
        if runs:
            print_backtests(runs, cost=a.cost, trail=a.trail)
        return

    while True:
        df = load(sym, period, a.interval)
        if len(df) < S.WARMUP + 5:
            print(f"Only {len(df)} candles downloaded; need at least {S.WARMUP + 5}. "
                  "Try a longer --period.")
            return
        D = S.Data(df)
        i = D.n - 2 if a.closed else D.n - 1
        report(sym, D, i, a.capital)
        if a.chart:
            draw_chart(sym, D, S.evaluate(D, i)["levels"])
            a.chart = False
        if not a.watch:
            return
        print(f"Next refresh in {a.watch}s. Ctrl+C to stop.\n")
        time.sleep(a.watch)


if __name__ == "__main__":
    main()