"""
ema_retest.py - Nifty / Bank Nifty 5-minute "EMA 9/21 cross, retest, hold" strategy (long only)
and the exact CALL option to watch when it fires.

Rules (on the INDEX 5-minute chart)
  1. EMA9 crosses ABOVE EMA21 (same trading day).
  2. Price pulls back and touches EMA9 (the retest) without closing below EMA21.
  3. BUY when a later candle closes above the retest candle's high AND above EMA9.
  4. Stop = low of the retest candle (at least 0.5 x ATR away). Target = RR x risk (default 2).
  5. One trade per cross. Entries only 09:30 - 14:30 IST. Square off everything at 15:15.

Which option? On a BUY signal the script picks the ATM call (nearest strike to the index)
and a one-strike-ITM alternative, and an expiry:
  NIFTY      strike step 50,  weekly expiry (Tuesday); skips an expiry that is under 3 days away
  BANKNIFTY  strike step 100, MONTHLY expiry only (last Tuesday); skips one under 5 days away
Always confirm the strike, expiry and lot size in your broker's option chain: holidays shift
expiries and NSE revises lot sizes (edit INSTRUMENTS below if they change).

Usage
  python ema_retest.py                          backtest both indices + an example alert
  python ema_retest.py --index banknifty        only Bank Nifty
  python ema_retest.py --watch 60 --telegram --capital 100000      LIVE: alerts on your phone
  python ema_retest.py --rr 1.5

It does NOT place orders and cannot see option premiums (read those in your broker app).
Yahoo gives only ~60 days of 5-minute candles, so a backtest has few trades. Options add spread,
charges and time decay that an index backtest ignores. Educational tool, not financial advice.
"""
import argparse
import calendar
import math
import os
import time
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EMA_FAST, EMA_SLOW = 9, 21
ENTRY_START, ENTRY_END, SQUARE_OFF = 9 * 60 + 30, 14 * 60 + 30, 15 * 60 + 15   # minutes since midnight IST
MIN_RISK_ATR = 0.5      # stop is at least this many ATRs below entry
MAX_RISK_PCT = 0.005    # skip setups whose stop is more than 0.5% away
TZ = "Asia/Kolkata"

INSTRUMENTS = {
    "nifty": {"name": "NIFTY", "yf": "^NSEI", "step": 50, "lot": 65, "expiry": "weekly"},
    "banknifty": {"name": "BANKNIFTY", "yf": "^NSEBANK", "step": 100, "lot": 30, "expiry": "monthly"},
}
# Tuesday expiries moved to the previous trading day by exchange holidays (2026, INCOMPLETE).
HOLIDAY_SHIFT = {date(2026, 10, 20): date(2026, 10, 19),
                 date(2026, 11, 10): date(2026, 11, 9),
                 date(2026, 11, 24): date(2026, 11, 23)}


# ------------------------------ data ------------------------------
def load(symbol, period="60d", interval="5m"):
    import yfinance as yf
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = df.index.tz_localize(TZ) if df.index.tz is None else df.index.tz_convert(TZ)
    return df


def prep(df):
    df = df.copy()
    c = df["Close"]
    df["EMA9"] = c.ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = c.ewm(span=EMA_SLOW, adjust=False).mean()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["day"] = df.index.date
    df["tmin"] = df.index.hour * 60 + df.index.minute
    return df


# ------------------------------ signals ------------------------------
def find_signals(df, rr=2.0):
    """Walk the candles once. Uses only past data at each candle (no look-ahead)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    e9, e21, atr = (df[k].to_numpy(float) for k in ("EMA9", "EMA21", "ATR"))
    day, tmin = df["day"].to_numpy(), df["tmin"].to_numpy()
    sigs, state = [], None
    for i in range(1, len(df)):
        if day[i] != day[i - 1]:
            state = None                       # setups never carry over the night
        if e9[i] <= e21[i]:
            state = None                       # fast EMA not above slow EMA: no long setup
            continue
        if e9[i - 1] <= e21[i - 1] and day[i] == day[i - 1]:
            state = {"cross": i, "retest": None, "used": False}    # fresh bullish cross
            continue
        if state is None or state["used"]:
            continue
        if c[i] < e21[i]:
            state = None                       # closed below EMA21: the retest failed
            continue
        r = state["retest"]
        if r is not None and c[i] > h[r] and c[i] > e9[i]:
            if ENTRY_START <= tmin[i] <= ENTRY_END:
                stop = l[r]
                risk = c[i] - stop
                min_risk = MIN_RISK_ATR * atr[i]
                if risk < min_risk:
                    stop, risk = c[i] - min_risk, min_risk
                if risk / c[i] <= MAX_RISK_PCT:
                    sigs.append({"i": i, "time": df.index[i], "ref": c[i], "stop": stop,
                                 "risk": risk, "target": c[i] + rr * risk,
                                 "retest_i": r, "cross_i": state["cross"]})
            state["used"] = True               # one trade per cross, taken or skipped
            continue
        if l[i] <= e9[i]:
            state["retest"] = i                # pulled back to the fast EMA (latest retest counts)
    return sigs


# ------------------------------ backtest ------------------------------
def run_backtest(df, sigs, cost=0.0001):
    """Enter at the NEXT candle's open, stop first if a candle hits both, square off 15:15."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
    day, tmin = df["day"].to_numpy(), df["tmin"].to_numpy()
    n, trades, busy = len(df), [], -1
    for s in sigs:
        i = s["i"]
        if i + 1 >= n or day[i + 1] != day[i] or i + 1 <= busy:
            continue
        entry, stop, target = o[i + 1], s["stop"], s["target"]
        if entry <= stop or entry >= target:
            continue                           # gapped through the levels: no trade
        j, exit_px, why = i + 1, None, None
        while j < n and day[j] == day[i + 1]:
            if tmin[j] >= SQUARE_OFF:
                exit_px, why = o[j], "15:15 square-off"
                break
            if l[j] <= stop:
                exit_px, why = min(o[j], stop), "stop"
                break
            if h[j] >= target:
                exit_px, why = target, "target"
                break
            j += 1
        if exit_px is None:
            exit_px, why, j = c[j - 1], "end of data", j - 1
        busy = j
        risk = entry - stop
        trades.append({"time": s["time"], "entry": entry, "exit": exit_px, "why": why,
                       "pts": exit_px - entry, "r": (exit_px - entry - 2 * cost * entry) / risk,
                       "risk": risk})
    return trades


def print_backtest(label, df, trades, rr):
    print("=" * 66)
    print(f"{label}  5-min EMA {EMA_FAST}/{EMA_SLOW} cross + retest   "
          f"{df.index[0].date()} to {df.index[-1].date()}  ({df['day'].nunique()} sessions)")
    if not trades:
        print("No trades in this period.")
        print("=" * 66)
        return
    r = np.array([t["r"] for t in trades])
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    eq = np.cumsum(r)
    dd = (eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()
    print(f"Trades {len(r)} | win rate {(r > 0).mean():.0%} (break-even at {1 / (1 + rr):.0%} "
          f"for {rr:g}:1) | avg {r.mean():+.2f} R | total {r.sum():+.1f} R")
    print(f"Profit factor {wins / losses if losses > 0 else float('inf'):.2f} | "
          f"max drawdown {dd:.1f} R | avg stop {np.mean([t['risk'] for t in trades]):.0f} pts")
    for why in ("target", "stop", "15:15 square-off", "end of data"):
        k = [t for t in trades if t["why"] == why]
        if k:
            print(f"  {why:18s} {len(k):3d} trades, avg {np.mean([t['r'] for t in k]):+.2f} R")
    print("-" * 66)
    print("Last trades:")
    for t in trades[-8:]:
        print(f"  {t['time']:%d %b %H:%M}  in {t['entry']:.1f}  out {t['exit']:.1f}  "
              f"{t['why']:17s} {t['r']:+.2f} R")
    print("=" * 66)
    print("R = profit divided by the initial risk. Under ~30 trades proves almost nothing, and")
    print("this ignores option spread, charges and time decay. Paper trade before real money.")


# ------------------------------ which option to watch ------------------------------
def shift(d):
    return HOLIDAY_SHIFT.get(d, d)


def next_weekly_expiry(today, min_days=3):
    """Next Tuesday expiry that is at least min_days away (today's own expiry is skipped)."""
    t = today + timedelta(days=(1 - today.weekday()) % 7)
    while (shift(t) - today).days < min_days:
        t += timedelta(days=7)
    return shift(t)


def last_tuesday(y, m):
    d = date(y, m, calendar.monthrange(y, m)[1])
    while d.weekday() != 1:
        d -= timedelta(days=1)
    return d


def monthly_expiry(today, min_days=5):
    y, m = today.year, today.month
    for _ in range(3):
        exp = shift(last_tuesday(y, m))
        if (exp - today).days >= min_days:
            return exp
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return exp


def pick_option(key, spot, today):
    inst = INSTRUMENTS[key]
    step = inst["step"]
    atm = int(math.floor(spot / step + 0.5) * step)
    exp = next_weekly_expiry(today) if inst["expiry"] == "weekly" else monthly_expiry(today)
    when = f"{exp:%d %b %Y (%a)}, {(exp - today).days} days left"
    return {"atm": atm, "itm": atm - step, "expiry": exp, "when": when,
            "main": f"{inst['name']} {atm} CE", "alt": f"{inst['name']} {atm - step} CE"}


def option_plan(sig, key, today, capital=None, delta=0.5, risk_pct=0.01):
    inst = INSTRUMENTS[key]
    opt = pick_option(key, sig["ref"], today)
    risk_pts, reward_pts = sig["risk"], sig["target"] - sig["ref"]
    lines = [f"{inst['name']} BUY signal at {sig['time']:%H:%M}",
             f"Index: entry ~{sig['ref']:.1f} | stop {sig['stop']:.1f} (-{risk_pts:.0f} pts) | "
             f"target {sig['target']:.1f} (+{reward_pts:.0f} pts)",
             f"OPTION TO WATCH: {opt['main']}   expiry {opt['when']}",
             f"Slightly safer (1 strike in the money): {opt['alt']}, same expiry",
             f"Premium moves roughly (delta {delta}): about -{delta * risk_pts:.0f} at the stop, "
             f"+{delta * reward_pts:.0f} at the target",
             "Exit on the INDEX levels. Square off by 15:15. Check strike/expiry in your broker's chain."]
    if inst["expiry"] == "monthly":
        lines.append("Bank Nifty has monthly expiry only, so premiums are higher than Nifty weekly.")
    if capital:
        per_lot = delta * risk_pts * inst["lot"]
        lots = int((capital * risk_pct) // per_lot) if per_lot > 0 else 0
        lines.append(f"Lot size {inst['lot']}: risk per lot ~{per_lot:,.0f}. Lots for {risk_pct:.0%} risk "
                     f"of {capital:,.0f}: {lots}" + ("  (0 = even 1 lot risks more than 1%)" if lots == 0 else ""))
    return "\n".join(lines)


# ------------------------------ alerts + live ------------------------------
def beep():
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1200, 400)
    except Exception:
        print("\a", end="")


def send_telegram(text):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("(Telegram not configured: add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env)")
        return
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=10)
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            print("Telegram API error:", result)
        else:
            print("Telegram notification sent successfully!")
    except Exception as ex:
        print("Telegram failed:", ex)


def market_open(now):
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and 9 * 60 + 15 <= m <= 15 * 60 + 30


def watch(keys, every, a):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    alerted = set()
    print(f"Watching {', '.join(INSTRUMENTS[k]['name'] for k in keys)} (5-min candles) every {every}s. "
          "Ctrl+C to stop.\n")
    while True:
        now = pd.Timestamp.now(tz=TZ)
        if not market_open(now):
            print(f"{now:%a %d %b %H:%M}  market closed (NSE: Mon-Fri 09:15-15:30 IST). Waiting...")
            time.sleep(every)
            continue
        for k in keys:
            try:
                df = prep(load(INSTRUMENTS[k]["yf"], period="5d"))
            except Exception as ex:
                print(f"{INSTRUMENTS[k]['name']}: data error ({ex})")
                continue
            if df.index[-1] + pd.Timedelta(minutes=5) > now:
                df = df.iloc[:-1]              # drop the candle that is still forming
            last = len(df) - 1
            fresh = [s for s in find_signals(df, a.rr)
                     if s["i"] == last and (k, s["time"]) not in alerted]
            for s in fresh:
                msg = option_plan(s, k, now.date(), a.capital, a.delta)
                print("\n>>> " + msg.replace("\n", "\n    ") + "\n")
                beep()
                if a.telegram:
                    send_telegram(msg)
                alerted.add((k, s["time"]))
            x = df.iloc[-1]
            trend = "EMA9 above EMA21" if x["EMA9"] > x["EMA21"] else "EMA9 below EMA21"
            print(f"{INSTRUMENTS[k]['name']:10s} {df.index[-1]:%H:%M}  close {x['Close']:.1f}  {trend}  "
                  f"{'(SIGNAL)' if fresh else 'no new signal'}")
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=["nifty", "banknifty", "both"], default="both")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--cost", type=float, default=0.0001, help="cost per side as a fraction (index proxy)")
    ap.add_argument("--watch", type=int, default=0, help="live mode: check every N seconds")
    ap.add_argument("--telegram", action="store_true")
    ap.add_argument("--capital", type=float, default=None)
    ap.add_argument("--delta", type=float, default=0.5)
    a = ap.parse_args()
    keys = list(INSTRUMENTS) if a.index == "both" else [a.index]
    if a.watch:
        watch(keys, a.watch, a)
        return
    for k in keys:
        df = prep(load(INSTRUMENTS[k]["yf"]))
        sigs = find_signals(df, a.rr)
        print_backtest(INSTRUMENTS[k]["name"], df, run_backtest(df, sigs, a.cost), a.rr)
        if sigs:
            print("\nExample of the alert you get live (built from the latest PAST signal):")
            print(option_plan(sigs[-1], k, date.today(), a.capital, a.delta))
            print()


if __name__ == "__main__":
    main()
