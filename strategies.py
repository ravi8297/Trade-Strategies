"""
strategies.py
Detectors for the concepts in the technical-analysis PDF:
  - Primary trend (Dow theory) using the 200-SMA
  - SMA rules (20-SMA pullback buy, 50/200 golden & death cross, 20/50 cross)
  - RSI overbought / oversold
  - Support & resistance (bounce, rejection, breakout, breakdown)
  - Head & Shoulders and Inverse Head & Shoulders (sloping neckline bonus)
  - Double top / double bottom
  - Triangles (ascending, descending, symmetric)
  - Flags (bull / bear)
  - Gaps (breakaway, runaway, exhaustion)
  - Cup & handle, rounding bottom
  - Basic candlesticks (engulfing, hammer, shooting star)
  - MACD cross, Bollinger squeeze breakout and dip-buy, 55-bar (turtle) breakout,
    RSI(2) pullback in an uptrend, +DI/-DI cross
  - ADX filter: trend signals count half when there is no trend (ADX < 20),
    contrarian signals count half when the trend is very strong (ADX > 30)
  - Volume confirmation, price targets, ATR stop-loss

NOT included on purpose:
  - Elliott Wave: labelling waves is subjective (two analysts label the same chart
    differently), so it can't be coded as a reliable rule.
  - Greater Fool Theory: it is an idea about valuations, not a chart signal.

Every detector looks ONLY at data up to bar i (no peeking at the future), so the
same code can be used live and in a backtest.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

# ------------------------- settings you can tune -------------------------
BUY_AT = 2.5        # total score needed for BUY
STRONG_AT = 4.0     # total score needed for STRONG BUY / STRONG SELL
SELL_AT = -2.5      # total score for SELL
VOL_MULT = 1.5      # volume must be this many times the 20-bar average to "confirm"
VOL_BONUS = 1.0     # extra score when volume confirms a breakout
GAP_PCT = 0.01      # minimum gap size (1%)
WARMUP = 205        # bars needed before signals are valid (200-SMA)


# ------------------------------ data + indicators ------------------------------
def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["VolAvg"] = df["Volume"].rolling(20).mean()

    # RSI(2) for short-term pullbacks
    g2 = d.clip(lower=0).ewm(alpha=1 / 2, adjust=False).mean()
    l2 = (-d.clip(upper=0)).ewm(alpha=1 / 2, adjust=False).mean()
    df["RSI2"] = 100 - 100 / (1 + g2 / l2)

    # MACD (12, 26, 9)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACDS"] = macd.ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2) and band width
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["BBU"], df["BBL"] = mid + 2 * sd, mid - 2 * sd
    df["BBW"] = (df["BBU"] - df["BBL"]) / mid

    # ADX and directional indicators (trend strength)
    up, dn = df["High"].diff(), -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    df["PDI"] = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / df["ATR"]
    df["MDI"] = 100 * mdm.ewm(alpha=1 / 14, adjust=False).mean() / df["ATR"]
    dx = 100 * (df["PDI"] - df["MDI"]).abs() / (df["PDI"] + df["MDI"])
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # 55-bar channel (turtle breakout), excluding the current bar
    df["HH55"] = df["High"].rolling(55).max().shift(1)
    df["LL55"] = df["Low"].rolling(55).min().shift(1)
    return df


class Data:
    """Holds the price table as fast numpy arrays."""

    def __init__(self, df):
        self.df = add_indicators(df)
        d = self.df
        self.n = len(d)
        self.o = d["Open"].to_numpy(float)
        self.h = d["High"].to_numpy(float)
        self.l = d["Low"].to_numpy(float)
        self.c = d["Close"].to_numpy(float)
        self.v = d["Volume"].to_numpy(float)
        self.sma20 = d["SMA20"].to_numpy(float)
        self.sma50 = d["SMA50"].to_numpy(float)
        self.sma200 = d["SMA200"].to_numpy(float)
        self.rsi = d["RSI"].to_numpy(float)
        self.atr = d["ATR"].to_numpy(float)
        self.vavg = d["VolAvg"].to_numpy(float)
        self.rsi2 = d["RSI2"].to_numpy(float)
        self.macd = d["MACD"].to_numpy(float)
        self.macds = d["MACDS"].to_numpy(float)
        self.bbu = d["BBU"].to_numpy(float)
        self.bbl = d["BBL"].to_numpy(float)
        self.bbw = d["BBW"].to_numpy(float)
        self.adx = d["ADX"].to_numpy(float)
        self.pdi = d["PDI"].to_numpy(float)
        self.mdi = d["MDI"].to_numpy(float)
        self.hh55 = d["HH55"].to_numpy(float)
        self.ll55 = d["LL55"].to_numpy(float)


def vol_ok(D, i, mult=VOL_MULT):
    va = D.vavg[i]
    return bool(va > 0 and D.v[i] > mult * va)


def _vtxt(flag):
    return " (volume spike confirms)" if flag else ""


# ------------------------------ pivots & levels ------------------------------
def pivots(D, i, w=5, lookback=150):
    """Swing highs/lows confirmed by w bars on each side (uses data up to i only)."""
    highs, lows = [], []
    for j in range(max(w, i - lookback), i - w + 1):
        seg_h = D.h[j - w:j + w + 1]
        seg_l = D.l[j - w:j + w + 1]
        if D.h[j] == seg_h.max() and (not highs or j - highs[-1][0] > w):
            highs.append((j, float(D.h[j])))
        if D.l[j] == seg_l.min() and (not lows or j - lows[-1][0] > w):
            lows.append((j, float(D.l[j])))
    return highs, lows


def sr_levels(highs, lows, tol=0.015):
    """Group nearby pivot prices into support/resistance levels (2+ touches)."""
    prices = sorted([p for _, p in highs] + [p for _, p in lows])
    clusters = []
    for p in prices:
        if clusters and abs(p - np.mean(clusters[-1])) / np.mean(clusters[-1]) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(float(np.mean(cl)), len(cl)) for cl in clusters if len(cl) >= 2]


# ------------------------------ trend ------------------------------
def primary_trend(D, i):
    """Dow theory: follow the primary trend, treat counter moves as secondary."""
    if D.c[i] > D.sma200[i] and D.sma50[i] > D.sma200[i]:
        return 1, "UP"
    if D.c[i] < D.sma200[i] and D.sma50[i] < D.sma200[i]:
        return -1, "DOWN"
    return 0, "SIDEWAYS"


# ------------------------------ detectors ------------------------------
# Each returns a list of (name, score, message, target_or_None). Positive = bullish.

def d_sma_pullback(D, i, ctx):
    out = []
    if D.sma20[i] > D.sma20[i - 5] and D.c[i - 1] < D.sma20[i - 1] and D.c[i] > D.sma20[i]:
        out.append(("SMA20 pullback buy", 1.0,
                    "Uptrend: price dipped below the 20-SMA and closed back above it", None))
    if D.sma20[i] < D.sma20[i - 5] and D.c[i - 1] > D.sma20[i - 1] and D.c[i] < D.sma20[i]:
        out.append(("SMA20 rejection sell", -1.0,
                    "Downtrend: price popped above the 20-SMA and closed back below it", None))
    return out


def d_ma_cross(D, i, ctx):
    out = []
    if D.sma50[i - 1] <= D.sma200[i - 1] and D.sma50[i] > D.sma200[i]:
        out.append(("Golden cross 50/200", 2.0, "50-SMA crossed ABOVE the 200-SMA", None))
    if D.sma50[i - 1] >= D.sma200[i - 1] and D.sma50[i] < D.sma200[i]:
        out.append(("Death cross 50/200", -2.0, "50-SMA crossed BELOW the 200-SMA", None))
    if D.sma20[i - 1] <= D.sma50[i - 1] and D.sma20[i] > D.sma50[i]:
        out.append(("SMA20/50 bullish cross", 1.0, "20-SMA crossed above the 50-SMA", None))
    if D.sma20[i - 1] >= D.sma50[i - 1] and D.sma20[i] < D.sma50[i]:
        out.append(("SMA20/50 bearish cross", -1.0, "20-SMA crossed below the 50-SMA", None))
    return out


def d_rsi(D, i, ctx):
    r, rp = D.rsi[i], D.rsi[i - 1]
    out = []
    if r < 30 and r > rp:
        out.append(("RSI oversold turning up", 1.0, f"RSI {r:.0f} (<30) and rising", None))
    if r > 70 and r < rp:
        out.append(("RSI overbought turning down", -1.0, f"RSI {r:.0f} (>70) and falling", None))
    return out


def d_candles(D, i, ctx):
    o, h, l, c = D.o[i], D.h[i], D.l[i], D.c[i]
    po, pc = D.o[i - 1], D.c[i - 1]
    out = []
    if pc < po and c > o and o <= pc and c >= po:
        out.append(("Bullish engulfing", 1.0, "Green candle fully covers the previous red candle", None))
    if pc > po and c < o and o >= pc and c <= po:
        out.append(("Bearish engulfing", -1.0, "Red candle fully covers the previous green candle", None))
    body, rng = abs(c - o), h - l
    if rng > 0 and body > 0:
        lower, upper = min(o, c) - l, h - max(o, c)
        if c < D.sma20[i] and lower >= 2 * body and upper <= 0.3 * rng:
            out.append(("Hammer", 1.0, "Long lower wick after a decline (buyers stepped in)", None))
        if c > D.sma20[i] and upper >= 2 * body and lower <= 0.3 * rng:
            out.append(("Shooting star", -1.0, "Long upper wick after a rise (sellers stepped in)", None))
    return out


def d_support_resistance(D, i, ctx):
    out = []
    c, o = D.c[i], D.o[i]
    v = vol_ok(D, i)
    for lvl, touches in ctx.levels:
        if D.c[i - 1] <= lvl < c:
            out.append(("Resistance breakout", 2.0 + (VOL_BONUS if v else 0),
                        f"Closed above resistance {lvl:.2f} ({touches} touches){_vtxt(v)}", None))
        elif D.c[i - 1] >= lvl > c:
            out.append(("Support breakdown", -2.0 - (VOL_BONUS if v else 0),
                        f"Closed below support {lvl:.2f} ({touches} touches){_vtxt(v)}", None))
        elif lvl <= c <= lvl * 1.01 and c > o:
            out.append(("Bounce at support", 1.0,
                        f"Green candle sitting on support {lvl:.2f} ({touches} touches)", None))
        elif lvl * 0.99 <= c <= lvl and c < o:
            out.append(("Rejection at resistance", -1.0,
                        f"Red candle at resistance {lvl:.2f} ({touches} touches)", None))
    return out


def d_double(D, i, ctx):
    out = []
    H, L = ctx.highs, ctx.lows
    v = vol_ok(D, i)
    if len(H) >= 2:
        (a, pa), (b, pb) = H[-2], H[-1]
        top = max(pa, pb)
        if b - a >= 10 and abs(pa - pb) / top <= 0.02 and i - b <= 40:
            neck = D.l[a:b + 1].min()
            if ((min(pa, pb) - neck) / min(pa, pb) >= 0.03 and D.h[b:i + 1].max() <= top * 1.01
                    and D.c[i - 1] >= neck > D.c[i]):
                tgt = neck - (top - neck)
                out.append(("Double top", -2.0 - (VOL_BONUS if v else 0),
                            f"Broke below neckline {neck:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt))
    if len(L) >= 2:
        (a, pa), (b, pb) = L[-2], L[-1]
        bot = min(pa, pb)
        if b - a >= 10 and abs(pa - pb) / max(pa, pb) <= 0.02 and i - b <= 40:
            neck = D.h[a:b + 1].max()
            if ((neck - max(pa, pb)) / neck >= 0.03 and D.l[b:i + 1].min() >= bot * 0.99
                    and D.c[i - 1] <= neck < D.c[i]):
                tgt = neck + (neck - bot)
                out.append(("Double bottom", 2.0 + (VOL_BONUS if v else 0),
                            f"Broke above neckline {neck:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt))
    return out


def d_head_shoulders(D, i, ctx):
    out = []
    H, L = ctx.highs, ctx.lows
    v = vol_ok(D, i)
    if len(H) >= 3:
        (a, pa), (b, pb), (c_, pc) = H[-3:]
        if (pb > pa and pb > pc and abs(pa - pc) / max(pa, pc) <= 0.06
                and (pb - max(pa, pc)) / pb >= 0.02 and i - c_ <= 30):
            t1 = a + int(np.argmin(D.l[a:b + 1]))
            t2 = b + int(np.argmin(D.l[b:c_ + 1]))
            if t2 > t1:
                y1, y2 = D.l[t1], D.l[t2]
                slope = (y2 - y1) / (t2 - t1)
                neck_i, neck_p = y1 + slope * (i - t1), y1 + slope * (i - 1 - t1)
                if D.c[i - 1] >= neck_p and D.c[i] < neck_i:
                    height = pb - (y1 + slope * (b - t1))
                    tgt = neck_i - height
                    bonus = 0.5 if slope < 0 else 0.0   # PDF: downward-sloping neckline is stronger
                    out.append(("Head & Shoulders (top)", -2.0 - bonus - (VOL_BONUS if v else 0),
                                f"Broke below neckline ~{neck_i:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt))
    if len(L) >= 3:
        (a, pa), (b, pb), (c_, pc) = L[-3:]
        if (pb < pa and pb < pc and abs(pa - pc) / max(pa, pc) <= 0.06
                and (min(pa, pc) - pb) / min(pa, pc) >= 0.02 and i - c_ <= 30):
            t1 = a + int(np.argmax(D.h[a:b + 1]))
            t2 = b + int(np.argmax(D.h[b:c_ + 1]))
            if t2 > t1:
                y1, y2 = D.h[t1], D.h[t2]
                slope = (y2 - y1) / (t2 - t1)
                neck_i, neck_p = y1 + slope * (i - t1), y1 + slope * (i - 1 - t1)
                if D.c[i - 1] <= neck_p and D.c[i] > neck_i:
                    height = (y1 + slope * (b - t1)) - pb
                    tgt = neck_i + height
                    bonus = 0.5 if slope > 0 else 0.0   # PDF: rising neckline is stronger
                    out.append(("Inverse Head & Shoulders", 2.0 + bonus + (VOL_BONUS if v else 0),
                                f"Broke above neckline ~{neck_i:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt))
    return out


def d_triangle(D, i, ctx):
    hp = [(j, p) for j, p in ctx.highs if j >= i - 70]
    lp = [(j, p) for j, p in ctx.lows if j >= i - 70]
    if len(hp) < 2 or len(lp) < 2 or i - max(hp[-1][0], lp[-1][0]) > 25:
        return []
    hs, hi = np.polyfit([j for j, _ in hp], [p for _, p in hp], 1)
    ls, li = np.polyfit([j for j, _ in lp], [p for _, p in lp], 1)
    price = D.c[i]
    hr, lr, flat = hs / price, ls / price, 0.0004
    up_i, lo_i = hs * i + hi, ls * i + li
    up_p, lo_p = hs * (i - 1) + hi, ls * (i - 1) + li
    j0 = min(hp[0][0], lp[0][0])
    height = (hs * j0 + hi) - (ls * j0 + li)
    if up_p <= lo_p or height <= 0:
        return []
    if abs(hr) <= flat and lr > flat:
        kind, up_w, dn_w = "Ascending triangle", 2.0, -1.0
    elif abs(lr) <= flat and hr < -flat:
        kind, up_w, dn_w = "Descending triangle", 1.0, -2.0
    elif hr < -flat and lr > flat:
        kind, up_w, dn_w = "Symmetric triangle", 1.0, -1.0
    else:
        return []
    v = vol_ok(D, i)
    if D.c[i - 1] <= up_p and price > up_i:
        tgt = up_i + height
        return [(kind + " breakout", up_w + (VOL_BONUS if v else 0),
                 f"Broke above the upper line ~{up_i:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt)]
    if D.c[i - 1] >= lo_p and price < lo_i:
        tgt = lo_i - height
        return [(kind + " breakdown", dn_w - (VOL_BONUS if v else 0),
                 f"Broke below the lower line ~{lo_i:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt)]
    return []


def d_flag(D, i, ctx):
    out = []
    for e in range(i - 4, i - 16, -1):
        if e < 12:
            break
        s = e - 10
        # bull flag: sharp rise (pole) then a small pullback, then breakout
        pole_low = D.l[s:e + 1].min()
        pole = D.c[e] - pole_low
        if pole / pole_low >= 0.10:
            cons_h, cons_l = D.h[e + 1:i].max(), D.l[e + 1:i].min()
            if cons_h <= D.h[e] * 1.03 and (D.c[e] - cons_l) <= 0.5 * pole and D.c[i] > cons_h:
                declining = D.v[e + 1:i].mean() < D.v[max(0, e - 5):e + 1].mean()
                bonus = VOL_BONUS if (declining or vol_ok(D, i)) else 0
                tgt = cons_h + pole
                out.append(("Bull flag", 2.0 + bonus,
                            f"Pole +{pole / pole_low:.0%}, tight pullback, broke above {cons_h:.2f}; "
                            f"target ~{tgt:.2f}", tgt))
                return out
        # bear flag: sharp fall then a small bounce, then breakdown
        pole_high = D.h[s:e + 1].max()
        drop = pole_high - D.c[e]
        if drop / pole_high >= 0.10:
            cons_h, cons_l = D.h[e + 1:i].max(), D.l[e + 1:i].min()
            if cons_l >= D.l[e] * 0.97 and (cons_h - D.c[e]) <= 0.5 * drop and D.c[i] < cons_l:
                declining = D.v[e + 1:i].mean() < D.v[max(0, e - 5):e + 1].mean()
                bonus = VOL_BONUS if (declining or vol_ok(D, i)) else 0
                tgt = cons_l - drop
                out.append(("Bear flag", -2.0 - bonus,
                            f"Pole -{drop / pole_high:.0%}, weak bounce, broke below {cons_l:.2f}; "
                            f"target ~{tgt:.2f}", tgt))
                return out
    return out


def d_gaps(D, i, ctx):
    out = []
    up = D.l[i] > D.h[i - 1] * (1 + GAP_PCT)
    dn = D.h[i] < D.l[i - 1] * (1 - GAP_PCT)
    if not (up or dn):
        return out
    base_hi, base_lo = D.h[i - 21:i - 1].max(), D.l[i - 21:i - 1].min()
    consolidating = (base_hi - base_lo) / D.c[i - 1] < 0.10
    ext_up = D.c[i - 1] > D.sma50[i - 1] * 1.15
    ext_dn = D.c[i - 1] < D.sma50[i - 1] * 0.85
    v = vol_ok(D, i)
    trend_up = D.c[i - 1] > D.sma50[i - 1] and D.sma50[i] > D.sma50[i - 5]
    trend_dn = D.c[i - 1] < D.sma50[i - 1] and D.sma50[i] < D.sma50[i - 5]
    if up:
        if consolidating and v:
            out.append(("Breakaway gap up", 2.0, "Gap up out of a tight range on heavy volume", None))
        elif ext_up:
            out.append(("Exhaustion gap up", -1.0,
                        "Gap up after a big run (>15% above 50-SMA): possible blow-off top", None))
        elif trend_up:
            out.append(("Runaway gap up", 1.0, "Gap up inside an uptrend: trend continuing", None))
    if dn:
        if consolidating and v:
            out.append(("Breakaway gap down", -2.0, "Gap down out of a tight range on heavy volume", None))
        elif ext_dn:
            out.append(("Exhaustion gap down", 1.0,
                        "Gap down after a big fall (>15% below 50-SMA): possible bottom", None))
        elif trend_dn:
            out.append(("Runaway gap down", -1.0, "Gap down inside a downtrend: trend continuing", None))
    return out


def d_cup_handle(D, i, ctx):
    H = ctx.highs
    for ai in range(len(H)):
        a, pa = H[ai]
        for bi in range(ai + 1, len(H)):
            b, pb = H[bi]
            span = b - a
            if not (25 <= span <= 110) or abs(pa - pb) / pa > 0.05 or not (3 <= i - b <= 30):
                continue
            rim = max(pa, pb)
            seg = D.l[a:b + 1]
            bottom, bidx = seg.min(), a + int(seg.argmin())
            depth = (min(pa, pb) - bottom) / min(pa, pb)
            if not (0.12 <= depth <= 0.35) or not (a + 0.25 * span <= bidx <= a + 0.75 * span):
                continue
            if (pb - D.l[b:i].min()) > 0.5 * (pb - bottom) or D.c[b:i].max() > rim * 1.005:
                continue
            if D.c[i - 1] <= rim < D.c[i]:
                v = vol_ok(D, i)
                tgt = rim + (rim - bottom)
                return [("Cup & handle", 2.0 + (VOL_BONUS if v else 0),
                         f"Broke above the rim {rim:.2f}; target ~{tgt:.2f}{_vtxt(v)}", tgt)]
    return []


def d_rounding_bottom(D, i, ctx):
    n = 60
    seg = D.c[i - n + 1:i + 1]
    x = np.arange(n)
    a, b, _ = np.polyfit(x, seg, 2)
    if a <= 0:
        return []
    vertex = -b / (2 * a)
    lip = seg[:8].max()
    if (0.3 * n <= vertex <= 0.7 * n and (lip - seg.min()) / lip >= 0.05
            and D.c[i - 1] <= lip < D.c[i]):
        return [("Rounding bottom", 1.0,
                 f"Gradual U-shaped recovery, closed above the left lip {lip:.2f}", None)]
    return []


def d_macd(D, i, ctx):
    m, s, mp, sp = D.macd[i], D.macds[i], D.macd[i - 1], D.macds[i - 1]
    out = []
    if mp <= sp and m > s:
        z = m < 0   # a cross below the zero line is an earlier, stronger turn
        out.append(("MACD bullish cross", 1.5 if z else 1.0,
                    "MACD crossed above its signal line" + (" below zero (early turn)" if z else ""), None))
    if mp >= sp and m < s:
        z = m > 0
        out.append(("MACD bearish cross", -1.5 if z else -1.0,
                    "MACD crossed below its signal line" + (" above zero (early turn)" if z else ""), None))
    return out


def d_bollinger(D, i, ctx):
    out = []
    w = D.bbw[i - 120:i]
    if np.isfinite(w).sum() > 60:
        recent_squeeze = np.nanmin(D.bbw[i - 5:i]) <= np.nanpercentile(w, 20)
        v = vol_ok(D, i)
        if recent_squeeze and D.c[i] > D.bbu[i] and D.c[i - 1] <= D.bbu[i - 1]:
            out.append(("Bollinger squeeze breakout up", 1.5 + (VOL_BONUS if v else 0),
                        f"Volatility squeeze, then closed above the upper band{_vtxt(v)}", None))
        if recent_squeeze and D.c[i] < D.bbl[i] and D.c[i - 1] >= D.bbl[i - 1]:
            out.append(("Bollinger squeeze breakdown", -1.5 - (VOL_BONUS if v else 0),
                        f"Volatility squeeze, then closed below the lower band{_vtxt(v)}", None))
    if D.c[i] > D.sma200[i] and D.c[i - 1] < D.bbl[i - 1] and D.c[i] > D.bbl[i]:
        out.append(("Bollinger dip buy (uptrend)", 1.0,
                    "Uptrend: price fell below the lower band and closed back inside", None))
    if D.c[i] < D.sma200[i] and D.c[i - 1] > D.bbu[i - 1] and D.c[i] < D.bbu[i]:
        out.append(("Bollinger rally sell (downtrend)", -1.0,
                    "Downtrend: price rose above the upper band and closed back inside", None))
    return out


def d_channel_breakout(D, i, ctx):
    out = []
    v = vol_ok(D, i)
    if D.c[i - 1] <= D.hh55[i] < D.c[i]:
        out.append(("55-bar high breakout", 1.5 + (VOL_BONUS if v else 0),
                    f"Closed above the highest high of the last 55 bars ({D.hh55[i]:.2f}){_vtxt(v)}", None))
    if D.c[i - 1] >= D.ll55[i] > D.c[i]:
        out.append(("55-bar low breakdown", -1.5 - (VOL_BONUS if v else 0),
                    f"Closed below the lowest low of the last 55 bars ({D.ll55[i]:.2f}){_vtxt(v)}", None))
    return out


def d_rsi2_pullback(D, i, ctx):
    out = []
    if D.c[i] > D.sma200[i] and D.rsi2[i] < 10 <= D.rsi2[i - 1]:
        out.append(("RSI(2) pullback buy", 1.5,
                    f"Uptrend (above 200-SMA) with a sharp short-term dip: RSI(2) = {D.rsi2[i]:.0f}", None))
    if D.c[i] < D.sma200[i] and D.rsi2[i] > 90 >= D.rsi2[i - 1]:
        out.append(("RSI(2) spike sell", -1.5,
                    f"Downtrend (below 200-SMA) with a sharp short-term spike: RSI(2) = {D.rsi2[i]:.0f}", None))
    return out


def d_di_cross(D, i, ctx):
    out = []
    if D.adx[i] > 20:
        if D.pdi[i - 1] <= D.mdi[i - 1] and D.pdi[i] > D.mdi[i]:
            out.append(("DI bullish cross", 1.0, f"+DI crossed above -DI with ADX {D.adx[i]:.0f} (real trend)", None))
        if D.pdi[i - 1] >= D.mdi[i - 1] and D.pdi[i] < D.mdi[i]:
            out.append(("DI bearish cross", -1.0, f"-DI crossed above +DI with ADX {D.adx[i]:.0f} (real trend)", None))
    return out


DETECTORS = [d_sma_pullback, d_ma_cross, d_rsi, d_candles, d_support_resistance, d_double,
             d_head_shoulders, d_triangle, d_flag, d_gaps, d_cup_handle, d_rounding_bottom,
             d_macd, d_bollinger, d_channel_breakout, d_rsi2_pullback, d_di_cross]

# ADX filter: trend-following signals are unreliable when there is no trend (ADX < 20);
# contrarian signals fail when a trend is very strong (ADX > 30).
TREND_FOLLOWING = {
    "Resistance breakout", "Support breakdown", "Golden cross 50/200", "Death cross 50/200",
    "SMA20/50 bullish cross", "SMA20/50 bearish cross", "MACD bullish cross", "MACD bearish cross",
    "55-bar high breakout", "55-bar low breakdown", "Bollinger squeeze breakout up",
    "Bollinger squeeze breakdown", "DI bullish cross", "DI bearish cross", "Bull flag", "Bear flag",
}
CONTRARIAN = {
    "RSI oversold turning up", "RSI overbought turning down", "Rejection at resistance",
    "Bounce at support", "Shooting star", "Hammer",
}


# ------------------------------ scoring ------------------------------
def action_for(score):
    if score >= STRONG_AT:
        return "STRONG BUY"
    if score >= BUY_AT:
        return "BUY"
    if score <= -STRONG_AT:
        return "STRONG SELL"
    if score <= SELL_AT:
        return "SELL"
    return "HOLD / WAIT"


def evaluate(D, i):
    """Score bar i using only data up to i."""
    highs, lows = pivots(D, i)
    ctx = SimpleNamespace(highs=highs, lows=lows, levels=sr_levels(highs, lows))
    found = {}
    for det in DETECTORS:
        for sig in det(D, i, ctx):
            if sig[0] not in found or abs(sig[1]) > abs(found[sig[0]][1]):
                found[sig[0]] = sig
    sigs = list(found.values())

    tr, trname = primary_trend(D, i)
    total = float(tr)          # trading with the primary trend gets +1 / -1
    adjusted = []
    adx = D.adx[i]
    for name, s, msg, tgt in sigs:
        if adx < 20 and name in TREND_FOLLOWING:
            s *= 0.5
            msg += f" [weak trend, ADX {adx:.0f}: half weight]"
        elif adx > 30 and name in CONTRARIAN:
            s *= 0.5
            msg += f" [very strong trend, ADX {adx:.0f}: contrarian half weight]"
        if tr != 0 and np.sign(s) != tr:
            s *= 0.5           # counter-trend signals count half (Dow theory: secondary moves)
            msg += " [against primary trend, half weight]"
        adjusted.append((name, s, msg, tgt))
        total += s

    return {
        "i": i, "price": float(D.c[i]), "trend": trname, "rsi": float(D.rsi[i]),
        "atr": float(D.atr[i]), "adx": float(D.adx[i]), "stop": float(D.c[i] - 2 * D.atr[i]),
        "score": total, "action": action_for(total), "signals": adjusted,
        "raw": [(n, s) for n, s, _, _ in sigs], "levels": ctx.levels,
    }