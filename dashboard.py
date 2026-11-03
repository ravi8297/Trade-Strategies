"""
Stock Tools Dashboard - Angel One Option Selector (PAPER ONLY)

One-file Streamlit app:
- NIFTY / BANKNIFTY 5-minute EMA 9/21 cross + retest
- Angel One SmartAPI option contract master, option Greeks and FULL quotes
- ATM / ITM CE comparison, liquidity and spread filters
- Paper trade ledger with mark-to-market and stop/target checks
- Telegram alerts (optional)

IMPORTANT:
This app contains no order-placement, modification or cancellation code.
Angel One credentials must be stored locally in .env, never shared in chat/GitHub.

Install:
    pip install streamlit yfinance pandas numpy requests python-dotenv pyotp

Run:
    streamlit run dashboard.py
"""

import os
import math
import json
import uuid
import socket
import calendar
import time
import threading
import sqlite3
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None
import yfinance as yf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import pyotp
except ImportError:
    pyotp = None


# ============================= APP CONFIG =============================

st.set_page_config(page_title="Stock Tools | Angel One Paper Trader",
                   page_icon="📈", layout="wide")

TZ = "Asia/Kolkata"
API_ROOT = "https://apiconnect.angelone.in"
MASTER_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
EMA_FAST, EMA_SLOW = 9, 21
ENTRY_START, ENTRY_END, SQUARE_OFF = 9 * 60 + 30, 14 * 60 + 30, 15 * 60 + 15
MIN_RISK_ATR = 0.5
MAX_RISK_PCT = 0.005

INSTRUMENTS = {
    "NIFTY": {"yf": "^NSEI", "step": 50, "index_token": "99926000"},
    "BANKNIFTY": {"yf": "^NSEBANK", "step": 100, "index_token": "99926009"},
}


# ============================= GENERAL HELPERS =============================

def local_ip() -> str:
    configured = os.getenv("ANGEL_CLIENT_LOCAL_IP", "").strip()
    if configured:
        return configured
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        value = sock.getsockname()[0]
        sock.close()
        return value
    except Exception:
        return "127.0.0.1"


def mac_address() -> str:
    node = uuid.getnode()
    return ":".join(f"{(node >> shift) & 0xff:02x}" for shift in range(40, -1, -8))


def money(value) -> str:
    try:
        return f"₹{float(value):,.2f}"
    except Exception:
        return "—"


# ============================= MARKET DATA / EMA STRATEGY =============================

@st.cache_data(ttl=60)
def load_index_data(symbol: str, period: str = "60d") -> pd.DataFrame:
    data = yf.download(symbol, period=period, interval="5m",
                       auto_adjust=True, progress=False, threads=False)
    if data is None or data.empty:
        raise ValueError(f"No Yahoo Finance candles returned for {symbol}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.loc[:, ~data.columns.duplicated()]
    data = data[["Open", "High", "Low", "Close"]].dropna().copy()
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC").tz_convert(TZ)
    else:
        data.index = data.index.tz_convert(TZ)
    return data


def prepare_index_data(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close = df["Close"]
    df["EMA9"] = close.ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = close.ewm(span=EMA_SLOW, adjust=False).mean()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - close.shift()).abs(),
        (df["Low"] - close.shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["day"] = df.index.date
    df["tmin"] = df.index.hour * 60 + df.index.minute
    return df


def find_signals(df: pd.DataFrame, rr: float = 2.0) -> list:
    if len(df) < EMA_SLOW + 2:
        return []
    high = df["High"].to_numpy(float)
    low = df["Low"].to_numpy(float)
    close = df["Close"].to_numpy(float)
    e9 = df["EMA9"].to_numpy(float)
    e21 = df["EMA21"].to_numpy(float)
    atr = df["ATR"].to_numpy(float)
    days = df["day"].to_numpy()
    tmin = df["tmin"].to_numpy()

    signals, state = [], None
    for i in range(1, len(df)):
        if days[i] != days[i - 1]:
            state = None
        if e9[i] <= e21[i]:
            state = None
            continue
        if e9[i - 1] <= e21[i - 1] and days[i] == days[i - 1]:
            state = {"cross": i, "retest": None, "used": False}
            continue
        if state is None or state["used"]:
            continue
        if close[i] < e21[i]:
            state = None
            continue

        r = state["retest"]
        if r is not None and close[i] > high[r] and close[i] > e9[i]:
            if ENTRY_START <= tmin[i] <= ENTRY_END:
                stop = low[r]
                risk = close[i] - stop
                minimum = MIN_RISK_ATR * atr[i]
                if risk < minimum:
                    stop, risk = close[i] - minimum, minimum
                if risk > 0 and risk / close[i] <= MAX_RISK_PCT:
                    signals.append({
                        "i": i, "time": df.index[i], "ref": close[i],
                        "stop": stop, "risk": risk,
                        "target": close[i] + rr * risk,
                        "retest_i": r, "cross_i": state["cross"]
                    })
            state["used"] = True
            continue
        if low[i] <= e9[i]:
            state["retest"] = i
    return signals


def backtest(df: pd.DataFrame, signals: list, cost: float = 0.0001) -> list:
    op = df["Open"].to_numpy(float)
    hi = df["High"].to_numpy(float)
    lo = df["Low"].to_numpy(float)
    cl = df["Close"].to_numpy(float)
    days = df["day"].to_numpy()
    tmin = df["tmin"].to_numpy()
    trades, busy = [], -1
    for sig in signals:
        i = sig["i"]
        en_i = i + 1
        if en_i >= len(df) or days[en_i] != days[i] or en_i <= busy:
            continue
        entry, stop, target = op[en_i], sig["stop"], sig["target"]
        if entry <= stop or entry >= target:
            continue
        j, exit_px, reason = en_i, None, None
        while j < len(df) and days[j] == days[en_i]:
            if tmin[j] >= SQUARE_OFF:
                exit_px, reason = op[j], "15:15 square-off"
                break
            if lo[j] <= stop:
                exit_px, reason = min(op[j], stop), "Stop loss"
                break
            if hi[j] >= target:
                exit_px, reason = target, "Target"
                break
            j += 1
        if exit_px is None:
            j = max(en_i, j - 1)
            exit_px, reason = cl[j], "End of data"
        busy = j
        risk = entry - stop
        net = exit_px - entry - 2 * cost * entry
        trades.append({
            "Signal time": sig["time"], "Entry time": df.index[en_i],
            "Entry": entry, "Stop": stop, "Target": target,
            "Exit": exit_px, "Exit reason": reason,
            "Points": exit_px - entry, "R": net / risk if risk else 0
        })
    return trades


# ============================= ANGEL ONE SMARTAPI (READ-ONLY) =============================

class AngelOneReadOnly:
    """Only login, option Greeks and market quote methods are implemented."""

    def __init__(self):
        self.api_key = os.getenv("ANGEL_API_KEY", "").strip()
        self.client_code = os.getenv("ANGEL_CLIENT_CODE", "").strip()
        self.pin = os.getenv("ANGEL_PIN", "").strip()
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "").strip()
        self.public_ip = os.getenv("ANGEL_CLIENT_PUBLIC_IP", "").strip()
        self.jwt = None

    def credentials_ready(self) -> bool:
        return all([self.api_key, self.client_code, self.pin,
                    self.totp_secret, self.public_ip])

    def login(self) -> str:
        if not self.credentials_ready():
            raise RuntimeError(
                "Add ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN, "
                "ANGEL_TOTP_SECRET and ANGEL_CLIENT_PUBLIC_IP to .env"
            )
        if pyotp is None:
            raise RuntimeError("Install pyotp: pip install pyotp")

        totp = pyotp.TOTP(self.totp_secret).now()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": local_ip(),
            "X-ClientPublicIP": self.public_ip,
            "X-MACAddress": mac_address(),
            "X-PrivateKey": self.api_key,
        }
        body = {"clientcode": self.client_code,
                "password": self.pin, "totp": totp}
        response = requests.post(
            API_ROOT + "/rest/auth/angelbroking/user/v1/loginByPassword",
            json=body, headers=headers, timeout=20
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("status") or not payload.get("data", {}).get("jwtToken"):
            raise RuntimeError(f"Angel One login failed: {payload}")
        self.jwt = payload["data"]["jwtToken"]
        return self.jwt

    def headers(self) -> dict:
        if not self.jwt:
            self.login()
        return {
            "Authorization": "Bearer " + self.jwt,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-UserType": "USER",
            "X-SourceID": "WEB",
            "X-ClientLocalIP": local_ip(),
            "X-ClientPublicIP": self.public_ip,
            "X-MACAddress": mac_address(),
            "X-PrivateKey": self.api_key,
        }

    def post(self, path: str, body: dict) -> dict:
        response = requests.post(
            API_ROOT + path, json=body, headers=self.headers(), timeout=25
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("status"):
            raise RuntimeError(
                f"Angel One API error {payload.get('errorcode')}: "
                f"{payload.get('message')}"
            )
        return payload.get("data")

    def historical_candles(self, token: str, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        """Fetch read-only 5-minute NFO candles for one option contract."""
        data = self.post(
            "/rest/secure/angelbroking/historical/v1/getCandleData",
            {
                "exchange": "NFO", "symboltoken": str(token),
                "interval": "FIVE_MINUTE",
                "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
                "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
            }
        )
        if not data:
            return pd.DataFrame()
        frame = pd.DataFrame(data, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors="coerce")
        if frame["Datetime"].dt.tz is None:
            frame["Datetime"] = frame["Datetime"].dt.tz_localize(TZ)
        else:
            frame["Datetime"] = frame["Datetime"].dt.tz_convert(TZ)
        frame = frame.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna(subset=["Open", "High", "Low", "Close"])

    def option_greeks(self, underlying: str, expiry: date) -> pd.DataFrame:
        expiry_text = expiry.strftime("%d%b%Y").upper()
        data = self.post(
            "/rest/secure/angelbroking/marketData/v1/optionGreek",
            {"name": underlying, "expirydate": expiry_text}
        )
        frame = pd.DataFrame(data or [])
        if frame.empty:
            return frame
        for col in ["strikePrice", "delta", "gamma", "theta",
                    "vega", "impliedVolatility", "tradeVolume"]:
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "strikePrice" in frame:
            frame["strike"] = frame["strikePrice"]
        return frame

    def full_quotes(self, token_list: list) -> pd.DataFrame:
        # SmartAPI quote endpoint accepts up to 50 tokens per request.
        rows = []
        for start in range(0, len(token_list), 50):
            tokens = token_list[start:start + 50]
            data = self.post(
                "/rest/secure/angelbroking/market/v1/quote/",
                {"mode": "FULL", "exchangeTokens": {"NFO": tokens}}
            )
            rows.extend((data or {}).get("fetched", []))
        return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def download_instrument_master() -> pd.DataFrame:
    response = requests.get(MASTER_URL, timeout=40)
    response.raise_for_status()
    frame = pd.DataFrame(response.json())
    required = {"token", "symbol", "name", "expiry", "strike",
                "lotsize", "instrumenttype", "exch_seg"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Instrument master fields missing: {sorted(missing)}")
    frame = frame.copy()
    frame["strike"] = pd.to_numeric(frame["strike"], errors="coerce") / 100
    frame["lotsize"] = pd.to_numeric(frame["lotsize"], errors="coerce")
    frame["expiry_date"] = pd.to_datetime(
        frame["expiry"], format="%d%b%Y", errors="coerce"
    ).dt.date
    return frame


def get_option_contracts(master: pd.DataFrame, index_name: str,
                         expiry: date, spot: float) -> pd.DataFrame:
    step = INSTRUMENTS[index_name]["step"]
    atm = int(math.floor(spot / step + 0.5) * step)
    df = master[
        (master["exch_seg"] == "NFO")
        & (master["instrumenttype"] == "OPTIDX")
        & (master["name"] == index_name)
        & (master["expiry_date"] == expiry)
        & (master["symbol"].astype(str).str.endswith("CE"))
    ].copy()

    if df.empty:
        return df

    # Focus on ATM and nearby strikes to keep the selector practical.
    df["distance_from_atm"] = (df["strike"] - atm).abs()
    df = df[df["strike"].between(atm - 3 * step, atm + 3 * step)]
    df = df.sort_values(["distance_from_atm", "strike"]).head(15)
    return df


def all_expiry_option_contracts(master: pd.DataFrame, index_name: str, expiry: date,
                                sides: str = "CE + PE") -> pd.DataFrame:
    """All listed strikes for the chosen underlying and expiry; no ATM or Greek filter."""
    df = master.loc[
        (master["exch_seg"] == "NFO") &
        (master["instrumenttype"] == "OPTIDX") &
        (master["name"] == index_name) &
        (master["expiry_date"] == expiry)
    ].copy()
    suffixes = ("CE", "PE") if sides == "CE + PE" else (sides,)
    df = df[df["symbol"].astype(str).str.endswith(suffixes)]
    return df.drop_duplicates(subset=["token"]).sort_values(["strike", "symbol"])


def scan_every_strike(client, contracts, lookback_days, rr, pause_seconds=0.45):
    """Read-only sequential scan. Safe to call from a worker thread (no Streamlit UI calls)."""
    end_dt = datetime.now(pd.Timestamp.now(tz=TZ).tzinfo)
    start_dt = end_dt - timedelta(days=lookback_days * 2 + 3)
    now = pd.Timestamp.now(tz=TZ)
    rows, events, failures = [], [], []
    total = len(contracts)

    for pos, (_, contract) in enumerate(contracts.iterrows(), 1):
        symbol = str(contract["symbol"])
        try:
            raw = client.historical_candles(str(contract["token"]), start_dt, end_dt)
            if not raw.empty:
                raw = raw.loc[raw.index + pd.Timedelta(minutes=5) <= now]
            if len(raw) < 25:
                raise ValueError(f"Only {len(raw)} completed candles; need 25")

            enriched, found = analyze_option_candles(raw, rr)
            last = enriched.iloc[-1]
            latest_time = enriched.index[-1]

            for event in found:
                if pd.Timestamp(event["time"]) == latest_time:
                    events.append({**event, "contract": symbol})

            latest_buy = next(
                (e for e in reversed(found)
                 if pd.Timestamp(e["time"]) == latest_time
                 and e.get("stage") == "CONFIRMED PAPER BUY SETUP"),
                None
            )
            rows.append({
                "Contract": symbol,
                "Strike": contract["strike"],
                "Side": "CE" if symbol.endswith("CE") else "PE",
                "Last candle": latest_time,
                "Close": float(last["Close"]),
                "EMA9": float(last["EMA9"]),
                "EMA21": float(last["EMA21"]),
                "Support": float(last["Support"]),
                "Resistance": float(last["Resistance"]),
                "Trend": "Bullish" if last["EMA9"] > last["EMA21"] else "Bearish",
                "Target": float(latest_buy["target"]) if latest_buy else np.nan,
                "Latest events": ", ".join(
                    e["stage"] for e in found
                    if pd.Timestamp(e["time"]) == latest_time
                ) or "None"
            })
        except Exception as exc:
            failures.append({"Contract": symbol, "Error": str(exc)})

        if pos < total:
            time.sleep(pause_seconds)

    return pd.DataFrame(rows), events, pd.DataFrame(failures)


def build_option_comparison(client: AngelOneReadOnly, contracts: pd.DataFrame,
                            index_name: str, expiry: date, spot: float,
                            delta_min: float, delta_max: float,
                            max_spread_pct: float) -> pd.DataFrame:
    if contracts.empty:
        return pd.DataFrame()

    greeks = client.option_greeks(index_name, expiry)
    if greeks.empty:
        raise RuntimeError("Angel One returned no option Greeks for this expiry.")

    greeks = greeks[greeks["optionType"].astype(str).str.upper() == "CE"].copy()
    greeks["strike"] = pd.to_numeric(greeks["strikePrice"], errors="coerce")

    merged = contracts.merge(
        greeks[["strike", "delta", "gamma", "theta", "vega",
                "impliedVolatility", "tradeVolume"]],
        on="strike", how="left"
    )

    quotes = client.full_quotes(merged["token"].astype(str).tolist())
    if not quotes.empty:
        quotes = quotes.rename(columns={
            "symbolToken": "token", "tradingSymbol": "trading_symbol",
            "opnInterest": "open_interest", "tradeVolume": "quote_volume",
            "totBuyQuan": "buy_qty", "totSellQuan": "sell_qty"
        })
        for col in ["ltp", "open_interest", "quote_volume"]:
            if col in quotes:
                quotes[col] = pd.to_numeric(quotes[col], errors="coerce")
        merged = merged.merge(
            quotes[["token", "ltp", "open_interest", "quote_volume",
                    "depth"]].drop_duplicates("token"),
            on="token", how="left"
        )
    else:
        merged["ltp"] = np.nan
        merged["open_interest"] = np.nan
        merged["quote_volume"] = np.nan
        merged["depth"] = None

    def best_prices(depth):
        try:
            buy = depth.get("buy", [])
            sell = depth.get("sell", [])
            bid = float(buy[0].get("price", 0)) if buy else np.nan
            ask = float(sell[0].get("price", 0)) if sell else np.nan
            return pd.Series([bid, ask])
        except Exception:
            return pd.Series([np.nan, np.nan])

    if "depth" in merged:
        merged[["best_bid", "best_ask"]] = merged["depth"].apply(best_prices)
    else:
        merged["best_bid"], merged["best_ask"] = np.nan, np.nan

    merged["spread_pct"] = np.where(
        (merged["best_bid"] > 0) & (merged["best_ask"] > 0),
        (merged["best_ask"] - merged["best_bid"]) /
        ((merged["best_ask"] + merged["best_bid"]) / 2) * 100,
        np.nan
    )
    merged["abs_delta"] = merged["delta"].abs()
    merged["delta_ok"] = merged["abs_delta"].between(delta_min, delta_max)
    merged["spread_ok"] = merged["spread_pct"].isna() | (
        merged["spread_pct"] <= max_spread_pct
    )
    merged["liquid"] = (
        (merged["open_interest"].fillna(0) > 0)
        & (merged["quote_volume"].fillna(0) > 0)
    )
    merged["distance_from_atm"] = (
        merged["strike"] - round(spot / INSTRUMENTS[index_name]["step"])
        * INSTRUMENTS[index_name]["step"]
    ).abs()

    # Transparent ranking, not a prediction or guarantee.
    merged["score"] = (
        merged["delta_ok"].astype(int) * 3
        + merged["spread_ok"].astype(int) * 2
        + merged["liquid"].astype(int) * 2
        - merged["distance_from_atm"] /
        INSTRUMENTS[index_name]["step"] * 0.25
    )
    merged["qualifies"] = merged["delta_ok"] & merged["spread_ok"] & merged["liquid"]
    return merged.sort_values(["qualifies", "score"], ascending=[False, False])


# ============================= OPTION SIGNALS / AUTO LEVELS / ALERTS =============================

def analyze_option_candles(raw: pd.DataFrame, rr: float = 2.0):
    """Return enriched option candles, automatic levels and chronological setup events."""
    if raw is None or len(raw) < 25:
        raise ValueError("At least 25 completed 5-minute option candles are required.")
    df = raw.copy()
    df["EMA9"] = df["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-prev_close).abs(), (df["Low"]-prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    # Rolling prior-candle swing levels; shift(1) prevents look-ahead.
    df["Resistance"] = df["High"].rolling(20, min_periods=5).max().shift(1)
    df["Support"] = df["Low"].rolling(20, min_periods=5).min().shift(1)
    events=[]
    state=None
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i-1]
        ts=df.index[i]
        if row["EMA9"] > row["EMA21"] and prev["EMA9"] <= prev["EMA21"]:
            state={"cross_i":i,"retest_i":None,"entry_sent":False}
            events.append({"time":ts,"stage":"BULLISH EMA CROSSOVER","price":float(row["Close"]),"detail":"EMA 9 crossed above EMA 21"})
        if row["EMA9"] <= row["EMA21"]:
            if state is not None:
                events.append({"time":ts,"stage":"BULLISH SETUP INVALIDATED","price":float(row["Close"]),"detail":"EMA 9 fell to/below EMA 21"})
            state=None
            continue
        if pd.notna(row["Resistance"]) and row["Close"] > row["Resistance"] and prev["Close"] <= prev["Resistance"]:
            events.append({"time":ts,"stage":"RESISTANCE BREAKOUT","price":float(row["Close"]),"detail":f"Closed above prior 20-candle resistance {row['Resistance']:.2f}"})
        if state is None:
            continue
        if state["retest_i"] is None and row["Low"] <= row["EMA9"] and row["Close"] >= row["EMA21"]:
            state["retest_i"]=i
            events.append({"time":ts,"stage":"EMA9 RETEST","price":float(row["Close"]),"detail":"Price touched EMA 9 and held above EMA 21"})
            continue
        r=state["retest_i"]
        if r is not None and not state["entry_sent"] and row["Close"] > df.iloc[r]["High"] and row["Close"] > row["EMA9"]:
            stop=float(df.iloc[r]["Low"])
            risk=float(row["Close"]-stop)
            min_risk=float(0.5*row["ATR"]) if pd.notna(row["ATR"]) else 0
            if risk < min_risk:
                stop=float(row["Close"]-min_risk); risk=min_risk
            if risk>0:
                events.append({"time":ts,"stage":"CONFIRMED PAPER BUY SETUP","price":float(row["Close"]),"stop":stop,"target":float(row["Close"]+rr*risk),"detail":"Closed above retest-candle high and EMA 9"})
            state["entry_sent"]=True
    return df, events

def send_telegram_alert(message: str, chat_id: Optional[str] = None) -> tuple[bool, str]:
    """Send a Telegram message to an explicit destination; never expose the bot token."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    destination = (chat_id or os.getenv("TELEGRAM_ENTRY_CHAT_ID", "")).strip()
    if not token or not destination:
        return False, "Telegram is not configured. Check TELEGRAM_BOT_TOKEN and the relevant chat ID in .env."
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": destination, "text": message},
            timeout=15,
        )
        payload = response.json()
        if response.ok and payload.get("ok"):
            return True, "Telegram alert sent."
        return False, f"Telegram API rejected the message: {payload.get('description', 'unknown error')}"
    except Exception as exc:
        return False, f"Telegram send failed: {exc}"


def send_retest_telegram(message: str, message_id: Optional[int] = None):
    """Send the first retest alert, then edit that same Telegram message as H/L changes."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_RETEST_CHAT_ID", "").strip()
    if not token or not chat_id:
        return False, "Set TELEGRAM_BOT_TOKEN and TELEGRAM_RETEST_CHAT_ID in .env.", None
    try:
        method = "editMessageText" if message_id else "sendMessage"
        data = {"chat_id": chat_id, "text": message}
        if message_id:
            data["message_id"] = message_id
        response = requests.post(
            f"https://api.telegram.org/bot{token}/{method}",
            data=data, timeout=15
        )
        payload = response.json()
        if response.ok and payload.get("ok"):
            result = payload.get("result") or {}
            return True, "Retest Telegram message updated.", result.get("message_id", message_id)
        description = payload.get("description", "unknown error")
        # Telegram may reject an unchanged edit; that is not a delivery failure.
        if message_id and "message is not modified" in str(description).lower():
            return True, "Retest Telegram message unchanged.", message_id
        return False, f"Telegram retest message failed: {description}", message_id
    except Exception as exc:
        return False, f"Telegram retest message failed: {exc}", message_id


def send_entry_telegram(message: str) -> tuple[bool, str]:
    # Backward-compatible fallback to TELEGRAM_CHAT_ID for existing setups.
    return send_telegram_alert(
        message,
        os.getenv("TELEGRAM_ENTRY_CHAT_ID", "") or os.getenv("TELEGRAM_CHAT_ID", ""),
    )


def format_retest_alert(contract: str, ltp: float, ema9: float, ema21: float,
                        candle_high: float, candle_low: float, candle_time) -> str:
    stamp = pd.Timestamp(candle_time)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(TZ)
    else:
        stamp = stamp.tz_convert(TZ)
    return (
        "STOCK TOOLS - RETEST ALERT\n\n"
        f"{contract}\n"
        f"Retest candle is forming ({stamp:%H:%M}–{(stamp + pd.Timedelta(minutes=5)):%H:%M} IST)\n\n"
        f"Current price : ₹{ltp:.2f}\n"
        f"EMA 9         : ₹{ema9:.2f}\n"
        f"EMA 21        : ₹{ema21:.2f}\n"
        f"Current candle High: ₹{candle_high:.2f}\n"
        f"Current candle Low : ₹{candle_low:.2f}\n\n"
        f"Wait for this 5-minute candle to close. Watch breakout above its FINAL High.\n"
        "NOT A BUY SIGNAL. If price breaks EMA 21 before candle close, this retest is cancelled."
    )


def format_entry_alert(contract: str, entry: float, stop: float, target: float,
                       rr: float, live_price: float) -> str:
    return (
        "STOCK TOOLS - ENTRY ALERT\n\n"
        f"{contract}\n\n"
        f"Price broke above retest candle High: ₹{entry:.2f}\n"
        f"Current live price: ₹{live_price:.2f}\n\n"
        f"BUY       : ₹{entry:.2f} (breakout reference)\n"
        f"STOP-LOSS : ₹{stop:.2f} (Retest candle Low)\n"
        f"TARGET    : ₹{target:.2f} (1:{rr:g} Risk-Reward)\n\n"
        "Potential entry opportunity. Check live price/slippage and review chart before entering.\n"
        "Paper alert only. No broker order placed."
    )


def _alert_db_path():
    # Stored beside dashboard.py. Keep this file private and out of Git.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_tools_alerts.sqlite3")


def claim_alert_once(key):
    """Atomically reserve an alert key across reruns and multiple app sessions."""
    try:
        with sqlite3.connect(_alert_db_path(), timeout=10) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS sent_alerts (alert_key TEXT PRIMARY KEY, sent_at TEXT NOT NULL)"
            )
            cursor = conn.execute(
                "INSERT OR IGNORE INTO sent_alerts(alert_key, sent_at) VALUES (?, ?)",
                (key, datetime.now().isoformat(timespec="seconds")),
            )
            return cursor.rowcount == 1
    except sqlite3.Error:
        # Do not send if deduplication storage is unavailable: duplicates are worse.
        return False


def release_alert_claim(key):
    """Allow retry when Telegram rejects or cannot receive an alert."""
    try:
        with sqlite3.connect(_alert_db_path(), timeout=10) as conn:
            conn.execute("DELETE FROM sent_alerts WHERE alert_key = ?", (key,))
    except sqlite3.Error:
        pass


def alert_key_for(event, label):
    stamp = pd.Timestamp(event["time"])
    return f"{label}|{event['stage']}|{stamp.isoformat()}"


def format_buy_alert(event, label):
    stamp = pd.Timestamp(event["time"])
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(TZ)
    else:
        stamp = stamp.tz_convert(TZ)

    return (
        f"BUY ALERT\n\n"
        f"{label}\n"
        f"Time: {stamp.strftime('%Y-%m-%d %H:%M')} IST\n\n"
        f"Buy : ₹{float(event['price']):.2f}\n"
        f"SL  : ₹{float(event['stop']):.2f}\n"
        f"Sell: ₹{float(event['target']):.2f}\n\n"
        f"Why buying: EMA Retest Breakout\n\n"
        f"5-minute strategy\n"
        f"No broker order placed."
    )


def dispatch_new_alerts(events: list, label: str):
    if "sent_alert_keys" not in st.session_state:
        st.session_state.sent_alert_keys = set()

    sent_count = 0
    for event in events or []:
        if event.get("stage") != "CONFIRMED PAPER BUY SETUP":
            continue

        key = alert_key_for(event, label)
        if not claim_alert_once(key):
            continue

        ok, detail = send_entry_telegram(format_buy_alert(event, label))
        if ok:
            st.session_state.sent_alert_keys.add(key)
            sent_count += 1
        else:
            release_alert_claim(key)
            st.warning(f"{label}: {detail}")

    return sent_count


# Keep monitor state alive across Streamlit script reruns.
@st.cache_resource
def get_monitor_manager():
    return {
        "lock": threading.Lock(),
        "state": {
            "running": False,
            "stop_event": None,
            "thread": None,
            "status": "Stopped",
            "last_scan": None,
            "last_scan_duration": None,
            "last_results": pd.DataFrame(),
            "last_events": [],
            "last_errors": pd.DataFrame(),
            "last_alert_count": 0,
            "last_retest_count": 0,
            "live_candidates": pd.DataFrame(),
            "live_chart_data": {},
            "context": None,
            "sent_keys": set(),
        },
    }


def monitor_lock():
    return get_monitor_manager()["lock"]


def monitor_state():
    return get_monitor_manager()["state"]


def _monitor_worker(client, contracts, lookback_days, rr, context, stop_event):
    """
    Bootstrap completed candles once, then poll Angel One FULL quotes in batches.
    Retest alerts are intrabar; entry alerts are sent only after the retest candle
    has closed and a later live quote trades above its frozen high.
    """
    baselines = {}
    errors = []
    total = len(contracts)
    with monitor_lock():
        monitor_state()["status"] = f"Loading candle history for {total} contracts..."

    end_dt = datetime.now(pd.Timestamp.now(tz=TZ).tzinfo)
    start_dt = end_dt - timedelta(days=lookback_days * 2 + 3)
    now = pd.Timestamp.now(tz=TZ)

    # One historical request per contract at startup. This is deliberately
    # throttled; the quote polling phase uses 50-token batches.
    for pos, (_, contract) in enumerate(contracts.iterrows(), 1):
        if stop_event.is_set():
            break
        symbol = str(contract["symbol"])
        token = str(contract["token"])
        try:
            raw = client.historical_candles(token, start_dt, end_dt)
            if not raw.empty:
                raw = raw.loc[raw.index + pd.Timedelta(minutes=5) <= now]
            if len(raw) < 25:
                raise ValueError(f"Only {len(raw)} completed candles; need 25")
            enriched, _ = analyze_option_candles(raw, rr)
            last = enriched.iloc[-1]
            baselines[token] = {
                "symbol": symbol,
                "strike": float(contract.get("strike", np.nan)),
                "side": "CE" if symbol.endswith("CE") else "PE",
                "ema9": float(last["EMA9"]),
                "ema21": float(last["EMA21"]),
                "resistance": float(last["Resistance"]) if pd.notna(last["Resistance"]) else np.nan,
                "last_close": float(last["Close"]),
                "last_candle_time": pd.Timestamp(enriched.index[-1]),
                "state": "WAITING_FOR_RETEST",
                "retest_time": None,
                "retest_high": None,
                "retest_low": None,
                "retest_message_id": None,
                "retest_closed": False,
                "entry_sent": False,
                "last_ltp": None,
                "live_candle_time": None,
                "live_high": None,
                "live_low": None,
                "history": enriched[["Close", "EMA9", "EMA21"]].tail(40).copy(),
                "candles": raw[["Open", "High", "Low", "Close", "Volume"]].tail(120).copy(),
                "live_open": None,
            }
        except Exception as exc:
            errors.append({"Contract": symbol, "Error": str(exc)})
        if pos < total:
            time.sleep(0.45)

    if stop_event.is_set():
        with monitor_lock():
            monitor_state()["running"] = False
            monitor_state()["status"] = "Stopped"
        return

    active_contracts = contracts[contracts["token"].astype(str).isin(baselines)].copy()
    tokens = active_contracts["token"].astype(str).tolist()
    token_to_contract = {
        str(row["token"]): row for _, row in active_contracts.iterrows()
    }
    with monitor_lock():
        monitor_state()["last_errors"] = pd.DataFrame(errors)
        monitor_state()["status"] = (
            f"Live quote monitoring: {len(tokens)}/{total} contracts loaded. "
            "Polling Angel One FULL quotes in batches."
        )

    poll_seconds = max(5, int(os.getenv("STOCKTOOLS_POLL_SECONDS", "10")))
    while not stop_event.is_set() and tokens:
        cycle_started = time.time()
        retest_alerts = 0
        entry_alerts = 0
        quote_errors = []
        try:
            quotes = client.full_quotes(tokens)
            if quotes.empty:
                raise RuntimeError("Angel One returned no live quotes.")
            quotes = quotes.rename(columns={"symbolToken": "token"})
            quotes["token"] = quotes["token"].astype(str)
            quotes["ltp"] = pd.to_numeric(quotes["ltp"], errors="coerce")
            quote_map = {
                str(row["token"]): float(row["ltp"])
                for _, row in quotes.iterrows()
                if pd.notna(row.get("ltp")) and float(row["ltp"]) > 0
            }
            now_ts = pd.Timestamp.now(tz=TZ)
            candle_start = now_ts.floor("5min")

            for token, state in baselines.items():
                ltp = quote_map.get(token)
                if ltp is None:
                    continue

                # Keep a compact live price series for charts.
                hist = state["history"]
                hist.loc[now_ts] = [ltp, state["ema9"], state["ema21"]]
                state["history"] = hist.tail(80)

                # Start/update the currently forming 5-minute option candle.
                if state["live_candle_time"] != candle_start:
                    previous_candle_time = state["live_candle_time"]
                    # Finalize the candle that just ended and refresh indicators.
                    if previous_candle_time is not None and state["live_open"] is not None:
                        completed = pd.DataFrame(
                            [{
                                "Open": float(state["live_open"]),
                                "High": float(state["live_high"]),
                                "Low": float(state["live_low"]),
                                "Close": float(state["last_ltp"] if state["last_ltp"] is not None else ltp),
                                "Volume": 0.0,
                            }],
                            index=[previous_candle_time],
                        )
                        state["candles"] = pd.concat([state["candles"], completed]).tail(160)
                        cdf = state["candles"].copy()
                        cdf["EMA9"] = cdf["Close"].ewm(span=EMA_FAST, adjust=False).mean()
                        cdf["EMA21"] = cdf["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
                        cdf["Resistance"] = cdf["High"].rolling(20, min_periods=5).max().shift(1)
                        state["ema9"] = float(cdf["EMA9"].iloc[-1])
                        state["ema21"] = float(cdf["EMA21"].iloc[-1])
                        state["resistance"] = (
                            float(cdf["Resistance"].iloc[-1])
                            if pd.notna(cdf["Resistance"].iloc[-1]) else np.nan
                        )
                        state["history"] = cdf[["Close", "EMA9", "EMA21"]].tail(80)
                    # Freeze the previous retest candle at its actual boundary.
                    if state["retest_time"] is not None and not state["retest_closed"]:
                        if candle_start > state["retest_time"]:
                            state["retest_closed"] = True
                    state["live_candle_time"] = candle_start
                    state["live_open"] = ltp
                    state["live_high"] = ltp
                    state["live_low"] = ltp
                else:
                    state["live_high"] = max(float(state["live_high"]), ltp)
                    state["live_low"] = min(float(state["live_low"]), ltp)

                # If a retest is already active, update its live candle extremes.
                if state["state"] == "RETEST_FORMING" and state["live_candle_time"] == state["retest_time"]:
                    state["retest_high"] = max(float(state["retest_high"]), ltp)
                    state["retest_low"] = min(float(state["retest_low"]), ltp)

                ema9, ema21 = state["ema9"], state["ema21"]
                if ema9 <= ema21:
                    if state["state"] in ("RETEST_FORMING", "WAITING_FOR_BREAKOUT"):
                        state["state"] = "CANCELLED_EMA21_BREAK"
                    continue

                # Intrabar retest: price trades down to EMA9 (within one 0.05 tick)
                # while remaining above EMA21. The alert is one per 5-minute candle.
                if state["state"] == "WAITING_FOR_RETEST":
                    if ltp <= ema9 + 0.05 and ltp > ema21:
                        state.update({
                            "state": "RETEST_FORMING",
                            "retest_time": candle_start,
                            "retest_high": float(state["live_high"]),
                            "retest_low": float(state["live_low"]),
                            "retest_closed": False,
                            "entry_sent": False,
                        })
                        key = f"RETEST|{token}|{candle_start.isoformat()}"
                        if claim_alert_once(key):
                            msg = format_retest_alert(
                                state["symbol"], ltp, ema9, ema21,
                                state["retest_high"], state["retest_low"], candle_start
                            )
                            ok, detail, message_id = send_retest_telegram(msg)
                            if ok:
                                state["retest_message_id"] = message_id
                                retest_alerts += 1
                            else:
                                release_alert_claim(key)
                                quote_errors.append({"Contract": state["symbol"], "Error": detail})

                elif state["state"] == "RETEST_FORMING":
                    # Fail the retest immediately if EMA21 is lost before candle close.
                    if ltp <= ema21:
                        state["state"] = "CANCELLED_EMA21_BREAK"
                        cancel_text = (
                            f"STOCK TOOLS - RETEST CANCELLED\n\n{state['symbol']}\n"
                            f"Price fell to/below EMA 21 (₹{ema21:.2f}) before candle close.\n"
                            "No entry alert will be generated for this retest."
                        )
                        if state.get("retest_message_id"):
                            send_retest_telegram(cancel_text, state["retest_message_id"])
                    elif state.get("retest_message_id"):
                        update_text = format_retest_alert(
                            state["symbol"], ltp, ema9, ema21,
                            float(state["retest_high"]), float(state["retest_low"]),
                            state["retest_time"]
                        )
                        ok, detail, updated_id = send_retest_telegram(
                            update_text, state["retest_message_id"]
                        )
                        if ok:
                            state["retest_message_id"] = updated_id
                        else:
                            quote_errors.append({"Contract": state["symbol"], "Error": detail})

                elif state["state"] == "WAITING_FOR_BREAKOUT":
                    # Only a subsequent candle may trigger the entry.
                    if state["retest_closed"] and ltp > float(state["retest_high"]) and not state["entry_sent"]:
                        entry = float(state["retest_high"])
                        stop = float(state["retest_low"])
                        risk = entry - stop
                        target = entry + rr * risk
                        resistance = state.get("resistance", np.nan)
                        # Avoid issuing a target that sits beyond the recent option resistance.
                        if risk <= 0 or (pd.notna(resistance) and resistance >= entry and target > resistance):
                            state["state"] = "SKIPPED_TARGET_ROOM"
                            continue
                        key = f"ENTRY|{token}|{pd.Timestamp(state['retest_time']).isoformat()}"
                        if claim_alert_once(key):
                            msg = format_entry_alert(
                                state["symbol"], entry, stop, target, rr, ltp
                            )
                            ok, detail = send_entry_telegram(msg)
                            if ok:
                                state["entry_sent"] = True
                                state["state"] = "ENTRY_ALERT_SENT"
                                entry_alerts += 1
                            else:
                                release_alert_claim(key)
                                quote_errors.append({"Contract": state["symbol"], "Error": detail})

                # At the first quote after the retest candle boundary, freeze H/L.
                if (state["state"] == "RETEST_FORMING"
                        and state["retest_time"] is not None
                        and candle_start > state["retest_time"]):
                    state["retest_closed"] = True
                    state["state"] = "WAITING_FOR_BREAKOUT"
                    if state.get("retest_message_id"):
                        closed_text = (
                            "STOCK TOOLS - RETEST CANDLE CLOSED\n\n"
                            f"{state['symbol']}\n"
                            f"Final retest candle High: ₹{float(state['retest_high']):.2f}\n"
                            f"Final retest candle Low : ₹{float(state['retest_low']):.2f}\n\n"
                            f"Watch breakout above ₹{float(state['retest_high']):.2f}.\n"
                            "Entry alert will be sent only if a later live quote breaks this High.\n"
                            "This is not a buy signal yet."
                        )
                        ok, detail, updated_id = send_retest_telegram(
                            closed_text, state["retest_message_id"]
                        )
                        if ok:
                            state["retest_message_id"] = updated_id
                        else:
                            quote_errors.append({"Contract": state["symbol"], "Error": detail})

                state["last_ltp"] = ltp

            # Reset failed/used setup only on a fresh EMA regime after a new completed
            # candle refresh; do not repeatedly alert the same retest candle.
            duration = round(time.time() - cycle_started, 2)
            live_rows = []
            chart_data = {}
            for token, state in baselines.items():
                if state["state"] in ("RETEST_FORMING", "WAITING_FOR_BREAKOUT",
                                      "CANCELLED_EMA21_BREAK", "SKIPPED_TARGET_ROOM",
                                      "ENTRY_ALERT_SENT"):
                    live_rows.append({
                        "Contract": state["symbol"],
                        "State": state["state"],
                        "Live LTP": state["last_ltp"],
                        "EMA 9": state["ema9"],
                        "EMA 21": state["ema21"],
                        "Retest candle": state["retest_time"],
                        "Retest High": state["retest_high"],
                        "Retest Low": state["retest_low"],
                        "Retest closed": state["retest_closed"],
                        "Resistance": state["resistance"],
                    })
                    chart_data[state["symbol"]] = state["history"].copy()

            with monitor_lock():
                monitor_state().update({
                    "last_scan": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "last_scan_duration": duration,
                    "live_candidates": pd.DataFrame(live_rows),
                    "live_chart_data": chart_data,
                    "last_alert_count": entry_alerts,
                    "last_retest_count": retest_alerts,
                    "context": context,
                    "status": (
                        f"Live polling active | {len(quote_map)}/{len(tokens)} quotes | "
                        f"{retest_alerts} retest alert(s), {entry_alerts} entry alert(s) "
                        f"this cycle | {duration}s"
                    ),
                })
                if quote_errors:
                    old = monitor_state()["last_errors"]
                    monitor_state()["last_errors"] = pd.concat(
                        [old, pd.DataFrame(quote_errors)], ignore_index=True
                    ).tail(200)

        except Exception as exc:
            with monitor_lock():
                monitor_state()["status"] = (
                    f"Live quote error: {exc}. Retrying in {poll_seconds} seconds."
                )
        stop_event.wait(max(1, poll_seconds - (time.time() - cycle_started)))

    with monitor_lock():
        monitor_state()["running"] = False
        monitor_state()["status"] = "Stopped"

def start_background_monitor(client, contracts, lookback_days, rr, context):
    with monitor_lock():
        if monitor_state()["running"]:
            return False

        stop_event = threading.Event()
        worker = threading.Thread(
            target=_monitor_worker,
            args=(client, contracts.copy(), lookback_days, rr, context, stop_event),
            daemon=True,
            name="StockToolsOptionMonitor",
        )
        monitor_state().update({
            "running": True,
            "stop_event": stop_event,
            "thread": worker,
            "status": "Starting monitor...",
            "context": context,
            "last_alert_count": 0,
        })
        worker.start()
        return True


def stop_background_monitor():
    with monitor_lock():
        stop_event = monitor_state().get("stop_event")
        if stop_event is not None:
            stop_event.set()
        monitor_state()["status"] = "Stopping after the current API request..."


# ============================= PAPER TRADING =============================

def init_state():
    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []
    if "angel_client" not in st.session_state:
        st.session_state.angel_client = None
    if "option_cache" not in st.session_state:
        st.session_state.option_cache = {}


def update_paper_trade_prices(client: AngelOneReadOnly):
    open_trades = [t for t in st.session_state.paper_trades if t["Status"] == "OPEN"]
    if not open_trades:
        return
    quotes = client.full_quotes([t["Token"] for t in open_trades])
    if quotes.empty:
        return
    ltp_by_token = {
        str(row["symbolToken"]): float(row["ltp"])
        for _, row in quotes.iterrows()
        if pd.notna(row.get("ltp"))
    }
    for trade in open_trades:
        ltp = ltp_by_token.get(str(trade["Token"]))
        if ltp is None:
            continue
        trade["Last LTP"] = ltp
        trade["Unrealized P/L"] = (
            (ltp - trade["Entry Premium"]) * trade["Quantity"]
        )
        if ltp <= trade["Stop Premium"]:
            trade["Status"] = "STOP HIT (paper)"
            trade["Exit Premium"] = trade["Stop Premium"]
            trade["Realized P/L"] = ((trade["Exit Premium"] - trade["Entry Premium"]) * trade["Quantity"])
            send_telegram_alert(f"PAPER EXIT ALERT | {trade['Contract']}\nSTOP HIT\nExit premium: ₹{trade['Exit Premium']:.2f}\nPaper P/L: ₹{trade['Realized P/L']:.2f}\nNo broker order placed.")
        elif ltp >= trade["Target Premium"]:
            trade["Status"] = "TARGET HIT (paper)"
            trade["Exit Premium"] = trade["Target Premium"]
            trade["Realized P/L"] = ((trade["Exit Premium"] - trade["Entry Premium"]) * trade["Quantity"])
            send_telegram_alert(f"PAPER EXIT ALERT | {trade['Contract']}\nTARGET HIT\nExit premium: ₹{trade['Exit Premium']:.2f}\nPaper P/L: ₹{trade['Realized P/L']:.2f}\nNo broker order placed.")


# ============================= UI =============================

init_state()

st.title("📈 Stock Tools – Angel One Option Selector")
st.caption("NIFTY / BANKNIFTY • EMA 9/21 • Angel One market data • PAPER TRADING ONLY")

with st.sidebar:
    st.header("Strategy & risk settings")
    indices = st.multiselect("Indices", list(INSTRUMENTS), default=["NIFTY", "BANKNIFTY"])
    capital = st.number_input("Paper capital (₹)", min_value=1000, value=50000, step=5000)
    risk_percent = st.number_input("Risk budget per paper trade (%)",
                                   min_value=0.1, max_value=2.0,
                                   value=1.0, step=0.1)
    risk_budget = capital * risk_percent / 100
    st.metric("Maximum risk budget", money(risk_budget))
    rr = st.selectbox("Strategy reward/risk", [1.0, 2.0, 3.0], index=1)
    period = st.selectbox("Index history", ["5d", "30d", "60d"], index=2)
    delta_min = st.slider("Minimum option Delta", 0.10, 0.90, 0.45, 0.05)
    delta_max = st.slider("Maximum option Delta", 0.10, 0.95, 0.75, 0.05)
    max_spread = st.slider("Maximum bid/ask spread (%)", 0.2, 5.0, 1.5, 0.1)
    st.caption("Risk budget is a ceiling, not a target. If one lot exceeds it, skip the paper trade.")

tab_market, tab_options, tab_paper, tab_setup = st.tabs([
    "📊 Market & Strategy", "🔎 Option Selector", "🧪 Paper Trades", "🔐 Angel One Setup"
])

market_signals = {}
market_frames = {}

with tab_market:
    st.subheader("Index signals")
    if st.button("Refresh index candles", key="refresh_indices"):
        load_index_data.clear()

    for name in indices:
        try:
            raw = load_index_data(INSTRUMENTS[name]["yf"], period)
            df = prepare_index_data(raw)
            now = pd.Timestamp.now(tz=TZ)
            if len(df) and df.index[-1] + pd.Timedelta(minutes=5) > now:
                df = df.iloc[:-1].copy()
            if df.empty:
                st.warning(f"{name}: no completed candles.")
                continue
            sigs = find_signals(df, rr)
            trades = backtest(df, sigs)
            if sigs:
                latest_sig=sigs[-1]
                if st.button(f"Send latest {name} index paper signal to Telegram",key=f"telegram_index_{name}"):
                    evt={"time":latest_sig["time"],"stage":"INDEX CONFIRMED PAPER BUY SETUP","price":latest_sig["ref"],"stop":latest_sig["stop"],"target":latest_sig["target"],"detail":"Index EMA 9/21 crossover, retest and breakout conditions met"}
                    ok,msg=send_telegram_alert(f"PAPER ALERT | {name} INDEX\n{evt['stage']}\nTime: {evt['time']} IST\nClose: {evt['price']:.2f}\nSL: {evt['stop']:.2f} | Target: {evt['target']:.2f}\nNo broker order placed.")
                    (st.success if ok else st.warning)(msg)
            market_frames[name] = df
            market_signals[name] = sigs
            last = df.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{name} close", f"{last['Close']:,.2f}")
            c2.metric("EMA 9", f"{last['EMA9']:,.2f}")
            c3.metric("EMA 21", f"{last['EMA21']:,.2f}")
            c4.metric("Trend", "Bullish" if last["EMA9"] > last["EMA21"] else "Bearish")
            st.caption(f"Last completed candle: {df.index[-1]:%d %b %Y %H:%M} IST")
            st.line_chart(df[["Close", "EMA9", "EMA21"]].tail(150), height=260)
            if sigs:
                sig = sigs[-1]
                st.success(
                    f"Latest historical signal: {sig['time']:%d %b %H:%M} | "
                    f"Reference {sig['ref']:.2f} | SL {sig['stop']:.2f} | "
                    f"Target {sig['target']:.2f}"
                )
            else:
                st.info("No qualifying BUY signal in the selected history.")
            if trades:
                tdf = pd.DataFrame(trades)
                st.write(f"Backtest: {len(trades)} trades | "
                         f"Win rate {(tdf['R'] > 0).mean():.1%} | "
                         f"Total {tdf['R'].sum():+.2f} R")
                with st.expander(f"{name} historical backtest trades"):
                    st.dataframe(tdf, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"{name}: {exc}")

with tab_options:
    st.subheader("Angel One option-chain comparison")
    st.warning(
        "Read-only data and simulated trades only. No live orders are sent. "
        "Angel One SmartAPI access and any required static-IP/API activation must be configured first."
    )

    api = st.session_state.angel_client
    ready = bool(api and api.credentials_ready())
    if not ready:
        st.info("Complete the Angel One .env setup in the Setup tab, then click Connect.")
    else:
        if st.button("Connect / refresh Angel One session", key="angel_connect"):
            try:
                api.login()
                st.success("Angel One read-only session connected.")
            except Exception as exc:
                st.error(f"Angel One connection failed: {exc}")

    if ready and indices:
        selected_index = st.selectbox("Choose index for option selection", indices)
        use_latest_signal = st.checkbox("Use latest EMA BUY signal as underlying reference", value=True)
        df_for_index = market_frames.get(selected_index)
        sig_for_index = (market_signals.get(selected_index) or [None])[-1]
        if use_latest_signal and sig_for_index:
            spot = float(sig_for_index["ref"])
            index_stop = float(sig_for_index["stop"])
            index_target = float(sig_for_index["target"])
            signal_time = sig_for_index["time"]
            st.success(f"Using BUY signal at {signal_time:%d %b %H:%M} IST")
        elif df_for_index is not None:
            spot = float(df_for_index["Close"].iloc[-1])
            index_stop = None
            index_target = None
        else:
            spot = st.number_input("Enter index spot/reference manually", min_value=1.0, value=25000.0)
            index_stop = None
            index_target = None

        if st.button("Load Angel One contracts, Greeks & quotes", key="load_option_chain"):
            try:
                if not api.jwt:
                    api.login()
                master = download_instrument_master()
                available = get_option_contracts(
                    master, selected_index, date.today(), spot
                )
                all_expiries = sorted(
                    master[
                        (master["exch_seg"] == "NFO")
                        & (master["instrumenttype"] == "OPTIDX")
                        & (master["name"] == selected_index)
                        & (master["expiry_date"] >= date.today())
                    ]["expiry_date"].dropna().unique().tolist()
                )
                if not all_expiries:
                    raise RuntimeError("No future option expiries found in instrument master.")
                st.session_state.option_cache["master"] = master
                st.session_state.option_cache["index"] = selected_index
                st.session_state.option_cache["spot"] = spot
                st.session_state.option_cache["expiries"] = all_expiries
                st.session_state.option_cache["index_stop"] = index_stop
                st.session_state.option_cache["index_target"] = index_target
                st.session_state.option_cache["loaded"] = False
                st.success(f"Found {len(all_expiries)} available expiry dates. Select one below.")
            except Exception as exc:
                st.error(f"Could not load Angel One contracts: {exc}")

        cache = st.session_state.option_cache
        if cache.get("index") == selected_index and cache.get("expiries"):
            expiry = st.selectbox(
                "Available expiry (from Angel One instrument master)",
                cache["expiries"],
                format_func=lambda d: d.strftime("%d %b %Y")
            )
            st.markdown("#### Scan ALL listed option strikes (independent of selection)")
            st.caption("Start Monitoring loads completed 5-minute candles once, then polls Angel One live FULL quotes in batches (default 10-second cycle; set STOCKTOOLS_POLL_SECONDS in .env). Retest alerts are intrabar; Entry alerts require the retest candle to close and a later price breakout. Quote polling is not a tick-by-tick WebSocket feed. No trades are placed.")
            scan_sides = st.selectbox("Option types to scan", ["CE + PE", "CE", "PE"], key="all_strike_sides")
            scan_lookback = st.selectbox("Historical candle lookback (trading days)", [1, 3, 5], index=1, key="all_strike_lookback")
            all_contracts = all_expiry_option_contracts(cache["master"], selected_index, expiry, scan_sides)
            st.write(f"**{len(all_contracts)} contracts available** for {selected_index}, {expiry:%d %b %Y}, {scan_sides}. No strike-distance or liquidity filters are applied.")
            monitor_context = (selected_index, str(expiry), scan_sides)
            monitor_col1, monitor_col2 = st.columns(2)

            with monitor_col1:
                if st.button(
                    "▶ Start Monitoring",
                    key="start_all_strike_monitor",
                    disabled=all_contracts.empty
                ):
                    try:
                        if not api.jwt:
                            api.login()
                        started = start_background_monitor(
                            api, all_contracts, scan_lookback, rr, monitor_context
                        )
                        if started:
                            st.success("Monitoring started. The first full scan is running in the background.")
                        else:
                            st.info("A monitor is already running. Stop it before starting another.")
                    except Exception as exc:
                        st.error(f"Could not start monitoring: {exc}")

            with monitor_col2:
                if st.button("■ Stop Monitoring", key="stop_all_strike_monitor"):
                    stop_background_monitor()
                    st.info("Stop requested. The current scan will finish its current request, then stop.")

            with monitor_lock():
                monitor_running = monitor_state()["running"]
                monitor_status = monitor_state()["status"]
                monitor_last_scan = monitor_state()["last_scan"]
                monitor_duration = monitor_state()["last_scan_duration"]
                monitor_results = monitor_state()["last_results"].copy()
                monitor_events = list(monitor_state()["last_events"])
                monitor_errors = monitor_state()["last_errors"].copy()
                monitor_context_saved = monitor_state()["context"]
                monitor_alert_count = monitor_state()["last_alert_count"]
                monitor_retest_count = monitor_state().get("last_retest_count", 0)
                live_candidates = monitor_state().get("live_candidates", pd.DataFrame()).copy()
                live_chart_data = dict(monitor_state().get("live_chart_data", {}))

            if monitor_running and st_autorefresh is not None:
                st_autorefresh(interval=5000, key="stock_tools_monitor_refresh")
            elif monitor_running and st_autorefresh is None:
                st.info("Background alerts are active. Install streamlit-autorefresh to refresh this status automatically.")

            st.info(f"**Monitor status:** {monitor_status}")
            if monitor_last_scan:
                st.caption(
                    f"Last scan: {monitor_last_scan} | "
                    f"Duration: {monitor_duration}s | "
                    f"Entry alerts this cycle: {monitor_alert_count} | Retest alerts this cycle: {monitor_retest_count}"
                )

            if monitor_context_saved == monitor_context:
                st.markdown("#### Live retest / breakout candidates")
                if isinstance(live_candidates, pd.DataFrame) and not live_candidates.empty:
                    st.dataframe(live_candidates, use_container_width=True, hide_index=True)
                    chart_names = live_candidates["Contract"].dropna().astype(str).unique().tolist()
                    live_chart_symbol = st.selectbox(
                        "Live candidate chart", chart_names, key="live_candidate_chart_symbol"
                    )
                    live_chart = live_chart_data.get(live_chart_symbol)
                    if isinstance(live_chart, pd.DataFrame) and not live_chart.empty:
                        st.line_chart(live_chart[["Close", "EMA9", "EMA21"]], height=300)
                else:
                    st.info("No active retest or breakout candidates at the moment.")

                results = monitor_results
                if isinstance(results, pd.DataFrame) and not results.empty:
                    display_results = results.sort_values("Latest events", ascending=False).copy()
                    trend_help = "Bullish = EMA 9 is above EMA 21. Bearish = EMA 9 is below EMA 21."
                    column_help = {
                        "Contract": "Option name received from Angel One.",
                        "Strike": "The option strike price.",
                        "Side": "CE means Call option. PE means Put option.",
                        "Last candle": "Time of the latest completed 5-minute candle.",
                        "Close": "The option premium when that 5-minute candle closed.",
                        "EMA9": "Average price that follows the recent 9 candles more closely.",
                        "EMA21": "Average price based on the recent 21 candles.",
                        "Support": "A recent lower price area where the option may find buying interest.",
                        "Resistance": "A recent upper price area the option may struggle to cross.",
                        "Trend": trend_help,
                        "Target": "Estimated strategy target. It is shown only when a BUY setup is confirmed. It is not guaranteed to be reached.",
                        "Latest events": "The strategy event found on the latest completed candle. None means no new event."
                    }
                    trend_colors = lambda value: (
                        "color: #16803c; font-weight: 700"
                        if value == "Bullish" else
                        "color: #dc2626; font-weight: 700"
                        if value == "Bearish" else ""
                    )

                    trend_style = display_results.style.map(
                        trend_colors,
                        subset=["Trend"]
                    )
                    table_column_config = {
                        col: st.column_config.Column(col, help=help_text)
                        for col, help_text in column_help.items()
                    }
                    table_column_config["Target"] = st.column_config.NumberColumn(
                        "Target (₹)", help=column_help["Target"], format="₹%.2f"
                    )
                    st.dataframe(
                        trend_style,
                        column_config=table_column_config,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.markdown("#### Strategy chart")
                    st.caption("Choose one option to see its 5-minute closing price, EMA 9, EMA 21, support and resistance.")
                    chart_symbols = display_results["Contract"].dropna().astype(str).unique().tolist()
                    chart_symbol = st.selectbox(
                        "Option for chart",
                        chart_symbols,
                        key="monitor_strategy_chart_symbol"
                    )
                    if st.button("Load strategy chart", key="load_monitor_strategy_chart"):
                        try:
                            selected_contract = all_contracts[
                                all_contracts["symbol"].astype(str) == chart_symbol
                            ]
                            if selected_contract.empty:
                                st.warning("This option is not in the currently selected expiry.")
                            else:
                                chart_token = str(selected_contract.iloc[0]["token"])
                                chart_end = datetime.now(pd.Timestamp.now(tz=TZ).tzinfo)
                                chart_start = chart_end - timedelta(days=scan_lookback * 2 + 3)
                                chart_raw = api.historical_candles(chart_token, chart_start, chart_end)
                                if not chart_raw.empty:
                                    chart_now = pd.Timestamp.now(tz=TZ)
                                    chart_raw = chart_raw.loc[
                                        chart_raw.index + pd.Timedelta(minutes=5) <= chart_now
                                    ]
                                chart_data, _ = analyze_option_candles(chart_raw, rr)
                                st.session_state["monitor_strategy_chart_data"] = chart_data
                                st.session_state["monitor_strategy_chart_symbol"] = chart_symbol
                        except Exception as exc:
                            st.error(f"Could not load strategy chart: {exc}")

                    saved_chart = st.session_state.get("monitor_strategy_chart_data")
                    saved_symbol = st.session_state.get("monitor_strategy_chart_symbol")
                    if isinstance(saved_chart, pd.DataFrame) and not saved_chart.empty and saved_symbol == chart_symbol:
                        st.line_chart(
                            saved_chart[["Close", "EMA9", "EMA21", "Support", "Resistance"]].tail(120),
                            height=360
                        )
                        st.caption("How to read: price above EMA 9 and EMA 21 is stronger; EMA 9 above EMA 21 means bullish trend. Support and resistance are calculated from recent candles. These are estimates, not guarantees.")

                    st.download_button(
                        "Download monitoring scan CSV",
                        results.to_csv(index=False),
                        file_name="all_strike_monitor_scan.csv",
                        mime="text/csv"
                    )

                if monitor_events:
                    st.markdown("**Confirmed BUY setups from the latest scan**")
                    st.dataframe(
                        pd.DataFrame(monitor_events),
                        use_container_width=True,
                        hide_index=True
                    )

                if isinstance(monitor_errors, pd.DataFrame) and not monitor_errors.empty:
                    with st.expander(f"Failed/insufficient-data contracts ({len(monitor_errors)})"):
                        st.dataframe(monitor_errors, use_container_width=True, hide_index=True)

            st.divider()
            if st.button("Compare CE options for selected expiry", key="compare_options"):
                try:
                    contracts = get_option_contracts(
                        cache["master"], selected_index, expiry, cache["spot"]
                    )
                    comparison = build_option_comparison(
                        api, contracts, selected_index, expiry, cache["spot"],
                        delta_min, delta_max, max_spread
                    )
                    st.session_state.option_cache["comparison"] = comparison
                    st.session_state.option_cache["expiry"] = expiry
                    st.session_state.option_cache["loaded"] = True
                except Exception as exc:
                    st.error(f"Option comparison failed: {exc}")

        comparison = cache.get("comparison")
        if cache.get("loaded") and isinstance(comparison, pd.DataFrame) and not comparison.empty:
            show_cols = [
                "symbol", "strike", "lotsize", "ltp", "delta", "theta",
                "impliedVolatility", "quote_volume", "open_interest",
                "best_bid", "best_ask", "spread_pct", "qualifies", "score"
            ]
            show_cols = [c for c in show_cols if c in comparison.columns]
            st.dataframe(
                comparison[show_cols].rename(columns={
                    "symbol": "Contract", "strike": "Strike",
                    "lotsize": "Lot size", "ltp": "LTP",
                    "delta": "Delta", "theta": "Theta",
                    "impliedVolatility": "IV", "quote_volume": "Volume",
                    "open_interest": "Open interest", "best_bid": "Best bid",
                    "best_ask": "Best ask", "spread_pct": "Spread %",
                    "qualifies": "Passes filters", "score": "Score"
                }),
                use_container_width=True, hide_index=True
            )
            eligible = comparison[comparison["qualifies"] & comparison["ltp"].notna()].copy()
            if eligible.empty:
                st.warning(
                    "No contract passed all selected filters. Do not force a trade. "
                    "Relax filters only after reviewing liquidity and spreads."
                )
            else:
                eligible["display"] = eligible.apply(
                    lambda r: f"{r['symbol']} | LTP ₹{r['ltp']:.2f} | "
                              f"Delta {r['delta']:.2f} | Spread "
                              f"{r['spread_pct']:.2f}%",
                    axis=1
                )
                chosen_label = st.selectbox("Select a qualifying contract to paper trade",
                                            eligible["display"].tolist())
                chosen = eligible[eligible["display"] == chosen_label].iloc[0]
                st.markdown("#### Selected option chart analysis (5-minute candles)")
                st.caption("This independently analyses the selected CE contract. Index signals remain separate; option signals are not blocked by the index crossover state.")
                scan_days = st.selectbox("Option candle lookback", [1, 3, 5], index=2, format_func=lambda x: f"{x} trading-day(s)")
                if st.button("Scan selected CE candles & send new alerts", key="scan_option_candles"):
                    try:
                        end_dt=datetime.now().astimezone()
                        start_dt=end_dt-timedelta(days=scan_days*2+1)
                        option_raw=api.historical_candles(str(chosen["token"]),start_dt,end_dt)
                        # Exclude the currently forming 5-minute candle.
                        if len(option_raw) and option_raw.index[-1]+pd.Timedelta(minutes=5)>pd.Timestamp.now(tz=TZ):
                            option_raw=option_raw.iloc[:-1]
                        enriched, option_events=analyze_option_candles(option_raw,rr)
                        st.session_state.option_cache["option_candles"]=enriched
                        st.session_state.option_cache["option_events"]=option_events
                        st.session_state.option_cache["option_contract_label"]=str(chosen["symbol"])
                        sent=dispatch_new_alerts(option_events,str(chosen["symbol"]))
                        st.success(f"Scanned {len(enriched)} completed candles. Found {len(option_events)} setup events; {sent} new alert(s) processed.")
                    except Exception as exc:
                        st.error(f"Option candle scan failed: {exc}")
                option_chart=st.session_state.option_cache.get("option_candles")
                if isinstance(option_chart,pd.DataFrame) and not option_chart.empty and st.session_state.option_cache.get("option_contract_label")==str(chosen["symbol"]):
                    st.line_chart(option_chart[["Close","EMA9","EMA21"]].tail(150),height=280)
                    last_opt=option_chart.iloc[-1]
                    m1,m2,m3,m4=st.columns(4)
                    m1.metric("Option LTP / close",money(last_opt["Close"]))
                    m2.metric("EMA 9",money(last_opt["EMA9"]))
                    m3.metric("EMA 21",money(last_opt["EMA21"]))
                    m4.metric("Trend","Bullish" if last_opt["EMA9"]>last_opt["EMA21"] else "Bearish")
                    st.write(f"**Auto resistance:** {money(last_opt['Resistance'])}  |  **Auto support:** {money(last_opt['Support'])}")
                    ev=st.session_state.option_cache.get("option_events",[])
                    if ev:
                        st.dataframe(pd.DataFrame(ev).tail(30),use_container_width=True,hide_index=True)
                    else:
                        st.info("No crossover, retest, breakout or confirmed entry events found in the downloaded option candles.")
                st.markdown("#### Paper trade risk preview")
                option_entry = st.number_input(
                    "Simulated option entry premium (₹)",
                    min_value=0.05, value=float(chosen["ltp"]), step=0.05
                )
                lot_size = int(chosen["lotsize"])
                delta_value = abs(float(chosen["delta"]))
                if cache.get("index_stop") is not None and cache.get("index_target") is not None:
                    idx_risk = cache["spot"] - cache["index_stop"]
                    idx_reward = cache["index_target"] - cache["spot"]
                    opt_risk_per_unit = delta_value * idx_risk
                    opt_reward_per_unit = delta_value * idx_reward
                    option_stop = max(0.05, option_entry - opt_risk_per_unit)
                    option_target = option_entry + opt_reward_per_unit
                    st.caption(
                        "Option stop/target are Delta-based estimates from index levels. "
                        "They are not exact option-price levels; paper-trade LTP checks can gap."
                    )
                else:
                    stop_pct = st.number_input(
                        "Simulated option stop distance (%)",
                        min_value=1.0, max_value=80.0, value=20.0, step=1.0
                    )
                    reward_multiple = st.number_input(
                        "Option reward/risk", min_value=1.0, max_value=5.0,
                        value=2.0, step=0.5
                    )
                    opt_risk_per_unit = option_entry * stop_pct / 100
                    opt_reward_per_unit = opt_risk_per_unit * reward_multiple
                    option_stop = option_entry - opt_risk_per_unit
                    option_target = option_entry + opt_reward_per_unit

                risk_per_lot = opt_risk_per_unit * lot_size
                affordable_lots = int(risk_budget // risk_per_lot) if risk_per_lot > 0 else 0
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Lot size", lot_size)
                p2.metric("Estimated risk / lot", money(risk_per_lot))
                p3.metric("Risk budget", money(risk_budget))
                p4.metric("Affordable lots", affordable_lots)
                st.write(
                    f"**Simulated levels:** Entry {money(option_entry)} | "
                    f"Stop {money(option_stop)} | Target {money(option_target)}"
                )
                if affordable_lots < 1:
                    st.error("One lot exceeds your selected risk budget. Skip this paper trade.")
                else:
                    lots_to_simulate = st.number_input(
                        "Number of paper lots", min_value=1,
                        max_value=affordable_lots, value=1, step=1
                    )
                    quantity = int(lots_to_simulate) * lot_size
                    if st.button("▶ Start PAPER trade (no broker order)", type="primary"):
                        paper = {
                            "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Index": selected_index,
                            "Contract": str(chosen["symbol"]),
                            "Token": str(chosen["token"]),
                            "Expiry": expiry.strftime("%d %b %Y"),
                            "Lots": int(lots_to_simulate),
                            "Lot size": lot_size,
                            "Quantity": quantity,
                            "Entry Premium": float(option_entry),
                            "Stop Premium": float(option_stop),
                            "Target Premium": float(option_target),
                            "Last LTP": float(chosen["ltp"]),
                            "Unrealized P/L": 0.0,
                            "Realized P/L": 0.0,
                            "Status": "OPEN",
                        }
                        st.session_state.paper_trades.append(paper)
                        st.success("Paper trade added. No order was sent to Angel One.")

with tab_paper:
    st.subheader("Paper trade ledger")
    st.caption("Simulated trades only. No real positions or orders are created.")
    if st.button("Refresh paper trade LTP / check stop & target", key="refresh_paper"):
        client = st.session_state.angel_client
        if client and client.credentials_ready():
            try:
                if not client.jwt:
                    client.login()
                update_paper_trade_prices(client)
                st.success("Paper trade prices refreshed.")
            except Exception as exc:
                st.error(f"Could not refresh paper prices: {exc}")
        else:
            st.warning("Connect Angel One in the Setup tab first.")

    paper = st.session_state.paper_trades
    if paper:
        pdf = pd.DataFrame(paper)
        open_pl = pdf.loc[pdf["Status"] == "OPEN", "Unrealized P/L"].sum()
        closed_pl = pdf.loc[pdf["Status"] != "OPEN", "Realized P/L"].sum()
        a, b, c = st.columns(3)
        a.metric("Paper trades", len(pdf))
        b.metric("Open unrealized P/L", money(open_pl))
        c.metric("Closed realized P/L", money(closed_pl))
        st.dataframe(pdf.drop(columns=["Token"]), use_container_width=True, hide_index=True)

        open_indices = [i for i, t in enumerate(paper) if t["Status"] == "OPEN"]
        if open_indices:
            selected_trade_idx = st.selectbox(
                "Select open paper trade to close manually",
                open_indices,
                format_func=lambda i: (
                    f"{paper[i]['Contract']} | Entry ₹{paper[i]['Entry Premium']:.2f} "
                    f"| Last ₹{paper[i]['Last LTP']:.2f}"
                )
            )
            manual_exit = st.number_input(
                "Manual paper exit premium (₹)", min_value=0.05,
                value=max(0.05, float(paper[selected_trade_idx]["Last LTP"])),
                step=0.05
            )
            if st.button("Close selected PAPER trade"):
                trade = paper[selected_trade_idx]
                trade["Exit Premium"] = float(manual_exit)
                trade["Realized P/L"] = (
                    (manual_exit - trade["Entry Premium"]) * trade["Quantity"]
                )
                trade["Status"] = "CLOSED (manual paper)"
                send_telegram_alert(f"PAPER EXIT ALERT | {trade['Contract']}\nMANUAL PAPER EXIT\nExit premium: ₹{manual_exit:.2f}\nPaper P/L: ₹{trade['Realized P/L']:.2f}\nNo broker order placed.")
                st.success("Paper trade closed. No broker order was sent.")
        csv_data = pdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download paper ledger CSV", csv_data,
                           "angel_one_paper_trades.csv", "text/csv")
        if st.button("Clear all paper trades", type="secondary"):
            st.session_state.paper_trades = []
            st.rerun()
    else:
        st.info("No paper trades yet. Select a contract in the Option Selector tab.")

with tab_setup:
    st.subheader("Angel One SmartAPI setup")
    st.markdown(
        "Create/activate a SmartAPI app in your Angel One account. "
        "The app uses login, option Greeks and market quotes only."
    )
    st.markdown(
        "Create a local `.env` file in the same folder as `dashboard.py` "
        "and add the following keys:"
    )
    st.code(
        "ANGEL_API_KEY=your_smartapi_api_key\n"
        "ANGEL_CLIENT_CODE=your_client_code\n"
        "ANGEL_PIN=your_trading_pin\n"
        "ANGEL_TOTP_SECRET=your_authenticator_base32_secret\n"
        "ANGEL_CLIENT_PUBLIC_IP=your_registered_static_public_ip\n"
        "TELEGRAM_BOT_TOKEN=your_telegram_bot_token\n"
        "TELEGRAM_RETEST_CHAT_ID=-5588683077\n"
        "TELEGRAM_ENTRY_CHAT_ID=-5122452181\n"
        "STOCKTOOLS_POLL_SECONDS=10\n"
        "# Optional if automatic local IP detection is wrong:\n"
        "ANGEL_CLIENT_LOCAL_IP=192.168.1.10",
        language="bash"
    )
    st.warning(
        "Do not paste API keys, PINs, TOTP secrets, QR codes or tokens into ChatGPT, "
        "GitHub or screenshots. Keep .env in .gitignore. Confirm SmartAPI activation, "
        "IP requirements and current API policies directly with Angel One."
    )
    st.write("Local IP detected:", local_ip())
    st.write("MAC address detected:", mac_address())
    st.write("Public IP configured:", "Yes" if os.getenv("ANGEL_CLIENT_PUBLIC_IP") else "No")
    if st.button("Connect to Angel One (read-only)"):
        try:
            client = AngelOneReadOnly()
            client.login()
            st.session_state.angel_client = client
            st.success("Angel One session connected. This app has no order functions.")
        except Exception as exc:
            st.error(f"Connection failed: {exc}")
    if st.session_state.angel_client and st.session_state.angel_client.jwt:
        st.success("Angel One session is active in this Streamlit session.")

st.divider()
st.caption(
    "Educational paper-trading tool only. Index and every listed CE/PE strike for the selected expiry can be analysed independently. No order placement code exists in this file. "
    "Market data may be delayed or unavailable. Live quote polling is periodic, not tick-by-tick. Greeks and quotes can change quickly. "
    "Delta-based stop/target estimates are approximate; option premiums are affected by "
    "spread, volatility, theta, liquidity and gaps. Paper simulation does not guarantee "
    "real execution prices or future results. Verify expiry, strike and lot size with Angel One."
)
