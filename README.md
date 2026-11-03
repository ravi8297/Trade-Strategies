# Stock Tools — Angel One Options Strategy Dashboard

A Streamlit-based market analysis and paper-trading dashboard for **NIFTY** and **BANKNIFTY** options. It combines 5-minute EMA 9/EMA 21 strategy analysis, Angel One SmartAPI market data, option-chain comparison, live quote monitoring, Telegram notifications, and a local paper-trade ledger.

> **Important:** This project is for market analysis and paper trading only. It does not place, modify, or cancel orders with Angel One. Alerts are informational and are not investment advice or a guarantee of profit.

---

## Contents

- [Overview](#overview)
- [Key features](#key-features)
- [How the strategy works](#how-the-strategy-works)
- [Live monitoring and Telegram alerts](#live-monitoring-and-telegram-alerts)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Installation on Windows](#installation-on-windows)
- [Environment configuration](#environment-configuration)
- [Running the dashboard](#running-the-dashboard)
- [Dashboard sections](#dashboard-sections)
- [Paper trading](#paper-trading)
- [Project structure](#project-structure)
- [Security and GitHub checklist](#security-and-github-checklist)
- [Troubleshooting](#troubleshooting)
- [Limitations and risk notice](#limitations-and-risk-notice)

---

## Overview

Stock Tools is designed to help monitor NIFTY and BANKNIFTY option contracts using a 5-minute EMA crossover-and-retest approach.

The application can:

1. Load index candles and calculate EMA 9, EMA 21, ATR, support, and resistance.
2. Connect to Angel One SmartAPI for read-only option market data.
3. Load the Angel One instrument master and identify option contracts for a selected expiry.
4. Compare eligible option contracts using Greeks, bid/ask spread, volume, and open interest.
5. Scan listed CE and PE contracts for the selected index and expiry.
6. Monitor live option quotes periodically for EMA 9 retests and subsequent breakouts.
7. Send retest and entry notifications to separate Telegram groups.
8. Maintain simulated paper trades and track their indicative P/L.

The app is built as a single-file Streamlit application (`dashboard.py`) with local configuration stored in `.env`.

## Key features

### 1. Index market and strategy view

- Supports NIFTY and BANKNIFTY.
- Loads 5-minute index candles through Yahoo Finance.
- Calculates EMA 9 and EMA 21.
- Calculates ATR for strategy/risk analysis.
- Displays price, EMA values, trend, and historical signals.
- Includes a historical backtest view for the index strategy.

Yahoo Finance data is used for the index chart/analysis section. Angel One is used for the options market-data functions.

### 2. Angel One option selector

- Authenticates with Angel One SmartAPI using credentials supplied locally.
- Downloads the Angel One instrument master.
- Lists available expiries for the selected index.
- Loads option Greeks and quote information.
- Compares CE contracts using:
  - Delta
  - Theta
  - Implied volatility
  - Trading volume
  - Open interest
  - Best bid and ask
  - Bid/ask spread percentage
- Applies configurable Delta and spread filters.
- Shows a transparent ranking score to help compare contracts.

The comparison score is a screening aid, not a prediction of future performance.

### 3. All-strike option monitoring

For a selected index, expiry, and option side selection, the monitor can load all matching listed option contracts rather than limiting the scan to only ATM contracts.

Available side selections:

- `CE + PE`
- `CE`
- `PE`

The monitor bootstraps its analysis from completed 5-minute historical candles and then checks Angel One FULL quotes in batches.

### 4. Separate Telegram retest and entry alerts

The application supports two Telegram destinations:

- **Stock Tools - Retest Alerts:** early notification when a live option price touches EMA 9 while remaining above EMA 21.
- **Stock Tools - Entry Alerts:** notification only after the retest candle has closed and a later live quote breaks above the frozen retest-candle High.

The retest message is edited as the candle forms so its High and Low can be refreshed. If EMA 21 is lost before the retest candle closes, the setup is cancelled. Once the candle closes, its High and Low are frozen for the breakout check.

An entry alert includes:

- Contract
- Retest-candle High as the reference buy level
- Retest-candle Low as the stop-loss reference
- Selected risk-reward ratio
- Calculated target
- Current quote context

The calculated target is checked against the latest available prior-candle resistance. The monitor skips a candidate when the target would extend beyond that resistance, rather than forcing an unsuitable target.

**Polling note:** Live quotes are polled periodically. The default interval is 10 seconds and can be changed with `STOCKTOOLS_POLL_SECONDS`. This is not a tick-by-tick WebSocket feed; a fast price move between polling cycles can be missed. The actual execution price may differ from the reference level, especially if the market gaps.

### 5. Paper trading ledger

- Create simulated option trades from the option selector.
- Set the number of paper lots.
- Review entry premium, stop, target, quantity, and indicative P/L.
- Refresh quotes to check whether a simulated stop or target has been reached.
- Close a paper trade manually.
- Export the paper-trade ledger to CSV.

Paper trades are stored in the Streamlit session state in the current implementation. They should not be treated as a durable broker or exchange trade record.

---

## How the strategy works

The core option strategy uses 5-minute candles and EMA 9/EMA 21.

### Stage 1 — Bullish trend

The strategy looks for EMA 9 to cross above EMA 21. This is treated as a possible bullish setup, not an entry by itself.

### Stage 2 — EMA 9 retest

During live monitoring, the application checks the current option LTP against the EMA 9 calculated from the completed candle history. A retest alert is generated when the price reaches the EMA 9 area while remaining above EMA 21.

The current 5-minute candle's High and Low are tracked while it is forming.

### Stage 3 — Retest validation

If price falls below EMA 21 before the retest candle closes, the retest is cancelled.

If the candle completes while the setup remains valid, its High and Low are frozen:

- Retest High: breakout reference
- Retest Low: stop-loss reference

### Stage 4 — Breakout and entry notification

The monitor waits for a later live quote to move above the frozen retest High. It then calculates:

`Risk = Retest High - Retest Low`

`Target = Retest High + (Risk × Risk-Reward Ratio)`

For example, with a reference entry of ₹127, stop-loss of ₹124, and a 1:2 risk-reward ratio:

- Risk = ₹3
- Target = ₹127 + (₹3 × 2) = ₹133

The configured reward/risk selection is used by the monitor. A candidate can be skipped if the calculated target is beyond the latest prior-candle resistance.

> Strategy outputs are rule-based observations. They do not account for every market condition, execution cost, slippage, liquidity change, or sudden volatility event.

---

## Technology stack

| Component | Purpose |
|---|---|
| Python | Application and strategy logic |
| Streamlit | Interactive dashboard |
| Pandas | Candle, quote, and tabular data processing |
| NumPy | Numerical calculations |
| Requests | Angel One and Telegram HTTP API calls |
| yfinance | Index historical candle data |
| python-dotenv | Local `.env` configuration |
| pyotp | Time-based one-time password generation for Angel One login |
| streamlit-autorefresh | Automatic dashboard refresh while monitoring |
| SQLite | Local alert de-duplication ledger |
| Angel One SmartAPI | Read-only authentication, option Greeks, historical candles, and quotes |
| Telegram Bot API | Retest and entry notifications |

---

## Requirements

- Windows 10/11, macOS, or Linux.
- Python 3.10+ recommended.
- Angel One account with SmartAPI access configured.
- Angel One API key and required login details.
- Static public IP/API activation if required by your Angel One SmartAPI configuration.
- Telegram bot and two Telegram groups.
- Internet access to retrieve market data and send notifications.

The application depends on third-party APIs and their availability, permissions, and rate limits.

## Installation on Windows

Open PowerShell and run:

```powershell
# Move to the project folder
cd C:\path\to\stock-tools

# Create a virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
python -m pip install streamlit yfinance pandas numpy requests python-dotenv pyotp streamlit-autorefresh
```

If PowerShell blocks virtual-environment activation, you can use the environment's Python executable directly:

```powershell
.\venv\Scripts\python.exe -m pip install streamlit yfinance pandas numpy requests python-dotenv pyotp streamlit-autorefresh
.\venv\Scripts\python.exe -m streamlit run dashboard.py
```

## Environment configuration

Create a file named `.env` in the same directory as `dashboard.py`.

Example:

```env
# Angel One SmartAPI
ANGEL_API_KEY=your_angel_one_api_key
ANGEL_CLIENT_CODE=your_client_code
ANGEL_PIN=your_trading_login_pin
ANGEL_TOTP_SECRET=your_totp_secret
ANGEL_CLIENT_PUBLIC_IP=your_static_public_ip

# Telegram bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Separate Telegram destinations
TELEGRAM_RETEST_CHAT_ID=your_retest_group_chat_id
TELEGRAM_ENTRY_CHAT_ID=your_entry_group_chat_id

# Live quote polling interval in seconds (minimum supported interval is 5 seconds)
STOCKTOOLS_POLL_SECONDS=10
```

Replace every placeholder with your own values. Do not copy real credentials into this README or commit them to GitHub.

### Telegram setup

1. Create a Telegram bot using BotFather.
2. Add the bot to the Retest Alerts group.
3. Add the bot to the Entry Alerts group.
4. Ensure the bot has permission to send messages. Message editing also needs to be permitted for the bot's own messages.
5. Obtain each group's chat ID.
6. Put the two chat IDs in `.env` under the matching variable names.
7. Keep the bot token private.

If Telegram alerts do not arrive, verify the token, chat IDs, bot membership, and group permissions.

### Environment template

You can commit a `.env.example` file containing placeholder values, but never commit `.env`.

Example `.env.example`:

```env
ANGEL_API_KEY=
ANGEL_CLIENT_CODE=
ANGEL_PIN=
ANGEL_TOTP_SECRET=
ANGEL_CLIENT_PUBLIC_IP=

TELEGRAM_BOT_TOKEN=
TELEGRAM_RETEST_CHAT_ID=
TELEGRAM_ENTRY_CHAT_ID=

STOCKTOOLS_POLL_SECONDS=10
```

## Running the dashboard

From the project folder with the virtual environment activated:

```powershell
python -m streamlit run dashboard.py
```

Streamlit normally opens the dashboard in your browser. If it does not, open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

To stop the app, press `Ctrl + C` in the terminal.

## Dashboard sections

### Market & Strategy

- Select NIFTY and/or BANKNIFTY.
- Refresh index candles.
- Review EMA 9, EMA 21, trend, historical signals, and charts.
- Review the historical backtest output.

### Option Selector

- Connect to Angel One.
- Load contracts, expiries, Greeks, and quotes.
- Select the index and expiry.
- Choose CE, PE, or both sides for all-strike monitoring.
- Select the historical lookback and reward/risk ratio.
- Start or stop background monitoring.
- Review scan status, active retests, entry candidates, errors, and charts.
- Download the scan results as CSV.
- Compare eligible CE contracts and create simulated paper trades.

### Paper Trades

- Refresh the LTP for open paper trades.
- Review simulated stop/target status and P/L.
- Manually close a paper trade.
- Download the paper ledger as CSV.

### Angel One Setup

Use the setup section in the app to review the required local configuration. Keep all actual credentials in `.env`.

---

## Project structure

A typical local project directory:

```text
stock-tools/
│
├── dashboard.py                 # Main Streamlit application
├── .env                         # Local secrets; do not commit
├── .env.example                 # Safe placeholder configuration
├── .gitignore                   # Excludes secrets and local runtime files
├── requirements.txt             # Python dependencies (recommended)
├── README.md                    # Project documentation
├── stock_tools_alerts.sqlite3   # Local alert de-duplication database
└── venv/                        # Python virtual environment; do not commit
```

The SQLite alert database is created locally by the application to prevent duplicate notifications across reruns and monitor activity. It is a runtime file, not source code.

### Suggested `requirements.txt`

```text
streamlit
yfinance
pandas
numpy
requests
python-dotenv
pyotp
streamlit-autorefresh
```

For reproducible deployments, pin and test dependency versions in your own environment.

## Security and GitHub checklist

Before pushing the project:

1. Confirm `.env` is excluded from Git.
2. Confirm no API key, PIN, TOTP secret, bot token, or account information is present in source files, screenshots, logs, or commit history.
3. Exclude the virtual environment, Python cache, and local SQLite database.
4. Commit `.env.example` with placeholders only.
5. Review the staged files before pushing.

Suggested `.gitignore`:

```gitignore
# Secrets
.env
.env.*
!.env.example
.streamlit/secrets.toml

# Python
venv/
.venv/
__pycache__/
*.py[cod]

# Local runtime data
stock_tools_alerts.sqlite3
*.db

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
```

If a secret was committed or pushed previously, deleting it from the latest file is not enough. Revoke or rotate the exposed credential and review the repository history.

### Git commands

```powershell
git status
git add README.md dashboard.py .env.example .gitignore
git commit -m "Document Stock Tools options monitoring dashboard"
git push origin main
```

Adjust the branch name if your repository uses a different default branch.

---

## Troubleshooting

### `ModuleNotFoundError`

Activate the correct virtual environment and install the missing package:

```powershell
python -m pip install package_name
```

Check which Python is being used:

```powershell
python -c "import sys; print(sys.executable)"
```

### Angel One login fails

Check:

- API key and client code.
- PIN and current TOTP secret.
- Public IP configuration and SmartAPI activation requirements.
- System date and time.
- Angel One service availability and API response.

Do not paste credentials or full authentication responses into public issues.

### No option contracts or quotes appear

Check the selected index, expiry, option side, and market-data availability. The instrument master and quote APIs may change or return incomplete data. Review the dashboard's error output.

### Telegram alerts are not received

Check:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_RETEST_CHAT_ID`
- `TELEGRAM_ENTRY_CHAT_ID`
- Bot membership and group permissions.
- Whether the correct bot is present in both groups.

Retest and entry notifications use separate chat destinations.

### Monitoring is slow or quote requests fail

The selected expiry can contain many contracts. Historical candle bootstrap and batched quote calls take time, and Angel One applies API limits. Avoid setting an unnecessarily aggressive polling interval. The application enforces a minimum configured polling interval of 5 seconds, but the broker's limits and response time still apply.

### Dashboard does not refresh

Install the optional refresh package:

```powershell
python -m pip install streamlit-autorefresh
```

Restart Streamlit after installing it.

---

## Limitations and risk notice

- The monitor uses periodic quote polling, not a tick-by-tick WebSocket stream.
- A rapid price move between quote polls may not be observed.
- A breakout reference price is not a guaranteed execution price.
- Slippage, spreads, liquidity, brokerage, taxes, and other charges can affect real results.
- Historical data and current quotes may be delayed, incomplete, or unavailable.
- The strategy is rule-based and may produce false signals.
- Resistance and support are calculated reference levels, not guaranteed barriers.
- The current paper ledger is local/session-based and is not a broker statement.
- API rate limits, authentication expiry, network failures, and provider changes can interrupt monitoring.
- No automated live order placement is implemented.

Use this application for learning, analysis, and paper testing. Validate the strategy independently before making any financial decision.

## Disclaimer

This software is provided for educational and informational purposes only, without warranties of any kind. You are responsible for verifying data, testing the strategy, protecting credentials, and complying with the terms and policies of Angel One, Telegram, Yahoo Finance, and any other services you use.
