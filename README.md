# Bitget Futures Trading Engine

A production-grade, modular multi-symbol futures trading engine for Bitget USDT perpetual contracts.
Supports paper trading, live trading, and backtesting from a single codebase.

---

## Architecture Overview

```
bitget_engine/
├── main.py               # Orchestrator & entry point
├── config.py             # All configuration (single source of truth)
├── logger.py             # Logging setup (file + colored console)
├── universe.py           # Symbol discovery & volume filtering
├── data_feed.py          # OHLCV fetching with thread-safe cache
├── scanner.py            # Symbol scoring & ranking
├── regime.py             # Market regime detection (TRENDING/RANGING/HIGH_VOL)
├── strategy.py           # Signal generation per regime
├── risk_manager.py       # Position sizing, heat cap, drawdown limits
├── portfolio_manager.py  # Open position state & correlation matrix
├── execution_engine.py   # Unified order router (paper + live)
├── paper_engine.py       # Simulated fills with slippage & fees
├── bitget_rest.py        # Bitget REST API client (signed requests)
├── bitget_ws.py          # Bitget WebSocket price feed
├── metrics.py            # Sharpe, win rate, drawdown, CSV export
├── backtester.py         # Single-symbol historical backtester
├── discord_notifier.py   # Discord webhook alerts
├── requirements.txt
├── .env.template
└── logs/                 # Auto-created at runtime
```

---

## Quick Start

### 1. Install dependencies

```bash
cd bitget_engine
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.template .env
```

Edit `.env`:
```
BITGET_API_KEY=your_key
BITGET_SECRET_KEY=your_secret
BITGET_PASSPHRASE=your_passphrase
TRADING_MODE=paper              # Start with paper!
DISCORD_WEBHOOK_URL=            # Optional
```

### 3. Run paper trading (safe default)

```bash
python main.py --mode paper
```

### 4. Run a backtest first

```bash
python main.py --mode backtest --symbol BTCUSDT
```

### 5. Go live (only after testing)

```bash
python main.py --mode live
```

---

## Switching Modes

| Mode | Command | Description |
|------|---------|-------------|
| Paper | `python main.py --mode paper` | Simulated fills, no real money |
| Live | `python main.py --mode live` | Real Bitget API, real money |
| Backtest | `python main.py --mode backtest` | Historical replay |

Or set `TRADING_MODE=paper` / `live` in `.env` and just run `python main.py`.

---

## Configuring Portfolio Size

Edit `config.py` → `TradingConfig`:

```python
top_n_symbols: int = 5               # Trade top 5 ranked symbols
max_concurrent_positions: int = 5    # Hard cap on open positions
risk_per_trade_pct: float = 0.01     # Risk 1% per trade
portfolio_heat_cap_pct: float = 0.05 # Max 5% total open risk
max_daily_loss_pct: float = 0.04     # Stop after 4% daily loss
default_leverage: int = 3            # 3× leverage
min_volume_24h_usdt: float = 50_000_000  # Min $50M daily volume
```

For a $10,000 account:
- Each trade risks `$10,000 × 1% = $100`
- Max open risk = `$10,000 × 5% = $500`

---

## Safety Protocol: Testing Before Going Live

### Step 1 — Paper trade for at least 2 weeks
```bash
python main.py --mode paper
```
Watch the `logs/` folder for:
- `trading_engine.log` — full activity log
- `logs/trade_journal.csv` — all trades
- `logs/equity_curve.csv` — equity over time

### Step 2 — Backtest multiple symbols
```bash
python main.py --mode backtest --symbol BTCUSDT
python main.py --mode backtest --symbol ETHUSDT
python main.py --mode backtest --symbol SOLUSDT
```

### Step 3 — Validate API keys work (read-only test)
```python
from bitget_rest import BitgetRestClient
r = BitgetRestClient()
print(r.get_account())   # Should return your balance
```

### Step 4 — Start live with tiny size
Set `risk_per_trade_pct = 0.002` (0.2%) and `default_leverage = 1` initially.

---

## Running 24/7 on a Laptop

### Option A — tmux (recommended)
```bash
sudo apt install tmux
tmux new -s trading
source venv/bin/activate
python main.py --mode paper
# Detach: Ctrl+B then D
# Reattach: tmux attach -t trading
```

### Option B — systemd service
Create `/etc/systemd/system/trading.service`:
```ini
[Unit]
Description=Bitget Trading Engine
After=network.target

[Service]
User=youruser
WorkingDirectory=/path/to/bitget_engine
ExecStart=/path/to/bitget_engine/venv/bin/python main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable trading
sudo systemctl start trading
sudo journalctl -u trading -f   # Watch logs
```

### Option C — nohup
```bash
nohup python main.py --mode paper > logs/stdout.log 2>&1 &
echo $! > trading.pid   # Save PID
```

---

## How Signals Work

### Symbol Ranking (every scan cycle)
Each symbol is scored 0–1 on:
- **EMA trend strength** (35%) — fast/slow EMA divergence
- **ATR expansion** (25%) — volatility expanding = momentum
- **10-period momentum** (25%) — recent price change
- **Volume spike** (15%) — recent bar vs average

Top N symbols proceed to strategy evaluation.

### Regime Detection → Strategy Selection

| Regime | Trigger | Strategy |
|--------|---------|----------|
| TRENDING | ADX > 25 | EMA 9/21 crossover + ATR stop |
| RANGING | ADX ≤ 25 | Bollinger Band mean reversion |
| HIGH_VOLATILITY | ATR > 2× median | TRENDING rules + 50% size reduction |

### Risk per Trade
```
stop_distance = |entry - stop_loss| / entry    (as %)
risk_$ = equity × 1%
position_size = risk_$ / stop_distance
contracts = (position_size × leverage) / price
```

---

## Correlation Control

Every 10 minutes, the engine computes a rolling 50-bar correlation matrix.
If two symbols show `|correlation| > 0.80`, the second one will be blocked from opening.
This prevents doubling up on essentially the same trade.

---

## Performance Analytics

At shutdown (or manually), the engine exports:
- `logs/trade_journal.csv` — every trade with entry/exit/PnL/regime
- `logs/equity_curve.csv` — equity over time

Metrics computed:
- Sharpe Ratio (annualized)
- Win Rate
- Profit Factor
- Max Drawdown %
- Expectancy per trade

---

## Discord Alerts

Set `DISCORD_WEBHOOK_URL` in `.env`. You'll receive:
- 📈 / 📉 Trade opened
- ✅ / ❌ Trade closed with PnL
- 🔀 Regime change
- ⛔ Daily loss cap hit
- ⚠️ Engine errors
- 📊 Daily summary at shutdown

---

## Logs

```
logs/trading_engine.log     # Rotating log (10MB × 5 files)
logs/trade_journal.csv      # Trade records
logs/equity_curve.csv       # Equity snapshots
logs/backtest_trades.csv    # Backtest trade log
```

---

## Troubleshooting

### SSL Certificate / Network Blocking Issues

If you see errors like `CERTIFICATE_VERIFY_FAILED` or `Prohibited Access`:

**Diagnosis:**
```bash
# Check if your network is blocking Bitget's API
curl -I https://api.bitget.com/api/v2/public/time
```

If it returns HTML or "Prohibited Access", your network is blocking cryptocurrency APIs.

**Solutions:**

1. **Use a VPN (Recommended)**
   ```bash
   # If you have OpenVPN
   sudo openvpn --config your-vpn-config.ovpn
   
   # Or install and use ProtonVPN (free tier available)
   sudo apt install protonvpn
   protonvpn connect
   ```

2. **Whitelist api.bitget.com in your router/firewall settings**
   - Check your router admin panel (usually 192.168.1.1)
   - Look for "Parental Controls" or "Content Filtering"
   - Add api.bitget.com to the whitelist

3. **For development/testing ONLY** (not recommended for live trading):
   ```bash
   # Add to your .env file
   DISABLE_SSL_VERIFICATION=true
   ```
   ⚠️ This bypasses SSL security checks. Use only for paper trading/backtesting.

4. **Use mobile hotspot** as a temporary workaround

---

## Important Warnings

- **Never run live mode without paper trading first**
- **Default leverage is 3×** — reduce to 1× while learning
- **Default risk is 1%** — this is conservative but still real money
- **Max daily loss is 4%** — the engine will pause automatically
- The engine is designed to fail safely — it will shut down cleanly on errors
- API keys need **Trade** permission on Bitget (not Withdraw)

---

## Bitget API Key Setup

1. Log into Bitget → Account → API Management
2. Create new API key
3. Permissions: **Read + Trade** only (no withdraw)
4. IP whitelist: add your laptop's IP for extra security
5. Copy Key, Secret, Passphrase into `.env`
