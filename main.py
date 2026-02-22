"""
main.py - Trading Engine Orchestrator
Entry point for all modes: paper, live, backtest.

Usage:
    python main.py                  # Use mode from .env / config
    python main.py --mode paper     # Force paper trading
    python main.py --mode live      # Force live trading
    python main.py --mode backtest  # Run backtest on top symbols
    python main.py --mode backtest --symbol BTCUSDT
"""
import os
import sys
import signal
import argparse
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Optional
import pandas as pd

from config import get_config, reload_config
from logger import setup_logger, get_logger
from bitget_rest import BitgetRestClient
from bitget_ws import BitgetWebSocket
from universe import UniverseManager
from data_feed import DataFeed
from scanner import Scanner
from risk_manager import RiskManager
from portfolio_manager import PortfolioManager, Position
from execution_engine import ExecutionEngine
from paper_engine import PaperEngine
from metrics import MetricsTracker, TradeRecord
from discord_notifier import DiscordNotifier
from strategy import generate_signal, SignalDirection, compute_trailing_stop
from backtester import Backtester
from strategy_stats import StrategyStatsManager

log = get_logger("main")

# -----------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------
_shutdown_event = threading.Event()


def _handle_sigint(signum, frame):
    log.warning("Shutdown signal received — cleaning up …")
    _shutdown_event.set()


signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


# -----------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------

class TradingEngine:
    def __init__(self, mode_override: Optional[str] = None) -> None:
        cfg = get_config()
        if mode_override:
            cfg.trading.mode = mode_override

        self._cfg = cfg
        self._mode = cfg.trading.mode
        log.info("=== Bitget Futures Engine | mode=%s | pid=%d ===", self._mode.upper(), os.getpid())

        # Shared components
        self._rest = BitgetRestClient()
        self._universe = UniverseManager(self._rest)
        self._feed = DataFeed(self._rest)
        self._scanner = Scanner(self._feed)
        self._risk = RiskManager()
        self._stats = StrategyStatsManager()
        self._portfolio = PortfolioManager(self._feed, self._risk, self._stats)
        self._metrics = MetricsTracker()
        self._notifier = DiscordNotifier()

        # Mode-specific setup
        initial_equity = 10_000.0
        if self._mode == "live":
            try:
                acct = self._rest.get_account()
                initial_equity = float(acct.get("available", initial_equity))
                log.info("Live account equity: $%.2f", initial_equity)
            except Exception as e:
                log.error("Could not fetch live equity: %s", e)

        self._paper = PaperEngine(initial_equity=initial_equity)
        self._portfolio.set_equity(initial_equity)

        self._executor = ExecutionEngine(
            portfolio=self._portfolio,
            risk=self._risk,
            paper_engine=self._paper if self._mode == "paper" else None,
            rest_client=self._rest if self._mode == "live" else None,
            stats_manager=self._stats,
        )

        self._ws: Optional[BitgetWebSocket] = None
        if self._mode == "live":
            self._ws = BitgetWebSocket(on_price_update=self._on_ws_price)

        self._last_equity: float = initial_equity
        self._last_scan_time: float = 0.0
        self._last_correlation_time: float = 0.0
        self._correlation_interval: int = 600  # seconds

        # Regime-block cache: symbol → unix timestamp of last REGIME_BLOCKED
        # Shorter TTL for 1m timeframe so regime re-evaluates more frequently
        self._regime_block_cache: Dict[str, float] = {}
        self._regime_block_ttl: int = 120  # 2 minutes on 1m timeframe

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        if self._mode == "live" and self._ws:
            self._ws.start()

        # Initial universe discovery
        self._universe.refresh()

        # Live mode: configure exchange account after universe is known
        # Sets hedge_mode + cross margin + leverage on all active symbols
        # (mirrors passivbot's update_exchange_config — safe to call on every start)
        if self._mode == "live":
            try:
                symbols = self._universe.get_universe()
                self._rest.initialize_account(
                    symbols=symbols,
                    leverage=self._cfg.trading.default_leverage,
                )
                log.info(
                    "Exchange initialized: hedge_mode + cross margin + %d× leverage on %d symbols",
                    self._cfg.trading.default_leverage, len(symbols),
                )
            except Exception as e:
                log.error("Account initialization failed (will retry next cycle): %s", e)

        log.info("Engine running. Press Ctrl+C to stop.")

        while not _shutdown_event.is_set():
            try:
                self._cycle()
            except Exception as e:
                log.error("Main loop error: %s", e, exc_info=True)
                self._notifier.error(str(e))

            _shutdown_event.wait(timeout=self._cfg.trading.scan_interval_seconds)

        self._shutdown()

    def _cycle(self) -> None:

        """One full scan/trade cycle."""
        # Guard: do not execute if shutdown is already in progress
        if _shutdown_event.is_set():
            return
        now = time.time()

        # Refresh universe periodically
        if self._universe.needs_refresh():
            self._universe.refresh()

        # Update equity
        self._refresh_equity()

        equity = self._portfolio.get_equity()
        unrealized = sum(
            p.unrealized_pnl for p in self._portfolio.get_all_positions().values()
        )
        log.info(
            "Cycle start | equity=$%.2f | mtm=$%.2f | unrealized=%.2f | open_pos=%d | daily_pnl=%.2f",
            equity,
            equity + unrealized,
            unrealized,
            self._portfolio.count_open(),
            self._risk.get_daily_pnl() + unrealized,
        )

        # Expire old regime-block cache entries
        now_ts = time.time()
        self._regime_block_cache = {
            sym: ts for sym, ts in self._regime_block_cache.items()
            if now_ts - ts < self._regime_block_ttl
        }

        # Correlation matrix
        if now - self._last_correlation_time > self._correlation_interval:
            symbols = self._universe.get_universe()[:50]
            self._portfolio.compute_correlation_blocks(symbols)
            self._last_correlation_time = now

        # Trailing stop updates
        self._update_trailing_stops()

        # Exit checks for open positions
        self._check_exits()

        # Partial profit / breakeven / time-based stagnant exits
        self._manage_open_positions()

        # Scan for new opportunities
        universe = self._universe.get_universe()
        if not universe:
            log.warning("Empty universe — skipping scan")
            return

        scores = self._scanner.scan(universe)
        top_n = scores[:self._cfg.trading.top_n_symbols]

        # Log scanner summary for top symbols
        if scores:
            top_strs = [
                f"{s.symbol}(sc={s.score:.2f},rs={s.rs_btc_score:.2f})"
                for s in top_n[:5]
            ]
            log.debug("Scanner top-5: %s", " | ".join(top_strs))

        # Generate and execute signals
        current_prices = self._get_current_prices()
        trace_enabled = self._cfg.trading.log_decision_trace

        signals_evaluated = 0
        signals_generated = 0
        trades_placed = 0

        # Reset per-cycle trade counter so burst cap applies fresh each scan
        self._executor.reset_cycle_counter()

        # BTC regime gate: suppress alt LONG trades when BTC EMA trend is bearish.
        # On 1m, if BTC fast EMA < slow EMA, the macro environment is adverse for longs.
        btc_bullish = True  # default: permissive
        btc_df = self._feed.get_ohlcv("BTCUSDT")
        if btc_df is not None and len(btc_df) >= 50:
            ema_fast = float(btc_df["close"].ewm(span=9, adjust=False).mean().iloc[-1])
            ema_slow = float(btc_df["close"].ewm(span=26, adjust=False).mean().iloc[-1])
            btc_bullish = ema_fast > ema_slow
            if not btc_bullish:
                log.debug("BTC gate: EMA bearish (ema9=%.1f < ema26=%.1f) — alt LONGs suppressed", ema_fast, ema_slow)

        for score in top_n:
            sym = score.symbol
            # Skip already open
            if self._portfolio.get_position(sym):
                if trace_enabled:
                    log.info("TRACE %s: skip (position already open)", sym)
                else:
                    log.debug("%s: position already open, skipping", sym)
                continue

            # Skip symbols recently regime-blocked (saves OHLCV fetch + indicator compute)
            if sym in self._regime_block_cache:
                if trace_enabled:
                    remaining = int(self._regime_block_ttl - (time.time() - self._regime_block_cache[sym]))
                    log.debug("TRACE %s: regime-blocked (cached, %ds remaining)", sym, remaining)
                continue

            df = self._feed.get_ohlcv(sym)
            if df is None or df.empty:
                if trace_enabled:
                    log.info("TRACE %s: skip (no OHLCV data)", sym)
                else:
                    log.debug("%s: no OHLCV data available", sym)
                continue

            signals_evaluated += 1
            signal = generate_signal(sym, df)
            if signal.direction == SignalDirection.NONE:
                # Cache regime-blocked symbols so we skip them next N cycles
                if signal.reason == "REGIME_BLOCKED":
                    self._regime_block_cache[sym] = time.time()
                if trace_enabled:
                    log.info(
                        "TRACE %s: no signal | strat=%s | conf=%.2f | regime=%s | reason=%s",
                        sym,
                        signal.strategy,
                        signal.confidence,
                        signal.regime.value,
                        signal.reason,
                    )
                else:
                    log.debug("%s: no signal (regime=%s)", sym, signal.regime.value)
                continue

            # Signal detected!
            signals_generated += 1
            log.info(
                "🎯 %s signal: %s | entry=%.4f stop=%.4f | regime=%s | strat=%s | conf=%.2f",
                signal.direction.value,
                sym,
                signal.entry_price,
                signal.stop_loss,
                signal.regime.value,
                signal.strategy,
                signal.confidence,
            )
            # BTC gate: block LONG entries on alts when BTC is in a downtrend
            # Exemption: symbols with portfolio correlation to BTC < btc_gate_min_correlation
            # (low-corr alts are not reliably suppressed by BTC macro direction)
            if (signal.direction == SignalDirection.LONG
                    and sym != "BTCUSDT"
                    and not btc_bullish):
                btc_corr = self._portfolio.get_btc_correlation(sym)
                if btc_corr >= self._cfg.trading.btc_gate_min_correlation:
                    if trace_enabled:
                        log.info("TRACE %s: LONG blocked — BTC EMA bearish (gate, corr=%.2f)", sym, btc_corr)
                    continue
                else:
                    if trace_enabled:
                        log.info("TRACE %s: BTC gate bypassed — low BTC corr=%.2f < %.2f", sym, btc_corr, self._cfg.trading.btc_gate_min_correlation)
            placed = self._executor.execute_signal(signal, equity, current_prices)
            if placed:
                trades_placed += 1
                pos = self._portfolio.get_position(sym)
                if pos:
                    # Store entry-time RS score for deterioration exit tracking
                    pos.entry_rs_score = score.rs_btc_score
                    unrealized = pos.unrealized_pnl
                    position_cost = pos.entry_price * pos.quantity + 1e-10
                    pnl_pct = (unrealized / position_cost) * 100
                    log.info(
                        "\U0001f195 Trade opened: %s %s @ %.4f | notional=$%.2f | stop=%.4f | PnL $%.2f (%.2f%%)",
                        signal.direction.value,
                        sym,
                        signal.entry_price,
                        pos.quantity * pos.entry_price,
                        signal.stop_loss,
                        unrealized,
                        pnl_pct,
                    )
                    self._notifier.trade_opened(
                        symbol=sym,
                        direction=signal.direction.value,
                        entry=signal.entry_price,
                        stop=signal.stop_loss,
                        size_usd=pos.quantity * pos.entry_price,
                        regime=signal.regime.value,
                        pnl=unrealized,
                        pnl_pct=pnl_pct,
                        strategy=signal.strategy,
                        confidence=signal.confidence,
                    )
            elif trace_enabled:
                log.info("TRACE %s: signal not executed (see risk/execution logs)", sym)
        
        # Log scan summary
        if signals_evaluated > 0:
            log.info("Scan complete: evaluated %d symbols | signals: %d | trades placed: %d", 
                     signals_evaluated, signals_generated, trades_placed)

        # Record equity snapshot
        self._metrics.record_equity(equity)
        self._last_scan_time = now

    def _manage_open_positions(self) -> None:
        """Manage active positions each cycle:
        1. Partial profit take at 1R — close partial_exit_fraction of the position.
        2. Move stop to breakeven after 1R profit is locked.
        3. Time-based stagnant exit — close if < stagnant_exit_r_threshold R after N bars.
        """
        prices = self._get_current_prices()
        cfg = self._cfg.trading

        for sym, pos in list(self._portfolio.get_all_positions().items()):
            price = prices.get(sym)
            if not price:
                continue

            risk_ref = pos.risk_dollar if pos.risk_dollar > 0 else (pos.stop_distance * pos.quantity)
            if risk_ref <= 0:
                continue

            # Unrealized PnL in R multiples
            pos.update_price(price)
            r_current = pos.unrealized_pnl / (risk_ref + 1e-10)

            # 1. Partial profit: take 50% off the table at 1R
            if not pos.partial_taken and r_current >= cfg.partial_exit_r:
                taken = self._executor.partial_close_position(
                    sym, price, fraction=cfg.partial_exit_fraction, reason="PARTIAL_1R"
                )
                if taken:
                    pnl_partial = pos.unrealized_pnl * cfg.partial_exit_fraction
                    rec = TradeRecord(
                        trade_id=f"{sym}_P_{int(time.time())}",
                        symbol=sym, direction=pos.direction.value,
                        entry_price=pos.entry_price, exit_price=price,
                        quantity=pos.quantity * cfg.partial_exit_fraction,
                        pnl=pnl_partial, pnl_pct=pnl_partial / (pos.entry_price * pos.quantity + 1e-10),
                        fees=price * pos.quantity * cfg.partial_exit_fraction * cfg.taker_fee_pct,
                        entry_time=pos.entry_time.isoformat(),
                        exit_time=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=(datetime.now(timezone.utc) - pos.entry_time).total_seconds(),
                        regime=pos.regime, exit_reason="PARTIAL_1R",
                        stop_loss=pos.stop_loss, take_profit=pos.take_profit,
                        strategy=pos.strategy, confidence=pos.signal_confidence,
                        kelly_fraction=pos.kelly_fraction,
                        is_partial=True,
                    )
                    self._metrics.record_trade(rec)
                    self._notifier.trade_closed(
                        sym, pos.direction.value, pos.entry_price, price,
                        pnl_partial, pnl_partial / (pos.entry_price * pos.quantity + 1e-10), "PARTIAL_1R"
                    )

            # 2. Move stop to breakeven once 1R is achieved
            if not pos.breakeven_set and r_current >= cfg.breakeven_trigger_r:
                self._executor.set_breakeven_stop(sym, price)

            # 3. Time-based stagnant exit
            self._executor.check_time_based_exit(sym, price)

            # 4. RS deterioration exit: if symbol RS has dropped > threshold from entry, close
            # Catches momentum collapse (e.g. PIPPIN rs:0.70 -> rs:0.13 in same session)
            if pos.entry_rs_score > 0:
                current_score = self._scanner.get_score(sym)
                if current_score is not None:
                    rs_drop = pos.entry_rs_score - current_score.rs_btc_score
                    if rs_drop > self._cfg.trading.rs_exit_drop_threshold:
                        log.warning(
                            "⚠️  RS deterioration exit [%s]: entry_rs=%.2f current_rs=%.2f drop=%.2f (threshold=%.2f)",
                            sym, pos.entry_rs_score, current_score.rs_btc_score,
                            rs_drop, self._cfg.trading.rs_exit_drop_threshold,
                        )
                        self._executor.close_position(sym, price, "RS_DETERIORATION")

    def _check_exits(self) -> None:
        """Check all open positions for stop/target hits."""
        prices = self._get_current_prices()
        self._portfolio.update_prices(prices)

        for sym, pos in list(self._portfolio.get_all_positions().items()):
            price = prices.get(sym)
            if not price:
                continue

            should_close, reason = self._portfolio.check_stop_and_target(sym, price)
            if should_close:
                pnl_before = pos.unrealized_pnl
                pnl_pct = (pnl_before / (pos.entry_price * pos.quantity)) * 100 if pos.quantity > 0 else 0
                
                log.info("🔔 Closing %s %s @ %.4f | PnL: $%.2f (%.2f%%) | reason: %s", 
                         pos.direction.value, sym, price, pnl_before, pnl_pct, reason)
                
                self._executor.close_position(sym, price, reason)
                self._notifier.trade_closed(
                    sym, pos.direction.value,
                    pos.entry_price, price, pnl_before, pnl_pct, reason
                )
                # Record trade
                exit_ts = datetime.now(timezone.utc)
                rec = TradeRecord(
                    trade_id=f"{sym}_{int(time.time())}",
                    symbol=sym,
                    direction=pos.direction.value,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    quantity=pos.quantity,
                    pnl=pnl_before,
                    pnl_pct=pnl_before / (pos.entry_price * pos.quantity + 1e-10),
                    fees=pos.entry_price * pos.quantity * self._cfg.trading.taker_fee_pct,
                    entry_time=pos.entry_time.isoformat(),
                    exit_time=exit_ts.isoformat(),
                    duration_seconds=(exit_ts - pos.entry_time).total_seconds(),
                    regime=pos.regime,
                    exit_reason=reason,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    strategy=pos.strategy,
                    confidence=pos.signal_confidence,
                    kelly_fraction=pos.kelly_fraction,
                )
                self._metrics.record_trade(rec)

    def _update_trailing_stops(self) -> None:
        """Update trailing stops for all open positions."""
        prices = self._get_current_prices()
        for sym, pos in list(self._portfolio.get_all_positions().items()):
            price = prices.get(sym)
            if not price or pos.atr == 0:
                continue
            new_stop = compute_trailing_stop(
                pos.direction, price,
                pos.highest_price if pos.direction == SignalDirection.LONG else pos.lowest_price,
                pos.atr
            )
            # Only move stop in the profitable direction
            if pos.direction == SignalDirection.LONG and new_stop > pos.stop_loss:
                self._portfolio.update_stop_loss(sym, new_stop)
            elif pos.direction == SignalDirection.SHORT and new_stop < pos.stop_loss:
                self._portfolio.update_stop_loss(sym, new_stop)

    def _refresh_equity(self) -> None:
        if self._mode == "paper":
            # Cash from closed trades + unrealized PnL from open positions = true MTM equity
            cash = self._paper.get_equity()
            unrealized = sum(
                p.unrealized_pnl for p in self._portfolio.get_all_positions().values()
            )
            equity = cash + unrealized
        else:
            try:
                acct = self._rest.get_account()
                equity = float(acct.get("available", self._last_equity))
            except Exception:
                equity = self._last_equity

        self._last_equity = equity
        self._portfolio.set_equity(equity)

    def _get_current_prices(self) -> Dict[str, float]:
        if self._ws:
            return self._ws.get_all_prices()
        # REST fallback: get last close from cached candles
        prices = {}
        for sym in self._universe.get_universe()[:60]:
            df = self._feed.get_ohlcv(sym)
            if df is not None and not df.empty:
                prices[sym] = float(df["close"].iloc[-1])
        return prices

    def _on_ws_price(self, symbol: str, price: float) -> None:
        """Called by WebSocket thread on every price update."""
        pass  # Price cache handled inside ws; exits checked in cycle

    def _shutdown(self) -> None:
        log.info("Shutting down …")
        if self._ws:
            self._ws.stop()

        # Close all positions and record each trade to the journal
        prices = self._get_current_prices()
        exit_ts = datetime.now(timezone.utc)
        for sym in list(self._portfolio.get_open_symbols()):
            price = prices.get(sym, 0)
            if not price:
                continue
            pos = self._portfolio.get_position(sym)  # capture before close
            self._executor.close_position(sym, price, "SHUTDOWN")
            if pos:
                if pos.direction == SignalDirection.LONG:
                    pnl = (price - pos.entry_price) * pos.quantity
                else:
                    pnl = (pos.entry_price - price) * pos.quantity
                rec = TradeRecord(
                    trade_id=f"{sym}_{int(time.time())}",
                    symbol=sym,
                    direction=pos.direction.value,
                    entry_price=pos.entry_price,
                    exit_price=price,
                    quantity=pos.quantity,
                    pnl=pnl,
                    pnl_pct=pnl / (pos.entry_price * pos.quantity + 1e-10),
                    fees=pos.entry_price * pos.quantity * self._cfg.trading.taker_fee_pct,
                    entry_time=pos.entry_time.isoformat(),
                    exit_time=exit_ts.isoformat(),
                    duration_seconds=(exit_ts - pos.entry_time).total_seconds(),
                    regime=pos.regime,
                    exit_reason="SHUTDOWN",
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    strategy=pos.strategy,
                    confidence=pos.signal_confidence,
                    kelly_fraction=pos.kelly_fraction,
                )
                self._metrics.record_trade(rec)

        # Export analytics
        os.makedirs("logs", exist_ok=True)
        self._metrics.export_trade_journal()
        self._metrics.export_equity_curve()
        if self._mode == "paper":
            self._paper.export_equity_curve(self._cfg.trading.equity_curve_export_path)

        self._metrics.print_summary()
        self._metrics.print_regime_summary()
        stats = self._metrics.compute_stats()
        self._notifier.daily_summary(stats)
        log.info("Engine stopped cleanly.")


# -----------------------------------------------------------------------
# Backtest entry point
# -----------------------------------------------------------------------

def run_backtest(symbol: str = "BTCUSDT", candle_limit: int = 1000, granularity: str = "15m") -> None:
    cfg = get_config()

    # Try cached CSV first (Bitget API may be blocked)
    import os
    csv_path = f"logs/cached_{symbol.lower()}_{granularity}.csv"
    df = pd.DataFrame()

    if os.path.exists(csv_path):
        log.info("Loading cached data from %s", csv_path)
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        log.info("Loaded %d bars (%s → %s)", len(df), df.index[0], df.index[-1])
    else:
        log.info("Fetching %d × %s candles for %s backtest …", candle_limit, granularity, symbol)
        rest = BitgetRestClient()
        try:
            raw = rest.get_candles(symbol, granularity=granularity, limit=candle_limit)
            from data_feed import candles_to_df
            df = candles_to_df(raw)
        except Exception as e:
            log.error("Bitget API failed: %s. Trying yfinance fallback...", e)

        if df.empty:
            # yfinance fallback
            try:
                import yfinance as yf
                ticker = yf.Ticker(f"{symbol[:3]}-USD")
                df = ticker.history(period="60d", interval=granularity)
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.to_csv(csv_path)
                log.info("Fetched %d bars via yfinance, saved to %s", len(df), csv_path)
            except Exception as e2:
                log.error("yfinance also failed: %s", e2)

    if df.empty:
        log.error("No data available for %s", symbol)
        return

    bt = Backtester(initial_equity=10_000.0)
    bt.run(df, symbol=symbol)
    bt.print_results()
    bt.export_trades()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bitget Futures Trading Engine")
    parser.add_argument(
        "--mode",
        choices=["paper", "live", "backtest"],
        help="Override trading mode (default: from .env)"
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Symbol for backtest mode (default: BTCUSDT)"
    )
    parser.add_argument(
        "--granularity",
        default="15m",
        help="Candle granularity for backtest mode (default: 15m)"
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=10_000.0,
        help="Starting equity for paper/backtest mode (default: 10000)"
    )
    args = parser.parse_args()

    cfg = get_config()
    setup_logger("trading_engine", cfg.log_level, cfg.log_dir)

    mode = args.mode or cfg.trading.mode
    log.info("Starting engine | mode=%s", mode)

    if mode == "backtest":
        run_backtest(symbol=args.symbol, granularity=args.granularity)
    else:
        # PID lock file — prevent dual instances
        pid_file = os.path.join(cfg.log_dir, "trading.pid")
        os.makedirs(cfg.log_dir, exist_ok=True)
        if os.path.exists(pid_file):
            try:
                with open(pid_file) as _pf:
                    old_pid = int(_pf.read().strip())
                os.kill(old_pid, signal.SIGTERM)
                log.warning("Sent SIGTERM to previous instance (PID %d) — waiting 2s", old_pid)
                time.sleep(2)
            except (ProcessLookupError, ValueError, OSError):
                pass  # already dead
        with open(pid_file, "w") as _pf:
            _pf.write(str(os.getpid()))
        import atexit
        atexit.register(lambda: os.remove(pid_file) if os.path.exists(pid_file) else None)

        engine = TradingEngine(mode_override=mode)
        engine.run()


if __name__ == "__main__":
    main()
