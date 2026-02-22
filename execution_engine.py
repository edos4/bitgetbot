"""execution_engine.py - Unified Execution Router with Kelly sizing and risk analytics."""
import time
import threading
from collections import deque
from typing import Dict, Optional

from config import get_config
from strategy import Signal, SignalDirection
from portfolio_manager import PortfolioManager, Position
from risk_manager import RiskManager
from paper_engine import PaperEngine
from logger import get_logger
from strategy_stats import StrategyStatsManager

log = get_logger("execution")


class ExecutionEngine:
    def __init__(
        self,
        portfolio: PortfolioManager,
        risk: RiskManager,
        paper_engine: Optional[PaperEngine] = None,
        rest_client=None,
        stats_manager: Optional[StrategyStatsManager] = None,
    ) -> None:
        self._cfg = get_config().trading
        self._portfolio = portfolio
        self._risk = risk
        self._paper = paper_engine
        self._rest = rest_client
        self._mode = self._cfg.mode
        self._lock = threading.Lock()
        self._recent_trades = deque()
        self._symbol_last_trade: Dict[str, float] = {}  # symbol → epoch of last fill
        self._symbol_cooldown_dur: Dict[str, int] = {}  # symbol → adaptive cooldown seconds
        self._trades_this_cycle: int = 0               # counter reset at start of each scan cycle
        self._stats = stats_manager or StrategyStatsManager()
        log.info("ExecutionEngine initialized in %s mode", self._mode.upper())

    def execute_signal(
        self,
        signal: Signal,
        equity: float,
        current_prices: dict,
    ) -> bool:
        if signal.direction == SignalDirection.NONE:
            return False

        if self._portfolio.get_position(signal.symbol):
            log.debug("Skip %s — position already open", signal.symbol)
            return False

        # --- Hard signal validation ---
        # 1. Absolute price floor: sub-cent tokens (PEPE, SHIB-class) cause float precision
        #    errors: billions of tokens, ATR rounds to 0, stop collapses to entry.
        if signal.entry_price < self._cfg.min_entry_price:
            log.warning(
                "❌ Trade rejected [%s]: entry_price=%.10f below min_entry_price=%.6f "
                "(sub-cent token — float precision risk)",
                signal.symbol, signal.entry_price, self._cfg.min_entry_price,
            )
            return False
        if signal.stop_loss <= 0:
            log.warning("❌ Trade rejected [%s]: stop_loss=%.10f ≤ 0", signal.symbol, signal.stop_loss)
            return False

        # 2. ATR must be ≥ 0.005% of entry price — prevents division-by-zero and trivial sizing
        min_atr_abs = signal.entry_price * self._cfg.min_atr_fraction
        if signal.atr <= 0 or signal.atr < min_atr_abs:
            log.warning(
                "\u274c Trade rejected [%s]: ATR=%.10f below min (%.10f = %.3f%% of price)",
                signal.symbol, signal.atr, min_atr_abs, self._cfg.min_atr_fraction * 100,
            )
            return False

        # 3a. Stop distance must be ≥ price × min_stop_fraction — prevents DOGE-class
        #     cases where ATR is near-zero so ATR×N produces a trivially tight stop.
        #     At 0.05% of price: BTC@$68k → min $34 stop;  DOGE@$0.097 → min $0.000049.
        if self._cfg.min_stop_fraction > 0:
            min_stop_price = signal.entry_price * self._cfg.min_stop_fraction
            if signal.stop_distance < min_stop_price:
                log.warning(
                    "\u274c Trade rejected [%s]: stop_distance=%.8f below price floor "
                    "(%.8f = %.3f%% of entry)",
                    signal.symbol, signal.stop_distance,
                    min_stop_price, self._cfg.min_stop_fraction * 100,
                )
                return False

        # 3b. Stop distance must be ≥ 1.5× ATR — ensures stop is not within intrabar noise.
        min_stop_by_atr = signal.atr * self._cfg.min_stop_atr_multiple
        if signal.stop_distance < min_stop_by_atr:
            log.warning(
                "❌ Trade rejected [%s]: stop_distance=%.10f below ATR×%.1f=%.10f",
                signal.symbol, signal.stop_distance,
                self._cfg.min_stop_atr_multiple, min_stop_by_atr,
            )
            return False

        # Per-cycle burst guard: limit new entries within a single scan cycle
        max_per_cycle = self._cfg.max_new_trades_per_cycle
        if self._trades_this_cycle >= max_per_cycle:
            log.info(
                "❌ Trade blocked [%s]: per-cycle cap hit (%d/%d trades this cycle)",
                signal.symbol, self._trades_this_cycle, max_per_cycle,
            )
            return False
        # Minimum signal confidence gate: weak conf = noise on 1m
        if signal.confidence < self._cfg.min_signal_confidence:
            log.info(
                "\u274c Trade blocked [%s]: conf=%.2f below min=%.2f",
                signal.symbol, signal.confidence, self._cfg.min_signal_confidence,
            )
            return False
        if not self._can_trade_now():
            log.info("❌ Trade blocked [%s]: global trade frequency cap hit", signal.symbol)
            return False

        # --- Per-symbol adaptive cooldown ---
        # STOP_LOSS  → 1 candle (60 s): allow re-entry once market has reset
        # STAGNANT   → 2 candles (120 s): position dried up, wait longer
        # TP/other   → 0 s: re-entry immediately allowed
        cooldown = self._symbol_cooldown_dur.get(signal.symbol, 0)
        now = time.time()
        last = self._symbol_last_trade.get(signal.symbol, 0.0)
        if cooldown > 0 and now - last < cooldown:
            remaining = int(cooldown - (now - last))
            log.info("❌ Trade blocked [%s]: symbol on cooldown (%ds remaining)", signal.symbol, remaining)
            return False

        reward_risk = max(signal.reward_risk_ratio, 0.01)
        base_fraction = self._stats.get_kelly_fraction(signal.strategy, reward_risk)
        if base_fraction <= 0:
            log.info("❌ Trade blocked [%s]: Kelly fraction ≤ 0", signal.symbol)
            return False

        risk_fraction = min(base_fraction, self._cfg.max_risk_per_trade_pct)

        # Regime-behavioral risk scalar — regime is now an active sizing variable
        # TRENDING:        ×1.0  (full size — trend signals have best edge)
        # RANGING:         ×0.8  (mean-reversion on 1m has higher false-positive rate)
        # HIGH_VOLATILITY: ×0.5  (confirmed via confidence×0.5 already, but enforce here too)
        _REGIME_SCALARS = {"TRENDING": 1.0, "RANGING": 0.8, "HIGH_VOLATILITY": 0.5}
        regime_scalar = _REGIME_SCALARS.get(signal.regime.value, 1.0)
        if regime_scalar < 1.0:
            log.debug("Regime risk scalar [%s]: %s ×%.1f", signal.symbol, signal.regime.value, regime_scalar)
        risk_fraction *= regime_scalar

        # --- Tiered volume-ratio position sizing ---
        # signal.volume_ratio = current_vol / rolling_avg_vol (set by strategy)
        # < volume_ratio_min  → strategy already returned no-signal, but guard here too
        # volume_ratio_min to volume_ratio_half  → 50% size
        # volume_ratio_half to volume_ratio_boost → 100% size
        # > volume_ratio_boost → 125% size (pre-cap)
        vol_ratio = getattr(signal, "volume_ratio", 1.0)
        if vol_ratio < self._cfg.volume_ratio_min:
            log.info("❌ Trade blocked [%s]: volume_ratio=%.2f below min %.2f",
                     signal.symbol, vol_ratio, self._cfg.volume_ratio_min)
            return False
        elif vol_ratio < self._cfg.volume_ratio_half:
            risk_fraction *= 0.50
        elif vol_ratio > self._cfg.volume_ratio_boost:
            risk_fraction = min(risk_fraction * 1.25, self._cfg.max_risk_per_trade_pct)
        # else 1.0 → no adjustment

        # Scale position size by signal confidence (0.0–1.0); floor at 25%
        # Weak signal (conf=0.5, near-threshold) → half the size
        # Strong signal (conf=1.0, deep z / clear breakout) → full size
        if signal.confidence < 1.0:
            risk_fraction *= max(signal.confidence, 0.25)  # floor at 25% to keep min size meaningful

        corr_value = self._portfolio.get_max_correlation(signal.symbol)
        if corr_value > self._cfg.correlation_threshold:
            reduction = max(0.0, 1 - (corr_value - self._cfg.correlation_threshold) / (1 - self._cfg.correlation_threshold + 1e-10))
            risk_fraction *= reduction

        # Directional clustering penalty: same-direction + high correlation → halve risk
        if corr_value > self._cfg.directional_corr_penalty_threshold:
            open_positions = self._portfolio.get_all_positions()
            corr_row = self._portfolio._correlations.get(signal.symbol, {})
            same_dir_correlated = any(
                abs(corr_row.get(sym, 0.0)) > self._cfg.directional_corr_penalty_threshold
                and pos.direction == signal.direction
                for sym, pos in open_positions.items()
            )
            if same_dir_correlated:
                risk_fraction *= 0.5
                log.info(
                    "⚠ Same-direction cluster [%s] corr=%.2f %s: risk halved",
                    signal.symbol, corr_value, signal.direction.value,
                )

        if signal.atr_ratio > self._cfg.atr_volatility_multiplier:
            risk_fraction *= 0.5

        # ATR compression: low atr_ratio = compressed market = high false-signal probability
        # atr_ratio < 0.7 means current ATR is well below its rolling median → regime noise
        if 0 < signal.atr_ratio < self._cfg.atr_compression_ratio:
            risk_fraction *= 0.5
            log.debug(
                "ATR compression [%s]: atr_ratio=%.2f < %.2f → risk halved",
                signal.symbol, signal.atr_ratio, self._cfg.atr_compression_ratio,
            )

        # Same-direction decay ladder: each additional open same-dir position
        # reduces the new entry's risk — convexity in strong trends, not a hard cap.
        # 0 existing → ×1.00 | 1 → ×0.60 | 2 → ×0.35 | 3 → ×0.20
        # (Upstream risk_manager blocks at max_same_direction_positions=4)
        _SAME_DIR_DECAY = [1.00, 0.60, 0.35, 0.20]
        same_dir_count = self._risk.get_direction_count(signal.direction.value)
        if 0 < same_dir_count < len(_SAME_DIR_DECAY):
            decay_factor = _SAME_DIR_DECAY[same_dir_count]
            risk_fraction *= decay_factor
            log.info(
                "⚠ Same-direction decay [%s]: %d %s open → risk ×%.2f",
                signal.symbol, same_dir_count, signal.direction.value, decay_factor,
            )
        # Only enforce once strategy has min_ev_sample closed trades.
        # EV = RR × win_rate − (1 − win_rate)
        stats_snapshot = self._stats.get_snapshot(signal.strategy)
        if stats_snapshot.sample_size >= self._cfg.min_ev_sample:
            ev = reward_risk * stats_snapshot.win_rate - (1.0 - stats_snapshot.win_rate)
            if ev <= self._cfg.min_ev_threshold:
                log.info(
                    "❌ Trade blocked [%s]: EV=%.3f ≤ threshold=%.2f "
                    "(strat=%s win_rate=%.0f%% n=%d rr=%.2f)",
                    signal.symbol, ev, self._cfg.min_ev_threshold,
                    signal.strategy, stats_snapshot.win_rate * 100,
                    stats_snapshot.sample_size, reward_risk,
                )
                return False

        existing_syms = self._portfolio.get_open_symbols()

        decision = self._risk.evaluate(
            symbol=signal.symbol,
            requested_risk_fraction=risk_fraction,
            current_equity=equity,
            existing_positions=existing_syms,
            direction=signal.direction.value,
        )

        if not decision.approved:
            log.info("❌ Trade blocked [%s]: %s", signal.symbol, decision.reason)
            return False

        effective_fraction = decision.risk_fraction
        risk_dollar = decision.risk_dollar
        qty = risk_dollar / (signal.stop_distance + 1e-10)
        if qty <= 0:
            log.info("❌ Trade blocked [%s]: zero quantity after sizing", signal.symbol)
            return False

        notional_usd = qty * signal.entry_price

        # Bitget minimum order value: 5.1 USDT (exchange silently rejects below this)
        if notional_usd < self._cfg.min_order_notional_usdt:
            log.info(
                "❌ Trade blocked [%s]: notional=$%.2f below Bitget min=$%.1f",
                signal.symbol, notional_usd, self._cfg.min_order_notional_usdt,
            )
            return False

        # Hard per-trade notional cap: ≤ equity × max_notional_per_trade_x
        max_per_trade_notional = equity * self._cfg.max_notional_per_trade_x
        if notional_usd > max_per_trade_notional:
            qty = max_per_trade_notional / (signal.entry_price + 1e-10)
            notional_usd = qty * signal.entry_price
            risk_dollar = qty * signal.stop_distance
            effective_fraction = risk_dollar / (equity + 1e-10)
            log.info(
                "Notional cap applied [%s]: capped at $%.0f (%.1f× equity)",
                signal.symbol, max_per_trade_notional, self._cfg.max_notional_per_trade_x,
            )

        # Hard total notional cap: total portfolio notional ≤ equity × max_total_notional_x
        total_notional = self._portfolio.get_total_notional()
        max_total_notional = equity * self._cfg.max_total_notional_x
        if total_notional + notional_usd > max_total_notional:
            remaining_notional = max_total_notional - total_notional
            if remaining_notional <= 0:
                log.info(
                    "❌ Trade blocked [%s]: total notional cap hit ($%.0f / $%.0f)",
                    signal.symbol, total_notional, max_total_notional,
                )
                return False
            qty = remaining_notional / (signal.entry_price + 1e-10)
            notional_usd = remaining_notional
            risk_dollar = qty * signal.stop_distance
            effective_fraction = risk_dollar / (equity + 1e-10)
            log.info(
                "Total notional cap sizing down [%s]: $%.0f → $%.0f",
                signal.symbol, total_notional + notional_usd, max_total_notional,
            )

        if qty <= 0:
            log.info("❌ Trade blocked [%s]: zero quantity after notional cap", signal.symbol)
            return False

        if self._portfolio.breaches_variance_cap(notional_usd, signal.entry_price, signal.atr):
            log.info("❌ Trade blocked [%s]: portfolio variance cap exceeded", signal.symbol)
            return False

        # stats_snapshot already computed above for EV filter; reuse here
        expected_value = reward_risk * stats_snapshot.win_rate - (1 - stats_snapshot.win_rate)
        portfolio_heat = self._risk.get_portfolio_heat(equity)

        log.info(
            "analytics | symbol=%s strat=%s rr=%.2f expected_value=%.2f atr=%.4f kelly=%.2f%% win_rate=%.2f%% heat=%.2f%% corr=%.2f",
            signal.symbol,
            signal.strategy,
            reward_risk,
            expected_value,
            signal.atr,
            effective_fraction * 100,
            stats_snapshot.win_rate * 100,
            portfolio_heat * 100,
            corr_value,
        )

        log.info(
            "✓ Risk approved [%s]: qty=%.4f size=$%.2f leverage=%dx | strategy=%s",
            signal.symbol,
            qty,
            notional_usd,
            decision.leverage,
            signal.strategy,
        )

        success = self._place_order(signal, qty, decision.leverage, risk_dollar, effective_fraction)
        if success:
            now_ts = time.time()
            self._recent_trades.append(now_ts)
            self._symbol_last_trade[signal.symbol] = now_ts
            self._trades_this_cycle += 1
        return success

    def partial_close_position(
        self,
        symbol: str,
        current_price: float,
        fraction: float = 0.5,
        reason: str = "PARTIAL_PROFIT",
    ) -> bool:
        """Close `fraction` of an open position; does not register a full close."""
        with self._lock:
            pos = self._portfolio.get_position(symbol)
        if not pos or pos.partial_taken:
            return False

        partial_qty = pos.quantity * fraction
        if partial_qty <= 0:
            return False

        if self._mode == "paper" and self._paper:
            if pos.direction == SignalDirection.LONG:
                fill = self._paper.market_sell(symbol, partial_qty, current_price)
                pnl  = (fill.fill_price - pos.entry_price) * partial_qty
            else:
                fill = self._paper.market_buy(symbol, partial_qty, current_price)
                pnl  = (pos.entry_price - fill.fill_price) * partial_qty
            pnl -= fill.fee_paid
            self._paper.record_trade_pnl(pnl, 0.0)  # fee already deducted

        # Reduce position quantity and mark partial taken
        with self._portfolio._lock:
            if symbol in self._portfolio._positions:
                self._portfolio._positions[symbol].quantity -= partial_qty
                self._portfolio._positions[symbol].partial_taken = True

        log.info(
            "∂ Partial close %s: %.4f units @ %.4f | reason=%s | remaining=%.4f",
            symbol, partial_qty, current_price, reason,
            pos.quantity - partial_qty,
        )
        return True

    def set_breakeven_stop(
        self,
        symbol: str,
        current_price: float,
    ) -> bool:
        """Move stop loss to entry price (breakeven) after 1R profit is achieved."""
        pos = self._portfolio.get_position(symbol)
        if not pos or pos.breakeven_set:
            return False

        be_stop = pos.entry_price
        if pos.direction == SignalDirection.LONG and current_price > pos.entry_price:
            if pos.stop_loss < be_stop:
                self._portfolio.update_stop_loss(symbol, be_stop)
                with self._portfolio._lock:
                    if symbol in self._portfolio._positions:
                        self._portfolio._positions[symbol].breakeven_set = True
                log.info("→ Breakeven stop set: %s stop=%.4f", symbol, be_stop)
                return True
        elif pos.direction == SignalDirection.SHORT and current_price < pos.entry_price:
            if pos.stop_loss > be_stop:
                self._portfolio.update_stop_loss(symbol, be_stop)
                with self._portfolio._lock:
                    if symbol in self._portfolio._positions:
                        self._portfolio._positions[symbol].breakeven_set = True
                log.info("→ Breakeven stop set: %s stop=%.4f", symbol, be_stop)
                return True
        return False

    def check_time_based_exit(
        self,
        symbol: str,
        current_price: float,
    ) -> bool:
        """Exit if trade has stagnated: open for > stagnant_exit_bars and
        unrealized PnL < stagnant_exit_r_threshold × risk_dollar."""
        pos = self._portfolio.get_position(symbol)
        if not pos:
            return False

        import time as _time
        from datetime import timezone
        elapsed_seconds = (_time.time() - pos.entry_time.replace(tzinfo=timezone.utc).timestamp()
                           if pos.entry_time.tzinfo is None
                           else (_time.time() - pos.entry_time.timestamp()))
        bar_seconds = self._cfg.candle_seconds  # 900 for 15m
        elapsed_bars = elapsed_seconds / bar_seconds

        if elapsed_bars < self._cfg.stagnant_exit_bars:
            return False

        pos.update_price(current_price)
        risk_ref = pos.risk_dollar if pos.risk_dollar > 0 else (pos.stop_distance * pos.quantity)
        r_val = pos.unrealized_pnl / (risk_ref + 1e-10)

        if r_val < self._cfg.stagnant_exit_r_threshold:
            log.info(
                "⏱ Time-based exit %s: %.0f bars elapsed, r=%.2f < %.2f threshold",
                symbol, elapsed_bars, r_val, self._cfg.stagnant_exit_r_threshold,
            )
            return self.close_position(symbol, current_price, "TIME_EXIT")
        return False

    def close_position(
        self,
        symbol: str,
        current_price: float,
        reason: str = "MANUAL",
    ) -> bool:
        """Close an open position at the current market price."""
        pos = self._portfolio.get_position(symbol)
        if not pos:
            return False

        log.info("Closing %s @ %.4f | reason=%s", symbol, current_price, reason)

        if self._mode == "paper" and self._paper:
            if pos.direction == SignalDirection.LONG:
                fill = self._paper.market_sell(symbol, pos.quantity, current_price)
                pnl = (fill.fill_price - pos.entry_price) * pos.quantity
            else:
                fill = self._paper.market_buy(symbol, pos.quantity, current_price)
                pnl = (pos.entry_price - fill.fill_price) * pos.quantity
            # fee_paid already subtracted in paper engine's record_trade_pnl
            self._paper.record_trade_pnl(pnl - fill.fee_paid, 0.0)
        elif self._mode == "live":
            self._live_close(pos, current_price)

        self._portfolio.close_position(symbol, current_price, reason)
        # Adaptive cooldown: duration depends on why position was closed
        if reason == "STOP_LOSS":
            self._symbol_last_trade[symbol] = time.time()
            self._symbol_cooldown_dur[symbol] = 60    # 1 candle
        elif reason == "STAGNANT":
            self._symbol_last_trade[symbol] = time.time()
            self._symbol_cooldown_dur[symbol] = 120   # 2 candles
        else:
            # TP or manual close: allow immediate re-entry
            self._symbol_cooldown_dur[symbol] = 0
        return True

    def _place_order(self, signal: Signal, quantity: float, leverage: int, risk_dollar: float, risk_fraction: float) -> bool:
        price = signal.entry_price
        if self._mode == "paper":
            return self._paper_open(signal, quantity, price, leverage, risk_dollar, risk_fraction)
        return self._live_open(signal, quantity, price, leverage, risk_dollar, risk_fraction)

    def _paper_open(
        self,
        signal: Signal,
        quantity: float,
        price: float,
        leverage: int,
        risk_dollar: float,
        risk_fraction: float,
    ) -> bool:
        qty = round(max(quantity, 0.0), 4)
        if qty <= 0:
            log.warning("Zero quantity for %s — skip", signal.symbol)
            return False

        if signal.direction == SignalDirection.LONG:
            fill = self._paper.market_buy(signal.symbol, qty, price)
        else:
            fill = self._paper.market_sell(signal.symbol, qty, price)

        pos = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=fill.fill_price,
            quantity=qty,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            highest_price=fill.fill_price,
            lowest_price=fill.fill_price,
            atr=signal.atr,
            leverage=leverage,
            regime=signal.regime.value,
            strategy=signal.strategy,
            signal_confidence=signal.confidence,
            kelly_fraction=risk_fraction,
            stop_distance=signal.stop_distance,
            reward_risk_ratio=signal.reward_risk_ratio,
            risk_fraction=risk_fraction,
            risk_dollar=risk_dollar,
        )
        self._portfolio.open_position(pos)
        return True

    def _live_open(
        self,
        signal: Signal,
        quantity: float,
        price: float,
        leverage: int,
        risk_dollar: float,
        risk_fraction: float,
    ) -> bool:
        if not self._rest:
            log.error("No REST client configured for live trading")
            return False

        try:
            hold_side = "long" if signal.direction == SignalDirection.LONG else "short"
            self._rest.set_leverage(signal.symbol, leverage, hold_side)
        except Exception as e:
            log.error("Set leverage failed for %s: %s", signal.symbol, e)

        qty = round(quantity, 4)
        if qty <= 0:
            return False

        side = "buy" if signal.direction == SignalDirection.LONG else "sell"
        trade_side = "open"
        # holdSide selects which leg of the hedge position to open (passivbot pattern)
        # open + buy  → long leg;  open + sell → short leg
        hold_side = "long" if signal.direction == SignalDirection.LONG else "short"
        client_oid = f"te_{signal.symbol[:6]}_{int(time.time())}"

        for attempt in range(self._cfg.order_retry_attempts):
            try:
                result = self._rest.place_order(
                    symbol=signal.symbol,
                    side=side,
                    trade_side=trade_side,
                    order_type="market",
                    size=qty,
                    client_oid=client_oid,
                    hold_side=hold_side,
                )
                order_id = result.get("orderId")
                log.info("Live order placed: %s orderId=%s holdSide=%s", signal.symbol, order_id, hold_side)

                pos = Position(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    entry_price=price,
                    quantity=qty,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    highest_price=price,
                    lowest_price=price,
                    atr=signal.atr,
                    leverage=leverage,
                    regime=signal.regime.value,
                    strategy=signal.strategy,
                    signal_confidence=signal.confidence,
                    kelly_fraction=risk_fraction,
                    stop_distance=signal.stop_distance,
                    reward_risk_ratio=signal.reward_risk_ratio,
                    risk_fraction=risk_fraction,
                    risk_dollar=risk_dollar,
                )
                self._portfolio.open_position(pos)

                try:
                    self._rest.place_stop_order(
                        symbol=signal.symbol,
                        plan_type="loss_plan",
                        position_direction=signal.direction.value,  # "LONG" | "SHORT"
                        size=qty,
                        trigger_price=signal.stop_loss,
                    )
                except Exception as e:
                    log.error("Stop order failed for %s: %s", signal.symbol, e)

                return True

            except Exception as e:
                log.error("Order attempt %d failed for %s: %s", attempt + 1, signal.symbol, e)
                time.sleep(self._cfg.order_retry_delay_seconds)

        log.error("All order attempts failed for %s", signal.symbol)
        return False

    def _live_close(self, pos: Position, price: float) -> None:
        if not self._rest:
            return
        # Closing a LONG = sell the long leg;  closing a SHORT = buy the short leg
        # holdSide must match the position being closed (passivbot pattern)
        side = "sell" if pos.direction == SignalDirection.LONG else "buy"
        hold_side = "long" if pos.direction == SignalDirection.LONG else "short"
        for attempt in range(self._cfg.order_retry_attempts):
            try:
                self._rest.place_order(
                    symbol=pos.symbol,
                    side=side,
                    trade_side="close",
                    order_type="market",
                    size=pos.quantity,
                    hold_side=hold_side,
                )
                return
            except Exception as e:
                log.error("Close attempt %d failed for %s: %s", attempt + 1, pos.symbol, e)
                time.sleep(self._cfg.order_retry_delay_seconds)

    def reset_cycle_counter(self) -> None:
        """Call at the start of each scan cycle to reset the per-cycle trade limit."""
        self._trades_this_cycle = 0

    def _can_trade_now(self) -> bool:
        window_seconds = self._cfg.trade_frequency_window_seconds
        limit = self._cfg.max_trades_per_window
        now = time.time()
        while self._recent_trades and now - self._recent_trades[0] > window_seconds:
            self._recent_trades.popleft()
        return len(self._recent_trades) < limit
