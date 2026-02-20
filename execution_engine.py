"""execution_engine.py - Unified Execution Router with Kelly sizing and risk analytics."""
import time
import threading
from collections import deque
from typing import Optional

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

        if not self._can_trade_now():
            log.info("❌ Trade blocked [%s]: trade frequency cap hit", signal.symbol)
            return False

        if signal.stop_distance <= 0:
            log.warning("Stop distance invalid for %s — skip", signal.symbol)
            return False

        reward_risk = max(signal.reward_risk_ratio, 0.01)
        base_fraction = self._stats.get_kelly_fraction(signal.strategy, reward_risk)
        if base_fraction <= 0:
            log.info("❌ Trade blocked [%s]: Kelly fraction ≤ 0", signal.symbol)
            return False

        risk_fraction = min(base_fraction, self._cfg.max_risk_per_trade_pct)

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

        if signal.atr_ratio > self._cfg.atr_volatility_multiplier:
            risk_fraction *= 0.5

        existing_syms = self._portfolio.get_open_symbols()

        decision = self._risk.evaluate(
            symbol=signal.symbol,
            requested_risk_fraction=risk_fraction,
            current_equity=equity,
            existing_positions=existing_syms,
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
        if self._portfolio.breaches_variance_cap(notional_usd, signal.entry_price, signal.atr):
            log.info("❌ Trade blocked [%s]: portfolio variance cap exceeded", signal.symbol)
            return False

        stats_snapshot = self._stats.get_snapshot(signal.strategy)
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
            self._recent_trades.append(time.time())
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
        pos = self._portfolio.get_position(symbol)
        if not pos:
            return False

        log.info("Closing %s @ %.4f | reason=%s", symbol, current_price, reason)

        if self._mode == "paper":
            fill = (
                self._paper.market_sell(symbol, pos.quantity, current_price)
                if pos.direction == SignalDirection.LONG
                else self._paper.market_buy(symbol, pos.quantity, current_price)
            )
            if pos.direction == SignalDirection.LONG:
                pnl = (fill.fill_price - pos.entry_price) * pos.quantity
            else:
                pnl = (pos.entry_price - fill.fill_price) * pos.quantity
            self._paper.record_trade_pnl(pnl, fill.fee_paid)
        else:
            self._live_close(pos, current_price)

        self._portfolio.close_position(symbol, current_price, reason)
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
                )
                order_id = result.get("orderId")
                log.info("Live order placed: %s orderId=%s", signal.symbol, order_id)

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
                        side="sell" if signal.direction == SignalDirection.LONG else "buy",
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
        side = "sell" if pos.direction == SignalDirection.LONG else "buy"
        for attempt in range(self._cfg.order_retry_attempts):
            try:
                self._rest.place_order(
                    symbol=pos.symbol,
                    side=side,
                    trade_side="close",
                    order_type="market",
                    size=pos.quantity,
                )
                return
            except Exception as e:
                log.error("Close attempt %d failed for %s: %s", attempt + 1, pos.symbol, e)
                time.sleep(self._cfg.order_retry_delay_seconds)

    def _can_trade_now(self) -> bool:
        window_seconds = self._cfg.trade_frequency_window_seconds
        limit = self._cfg.max_trades_per_window
        now = time.time()
        while self._recent_trades and now - self._recent_trades[0] > window_seconds:
            self._recent_trades.popleft()
        return len(self._recent_trades) < limit
