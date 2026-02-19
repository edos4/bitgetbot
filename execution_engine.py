"""
execution_engine.py - Unified Execution Router
Routes orders to either the paper engine or live Bitget REST API
depending on config.trading.mode.
Handles retries, partial fills, and stop order placement.
"""
import time
import threading
import uuid
from typing import Optional, Tuple

from config import get_config
from strategy import Signal, SignalDirection
from portfolio_manager import PortfolioManager, Position
from risk_manager import RiskManager
from paper_engine import PaperEngine
from logger import get_logger

log = get_logger("execution")


class ExecutionEngine:
    """
    Single interface for placing and managing trades.
    Mode is determined at construction from config.
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        risk: RiskManager,
        paper_engine: Optional[PaperEngine] = None,
        rest_client=None,
    ) -> None:
        self._cfg = get_config().trading
        self._portfolio = portfolio
        self._risk = risk
        self._paper = paper_engine
        self._rest = rest_client
        self._mode = self._cfg.mode
        self._lock = threading.Lock()
        log.info("ExecutionEngine initialized in %s mode", self._mode.upper())

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def execute_signal(
        self,
        signal: Signal,
        equity: float,
        current_prices: dict,
    ) -> bool:
        """
        Evaluate risk, size position, and send order.
        Returns True if order was placed successfully.
        """
        if signal.direction == SignalDirection.NONE:
            return False

        # Don't trade a symbol already held
        if self._portfolio.get_position(signal.symbol):
            log.debug("Skip %s — position already open", signal.symbol)
            return False

        # Stop distance as a fraction of price
        stop_pct = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
        if stop_pct < 0.001:
            log.warning("Stop too tight for %s (%.4f%%) — skip", signal.symbol, stop_pct * 100)
            return False

        existing_syms = self._portfolio.get_open_symbols()
        corr = self._risk.get_correlated_with(signal.symbol)

        decision = self._risk.evaluate(
            symbol=signal.symbol,
            stop_loss_pct=stop_pct,
            current_equity=equity,
            existing_positions=existing_syms,
            correlated_with=corr,
        )

        if not decision.approved:
            log.info("Trade blocked [%s]: %s", signal.symbol, decision.reason)
            return False

        # Apply size multiplier (HIGH_VOL reduction)
        position_usd = decision.position_size_usd * signal.size_multiplier

        success = self._place_order(signal, position_usd, decision.leverage)
        return success

    def close_position(
        self,
        symbol: str,
        current_price: float,
        reason: str = "MANUAL",
    ) -> bool:
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

    # ------------------------------------------------------------------ #
    # Order placement
    # ------------------------------------------------------------------ #

    def _place_order(self, signal: Signal, position_usd: float, leverage: int) -> bool:
        price = signal.entry_price

        if self._mode == "paper":
            return self._paper_open(signal, position_usd, price, leverage)
        else:
            return self._live_open(signal, position_usd, price, leverage)

    def _paper_open(
        self, signal: Signal, position_usd: float, price: float, leverage: int
    ) -> bool:
        qty = self._paper.compute_quantity(position_usd, price, leverage)
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
        )
        self._portfolio.open_position(pos)
        return True

    def _live_open(
        self, signal: Signal, position_usd: float, price: float, leverage: int
    ) -> bool:
        """Place live order on Bitget with retry logic."""
        if not self._rest:
            log.error("No REST client configured for live trading")
            return False

        try:
            # Set leverage
            hold_side = "long" if signal.direction == SignalDirection.LONG else "short"
            self._rest.set_leverage(signal.symbol, leverage, hold_side)
        except Exception as e:
            log.error("Set leverage failed for %s: %s", signal.symbol, e)

        # Estimate qty (Bitget uses base asset qty for contracts)
        qty = round((position_usd * leverage) / price, 4)
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

                # Register position (approximate fill at signal price)
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
                )
                self._portfolio.open_position(pos)

                # Place stop loss order
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
        """Close live position via market order."""
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
