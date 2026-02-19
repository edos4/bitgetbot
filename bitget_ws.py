"""
bitget_ws.py - Bitget WebSocket Feed
Manages real-time price subscriptions for live trading mode.
Thread-safe price cache with callbacks.
"""
import json
import threading
import time
import hashlib
import hmac
import base64
from typing import Callable, Dict, List, Optional, Set

import websocket

from config import get_config
from logger import get_logger

log = get_logger("bitget_ws")


class BitgetWebSocket:
    """
    Manages a single WebSocket connection to Bitget.
    Subscribes to ticker channels for multiple symbols.
    Thread-safe price cache. Reconnects on disconnect.
    """

    PING_INTERVAL = 20  # seconds

    def __init__(self, on_price_update: Optional[Callable[[str, float], None]] = None) -> None:
        cfg = get_config()
        self._ws_url = cfg.api.ws_url
        self._on_price_update = on_price_update
        self._price_cache: Dict[str, float] = {}
        self._subscribed: Set[str] = set()
        self._lock = threading.Lock()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = threading.Event()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True, name="ws-thread")
        self._thread.start()
        # Wait up to 10s for connection
        self._connected.wait(timeout=10)
        log.info("WebSocket started")

    def stop(self) -> None:
        self._running = False
        if self._ws:
            self._ws.close()
        log.info("WebSocket stopped")

    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to ticker feed for list of symbols."""
        new_syms = [s for s in symbols if s not in self._subscribed]
        if not new_syms:
            return
        args = [{"instType": "USDT-FUTURES", "channel": "ticker", "instId": sym} for sym in new_syms]
        msg = json.dumps({"op": "subscribe", "args": args})
        if self._ws and self._connected.is_set():
            self._ws.send(msg)
            with self._lock:
                self._subscribed.update(new_syms)
            log.info("Subscribed to: %s", new_syms)
        else:
            log.warning("WS not connected – queuing subscriptions")
            # Will be sent on next connect via _on_open

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._price_cache.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._price_cache)

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _run_forever(self) -> None:
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self._ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=self.PING_INTERVAL, ping_timeout=10)
            except Exception as e:
                log.error("WebSocket error: %s", e)
            if self._running:
                log.warning("WebSocket disconnected – reconnecting in 5s")
                self._connected.clear()
                time.sleep(5)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self._connected.set()
        log.info("WebSocket connected")
        # Re-subscribe to all symbols
        with self._lock:
            syms = list(self._subscribed)
        if syms:
            args = [{"instType": "USDT-FUTURES", "channel": "ticker", "instId": s} for s in syms]
            ws.send(json.dumps({"op": "subscribe", "args": args}))

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            data = json.loads(message)

            # Heartbeat
            if data == {"event": "pong"} or message == "pong":
                return

            if "data" not in data:
                return

            for item in data["data"]:
                symbol = item.get("instId", "")
                last_price_str = item.get("lastPr") or item.get("last")
                if symbol and last_price_str:
                    price = float(last_price_str)
                    with self._lock:
                        self._price_cache[symbol] = price
                    if self._on_price_update:
                        self._on_price_update(symbol, price)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log.debug("WS message parse error: %s | msg=%s", e, message[:200])

    def _on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        log.error("WebSocket error: %s", error)

    def _on_close(self, ws: websocket.WebSocketApp, close_status_code: int, close_msg: str) -> None:
        self._connected.clear()
        log.warning("WebSocket closed: %s %s", close_status_code, close_msg)
