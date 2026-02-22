"""
bitget_rest.py - Bitget REST API Client
Handles authentication, request signing, and all REST endpoints used by the engine.
"""
import hashlib
import hmac
import base64
import time
import json
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import certifi

from config import get_config
from logger import get_logger

log = get_logger("bitget_rest")


def _sign(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    """Generate Bitget HMAC-SHA256 signature."""
    prehash = f"{timestamp}{method.upper()}{path}{body}"
    mac = hmac.new(secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256)
    return base64.b64encode(mac.digest()).decode()


def _ts() -> str:
    return str(int(time.time() * 1000))


class BitgetRestClient:
    def __init__(self) -> None:
        cfg = get_config()
        self._api_key = cfg.api.api_key
        self._secret = cfg.api.secret_key
        self._passphrase = cfg.api.passphrase
        self._base_url = cfg.api.base_url

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session = requests.Session()
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        
        # Configure SSL verification
        if cfg.api.disable_ssl_verification:
            log.warning("⚠️  SSL VERIFICATION DISABLED - Use only for development with SSL inspection issues")
            self._session.verify = False
            # Suppress urllib3 warnings about unverified HTTPS
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        else:
            # Use certifi's CA bundle for SSL verification
            self._session.verify = certifi.where()

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = _ts()
        return {
            "ACCESS-KEY": self._api_key,
            "ACCESS-SIGN": _sign(self._secret, ts, method, path, body),
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }

    def _get(self, path: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        url = self._base_url + path
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            full_path = f"{path}?{query}"
        else:
            full_path = path
            query = ""

        headers = self._headers("GET", full_path) if signed else {"Content-Type": "application/json"}
        resp = self._session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        
        # Add better error handling for non-JSON responses (e.g., blocking pages)
        try:
            data = resp.json()
        except Exception as e:
            log.error(f"Failed to parse JSON response from {url}")
            log.error(f"Response status: {resp.status_code}")
            log.error(f"Response content (first 500 chars): {resp.text[:500]}")
            raise RuntimeError(f"Invalid JSON response from API - possible network filtering/blocking") from e
            
        if data.get("code") not in ("00000", 0, "0"):
            raise RuntimeError(f"Bitget API error: {data.get('code')} - {data.get('msg')}")
        return data

    def _post(self, path: str, payload: Dict) -> Dict:
        body = json.dumps(payload)
        headers = self._headers("POST", path, body)
        url = self._base_url + path
        resp = self._session.post(url, headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        log.debug("POST %s → %s", path, data)
        if data.get("code") not in ("00000", 0, "0"):
            raise RuntimeError(f"Bitget API error: {data.get('code')} - {data.get('msg')}")
        return data

    # ------------------------------------------------------------------ #
    # Market Data
    # ------------------------------------------------------------------ #

    def get_usdt_perpetuals(self) -> List[Dict]:
        """Return all USDT-margined perpetual contracts."""
        data = self._get("/api/v2/mix/market/tickers", params={"productType": "USDT-FUTURES"})
        return data.get("data", [])

    def get_candles(self, symbol: str, granularity: str = "15m", limit: int = 200) -> List[List]:
        """
        Fetch OHLCV candles.
        Returns list of [timestamp, open, high, low, close, volume, quoteVolume].
        """
        params = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "granularity": granularity,
            "limit": str(limit),
        }
        data = self._get("/api/v2/mix/market/candles", params=params)
        return data.get("data", [])

    def get_ticker(self, symbol: str) -> Dict:
        data = self._get(
            "/api/v2/mix/market/ticker",
            params={"symbol": symbol, "productType": "USDT-FUTURES"},
        )
        return data.get("data", {})

    def get_orderbook(self, symbol: str, limit: int = 5) -> Dict:
        data = self._get(
            "/api/v2/mix/market/orderbook",
            params={"symbol": symbol, "productType": "USDT-FUTURES", "limit": str(limit)},
        )
        return data.get("data", {})

    # ------------------------------------------------------------------ #
    # Account
    # ------------------------------------------------------------------ #

    def get_account(self, margin_coin: str = "USDT") -> Dict:
        data = self._get(
            "/api/v2/mix/account/account",
            params={"productType": "USDT-FUTURES", "marginCoin": margin_coin},
            signed=True,
        )
        return data.get("data", {})

    def get_positions(self) -> List[Dict]:
        data = self._get(
            "/api/v2/mix/position/all-position",
            params={"productType": "USDT-FUTURES", "marginCoin": "USDT"},
            signed=True,
        )
        return data.get("data", [])

    def set_leverage(self, symbol: str, leverage: int, hold_side: str = "long") -> Dict:
        payload = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "leverage": str(leverage),
            "holdSide": hold_side,
        }
        data = self._post("/api/v2/mix/account/set-leverage", payload)
        return data.get("data", {})

    def set_position_mode(self, hedge: bool = True) -> Dict:
        """Set account-level position mode. hedge=True → hedge_mode (required for long+short simultaneously)."""
        payload = {
            "productType": "USDT-FUTURES",
            "posMode": "hedge_mode" if hedge else "one_way_mode",
        }
        data = self._post("/api/v2/mix/account/set-position-mode", payload)
        log.info("Position mode set: %s", "hedge_mode" if hedge else "one_way_mode")
        return data.get("data", {})

    def set_margin_mode(self, symbol: str, margin_mode: str = "crossed") -> Dict:
        """Set margin mode per symbol. Use 'crossed' (cross margin) — matches passivbot."""
        payload = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "marginMode": margin_mode,
        }
        data = self._post("/api/v2/mix/account/set-margin-mode", payload)
        return data.get("data", {})

    def initialize_account(self, symbols: Optional[List[str]] = None, leverage: int = 3) -> None:
        """
        One-time live account setup (mirrors passivbot's update_exchange_config):
          1. Set hedge mode on the account
          2. For each active symbol: set cross margin + set leverage for both sides
        Safe to call on every startup — exchange ignores if already set.
        """
        try:
            self.set_position_mode(hedge=True)
        except Exception as e:
            log.warning("set_position_mode failed (may already be set): %s", e)

        if symbols:
            for sym in symbols:
                try:
                    self.set_margin_mode(sym, "crossed")
                except Exception as e:
                    log.warning("set_margin_mode failed for %s: %s", sym, e)
                for side in ("long", "short"):
                    try:
                        self.set_leverage(sym, leverage, hold_side=side)
                    except Exception as e:
                        log.warning("set_leverage failed for %s %s: %s", sym, side, e)
                log.info("Account initialized: %s cross×%d hedge", sym, leverage)

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #

    def place_order(
        self,
        symbol: str,
        side: str,            # "buy" | "sell"
        trade_side: str,      # "open" | "close"
        order_type: str,      # "market" | "limit"
        size: float,
        price: Optional[float] = None,
        client_oid: Optional[str] = None,
        hold_side: Optional[str] = None,  # "long" | "short" — REQUIRED for hedge mode
    ) -> Dict:
        """
        Place a futures order.

        Bitget hedge-mode requirements (from passivbot research):
          - holdSide must be explicit: "long" or "short"
          - marginMode must be "crossed" (not "isolated") for cross margin
          - timeInForce="post_only" for limit orders ("PO", not "GTX" like Binance)

        holdSide derivation:
          open  + buy   → long position  → holdSide="long"
          open  + sell  → short position → holdSide="short"
          close + sell  → closing long   → holdSide="long"
          close + buy   → closing short  → holdSide="short"
        """
        # Derive holdSide if not explicitly given
        if hold_side is None:
            if trade_side == "open":
                hold_side = "long" if side == "buy" else "short"
            else:  # close
                hold_side = "long" if side == "sell" else "short"

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginMode": "crossed",      # cross margin (passivbot default)
            "marginCoin": "USDT",
            "size": str(size),
            "side": side,
            "tradeSide": trade_side,
            "holdSide": hold_side,        # required for hedge mode
            "orderType": order_type,
        }
        if price is not None:
            payload["price"] = str(price)
            payload["timeInForce"] = "post_only"  # "PO" on Bitget — avoids taker fee on limit orders
        if client_oid:
            payload["clientOid"] = client_oid[:64]  # Bitget max 64 chars

        data = self._post("/api/v2/mix/order/place-order", payload)
        log.info(
            "Order placed: %s %s %s holdSide=%s size=%s → orderId=%s",
            symbol, side, trade_side, hold_side, size,
            data.get("data", {}).get("orderId"),
        )
        return data.get("data", {})

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        payload = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "orderId": order_id,
        }
        data = self._post("/api/v2/mix/order/cancel-order", payload)
        return data.get("data", {})

    def get_order(self, symbol: str, order_id: str) -> Dict:
        data = self._get(
            "/api/v2/mix/order/detail",
            params={"symbol": symbol, "productType": "USDT-FUTURES", "orderId": order_id},
            signed=True,
        )
        return data.get("data", {})

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        params: Dict[str, str] = {"productType": "USDT-FUTURES"}
        if symbol:
            params["symbol"] = symbol
        data = self._get("/api/v2/mix/order/orders-pending", params=params, signed=True)
        return data.get("data", {}).get("entrustedList", [])

    def place_stop_order(
        self,
        symbol: str,
        plan_type: str,         # "loss_plan" | "profit_plan" | "moving_plan"
        position_direction: str, # "LONG" | "SHORT" — the direction of the position being hedged
        size: float,
        trigger_price: float,
        execute_price: Optional[float] = None,
    ) -> Dict:
        """
        Place a TP/SL stop order.

        holdSide = direction of the position being protected:
          LONG position  → stop is a sell → holdSide="long"
          SHORT position → stop is a buy  → holdSide="short"
        """
        hold_side = "long" if position_direction.upper() == "LONG" else "short"
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "planType": plan_type,
            "triggerPrice": str(trigger_price),
            "triggerType": "mark_price",
            "executePrice": str(execute_price) if execute_price else "0",
            "holdSide": hold_side,
            "size": str(size),
            "orderType": "market" if execute_price is None else "limit",
        }
        data = self._post("/api/v2/mix/order/place-tpsl-order", payload)
        return data.get("data", {})
