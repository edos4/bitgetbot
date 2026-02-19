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
        data = resp.json()
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

    # ------------------------------------------------------------------ #
    # Orders
    # ------------------------------------------------------------------ #

    def place_order(
        self,
        symbol: str,
        side: str,           # "buy" | "sell"
        trade_side: str,     # "open" | "close"
        order_type: str,     # "market" | "limit"
        size: float,
        price: Optional[float] = None,
        client_oid: Optional[str] = None,
    ) -> Dict:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginMode": "isolated",
            "marginCoin": "USDT",
            "size": str(size),
            "side": side,
            "tradeSide": trade_side,
            "orderType": order_type,
        }
        if price is not None:
            payload["price"] = str(price)
        if client_oid:
            payload["clientOid"] = client_oid

        data = self._post("/api/v2/mix/order/place-order", payload)
        log.info("Order placed: %s %s %s size=%s → orderId=%s",
                 symbol, side, trade_side, size, data.get("data", {}).get("orderId"))
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
        plan_type: str,       # "loss_plan" | "profit_plan" | "moving_plan"
        side: str,
        size: float,
        trigger_price: float,
        execute_price: Optional[float] = None,
    ) -> Dict:
        payload: Dict[str, Any] = {
            "symbol": symbol,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "planType": plan_type,
            "triggerPrice": str(trigger_price),
            "triggerType": "mark_price",
            "executePrice": str(execute_price) if execute_price else "0",
            "holdSide": "long" if side == "sell" else "short",
            "size": str(size),
            "orderType": "market" if execute_price is None else "limit",
        }
        data = self._post("/api/v2/mix/order/place-tpsl-order", payload)
        return data.get("data", {})
