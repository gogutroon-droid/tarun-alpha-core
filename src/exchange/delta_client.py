import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Union
import requests
import pandas as pd

logger = logging.getLogger("DeltaClient")


class DeltaAPIException(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None, response_body: Optional[Any] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class DeltaSecurityException(Exception):
    pass


class DeltaExchangeClient:

    TESTNET_BASE_URL = "https://cdn-ind.testnet.deltaex.org"
    TESTNET_PUB_WSS = "wss://socket-ind-pub.testnet.deltaex.org"
    TESTNET_PRV_WSS = "wss://socket-ind.testnet.deltaex.org"
    PRODUCTION_FORBIDDEN_URL = "https://api.india.delta.exchange"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        environment: str = "testnet",
        base_url: Optional[str] = None,
        trading_enabled: bool = False,
        timeout: int = 10,
    ):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.environment = (environment or "testnet").lower().strip()
        self.base_url = (base_url or self.TESTNET_BASE_URL).rstrip("/")
        self.trading_enabled = bool(trading_enabled)
        self.timeout = timeout
        self.session = requests.Session()
        self._products_cache: Dict[str, dict] = {}
        self._products_by_id: Dict[int, dict] = {}
        self._enforce_safety_guardrails()

    def _enforce_safety_guardrails(self):
        if self.environment != "testnet":
            raise DeltaSecurityException(
                f"FATAL SAFETY ERROR: Environment '{self.environment}' is not allowed. "
                "Only 'testnet' (Delta Exchange India Demo) is supported in this build."
            )

        if "api.india.delta.exchange" in self.base_url.lower() or "api.delta.exchange" in self.base_url.lower():
            raise DeltaSecurityException(
                f"FATAL SAFETY ERROR: Base URL '{self.base_url}' points to production real money! "
                f"Must use Demo/Testnet endpoint: '{self.TESTNET_BASE_URL}'."
            )

    def set_trading_enabled(self, enabled: bool):
        self.trading_enabled = bool(enabled)
        if self.trading_enabled:
            logger.warning("[SAFETY] Testnet trading has been ENABLED.")
        else:
            logger.info("[SAFETY] Testnet trading has been DISABLED.")

    def _generate_signature(
        self,
        method: str,
        timestamp: str,
        path: str,
        query_string: str = "",
        body_string: str = "",
    ) -> str:
        if not self.api_secret:
            raise DeltaSecurityException("API Secret is required for authenticated requests.")

        payload = f"{method.upper()}{timestamp}{path}{query_string}{body_string}"
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Any:
        self._enforce_safety_guardrails()

        path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        url = f"{self.base_url}{path}"
        method = method.upper()

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "alpha-core/1.0",
        }

        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        query_string = f"?{urllib.parse.urlencode(clean_params)}" if clean_params else ""
        body_string = json.dumps(data) if (data is not None and method in ["POST", "PUT", "DELETE"]) else ""

        if auth:
            if not self.api_key or not self.api_secret:
                raise DeltaSecurityException("API Key and Secret must be configured for authenticated requests.")

            timestamp = str(int(time.time()))
            signature = self._generate_signature(
                method=method,
                timestamp=timestamp,
                path=path,
                query_string=query_string,
                body_string=body_string,
            )

            headers["api-key"] = self.api_key
            headers["signature"] = signature
            headers["timestamp"] = timestamp

        try:
            if method == "GET":
                response = self.session.get(url, params=clean_params, headers=headers, timeout=self.timeout)
            elif method == "POST":
                response = self.session.post(url, params=clean_params, data=body_string, headers=headers, timeout=self.timeout)
            elif method == "PUT":
                response = self.session.put(url, params=clean_params, data=body_string, headers=headers, timeout=self.timeout)
            elif method == "DELETE":
                response = self.session.delete(url, params=clean_params, data=body_string if body_string else None, headers=headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            try:
                res_json = response.json()
            except Exception:
                res_json = {"raw_text": response.text}

            if not response.ok:
                err_msg = res_json.get("error", {}).get("message") if isinstance(res_json, dict) and "error" in res_json else response.text
                logger.error(f"Delta API error on {method} {path}: HTTP {response.status_code} - {err_msg}")
                raise DeltaAPIException(
                    message=f"HTTP {response.status_code}: {err_msg}",
                    status_code=response.status_code,
                    response_body=res_json,
                )

            if isinstance(res_json, dict) and res_json.get("success") is False:
                err_info = res_json.get("error", {})
                err_msg = err_info.get("message", "Unknown Delta API error")
                err_code = err_info.get("code", "UNKNOWN")
                raise DeltaAPIException(f"Delta Error [{err_code}]: {err_msg}", response_body=res_json)

            return res_json

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error on {method} {url}: {e}")
            raise DeltaAPIException(f"Network connection failed: {e}")

    def ping(self) -> bool:
        try:
            self.get_assets()
            return True
        except Exception as e:
            logger.warning(f"Delta ping failed: {e}")
            return False

    def get_assets(self) -> List[Dict[str, Any]]:
        res = self._request("GET", "/v2/assets")
        return res.get("result", []) if isinstance(res, dict) else res

    def get_products(self, refresh: bool = False) -> List[Dict[str, Any]]:
        if self._products_cache and not refresh:
            return list(self._products_cache.values())

        res = self._request("GET", "/v2/products")
        products = res.get("result", []) if isinstance(res, dict) else res

        self._products_cache.clear()
        self._products_by_id.clear()

        for p in products:
            sym = p.get("symbol", "").upper()
            p_id = p.get("id")
            if sym:
                self._products_cache[sym] = p
            if p_id is not None:
                self._products_by_id[p_id] = p

        logger.info(f"Cached {len(self._products_cache)} products from Delta Exchange.")
        return products

    def get_product(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self._products_cache:
            self.get_products()

        sym_clean = symbol.upper()
        if sym_clean in self._products_cache:
            return self._products_cache[sym_clean]

        for k, v in self._products_cache.items():
            if k == sym_clean or k.replace("_", "") == sym_clean.replace("_", ""):
                return v

        return None

    def get_product_id(self, symbol: str) -> Optional[int]:
        prod = self.get_product(symbol)
        return prod.get("id") if prod else None

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        sym_clean = symbol.upper()
        try:
            res = self._request("GET", f"/v2/tickers/{sym_clean}")
            return res.get("result", res) if isinstance(res, dict) else res
        except DeltaAPIException:
            all_tickers = self.get_tickers()
            for t in all_tickers:
                if t.get("symbol", "").upper() == sym_clean:
                    return t
            raise DeltaAPIException(f"Ticker for symbol '{symbol}' not found on Delta Exchange.")

    def get_tickers(self) -> List[Dict[str, Any]]:
        res = self._request("GET", "/v2/tickers")
        return res.get("result", []) if isinstance(res, dict) else res

    def get_mark_price(self, symbol: str) -> float:
        t = self.get_ticker(symbol)
        price_val = t.get("mark_price") or t.get("close") or t.get("spot_price") or t.get("price")
        if price_val is None:
            raise DeltaAPIException(f"Could not extract price from ticker: {t}")
        return float(price_val)

    def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        sym_clean = symbol.upper()
        res = self._request("GET", f"/v2/l2orderbook/{sym_clean}")
        return res.get("result", res) if isinstance(res, dict) else res

    def get_candles(
        self,
        symbol: str,
        resolution: str = "15m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        res_map = {
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        res_str = res_map.get(resolution.lower(), resolution)

        step_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
        }
        step_seconds = step_map.get(res_str, 900)
        end_ts = int(end_time) if end_time is not None else int(time.time())
        start_ts = int(start_time) if start_time is not None else int(end_ts - (limit * step_seconds))

        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "resolution": res_str,
            "start": start_ts,
            "end": end_ts,
        }

        try:
            res = self._request("GET", "/v2/history/candles", params=params)
            candle_data = res.get("result", []) if isinstance(res, dict) else res
        except DeltaAPIException:
            chart_params = {
                "symbol": symbol.upper(),
                "resolution": res_str,
                "from": start_ts,
                "to": end_ts,
            }
            res = self._request("GET", "/v2/chart/history", params=chart_params)
            candle_data = res.get("result", []) if isinstance(res, dict) else res

        if not candle_data:
            return pd.DataFrame(columns=["timestamp", "datetime", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(candle_data)

        col_rename = {
            "t": "timestamp",
            "time": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
        df = df.rename(columns=col_rename)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        if "timestamp" in df.columns:
            sample_ts = df["timestamp"].iloc[0]
            if sample_ts < 1e11:
                df["timestamp"] = (df["timestamp"] * 1000).astype(int)
            else:
                df["timestamp"] = df["timestamp"].astype(int)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

        df = df.sort_values("timestamp").reset_index(drop=True)
        if limit and len(df) > limit:
            df = df.iloc[-limit:].reset_index(drop=True)

        return df

    def get_balances(self) -> List[Dict[str, Any]]:
        res = self._request("GET", "/v2/wallet/balances", auth=True)
        return res.get("result", []) if isinstance(res, dict) else res

    def get_wallet_balance_dict(self) -> Dict[str, Dict[str, float]]:
        raw_list = self.get_balances()
        balance_map = {}
        for b in raw_list:
            asset = b.get("asset_symbol") or b.get("asset", {}).get("symbol") or str(b.get("asset_id", "UNKNOWN"))
            balance_map[asset.upper()] = {
                "balance": float(b.get("balance", 0.0)),
                "available": float(b.get("available_balance", b.get("available", 0.0))),
                "locked": float(b.get("locked_balance", 0.0)),
            }
        return balance_map

    def get_positions(self) -> List[Dict[str, Any]]:
        try:
            res = self._request("GET", "/v2/positions/margined", auth=True)
            return res.get("result", []) if isinstance(res, dict) else res
        except DeltaAPIException:
            res = self._request("GET", "/v2/positions", auth=True)
            return res.get("result", []) if isinstance(res, dict) else res

    def get_open_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        positions = self.get_positions()
        prod = self.get_product(symbol)
        prod_id = prod.get("id") if prod else None
        sym_clean = symbol.upper()

        for pos in positions:
            p_id = pos.get("product_id")
            p_sym = pos.get("product_symbol", "").upper()
            size = float(pos.get("size", 0.0))

            if (p_id == prod_id or p_sym == sym_clean) and size != 0.0:
                entry_p = float(pos.get("entry_price", 0.0))
                mark_p = float(pos.get("mark_price", entry_p))
                unrealized = float(pos.get("unrealized_pnl", 0.0))
                leverage = float(pos.get("leverage", 10.0))
                return {
                    "product_id": p_id,
                    "symbol": p_sym or sym_clean,
                    "size": size,
                    "direction": "LONG" if size > 0 else "SHORT",
                    "entry_price": entry_p,
                    "mark_price": mark_p,
                    "unrealized_pnl": unrealized,
                    "leverage": leverage,
                }
        return None

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"state": "open"}
        if symbol:
            prod_id = self.get_product_id(symbol)
            if prod_id:
                params["product_id"] = prod_id

        res = self._request("GET", "/v2/orders", params=params, auth=True)
        return res.get("result", []) if isinstance(res, dict) else res

    def get_order_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit}
        if symbol:
            prod_id = self.get_product_id(symbol)
            if prod_id:
                params["product_id"] = prod_id

        res = self._request("GET", "/v2/orders/history", params=params, auth=True)
        return res.get("result", []) if isinstance(res, dict) else res

    def get_order_status(self, order_id: Union[int, str]) -> Dict[str, Any]:
        res = self._request("GET", f"/v2/orders/{order_id}", auth=True)
        return res.get("result", res) if isinstance(res, dict) else res

    def _assert_trading_allowed(self, action_name: str):
        self._enforce_safety_guardrails()
        if not self.trading_enabled:
            raise DeltaSecurityException(
                f"TRADE REJECTED: Trading is currently DISABLED (TRADING_ENABLED=false). "
                f"Action '{action_name}' blocked at the client safety level."
            )

    def set_leverage(self, symbol: str, leverage: int = 10) -> Dict[str, Any]:
        self._assert_trading_allowed("set_leverage")
        prod_id = self.get_product_id(symbol)
        if not prod_id:
            raise DeltaAPIException(f"Cannot set leverage: Unknown product '{symbol}'")

        data = {
            "product_id": prod_id,
            "leverage": str(leverage),
        }
        try:
            res = self._request("POST", f"/v2/products/{prod_id}/leverage", data=data, auth=True)
            return res
        except DeltaAPIException:
            return self._request("POST", "/v2/orders/leverage", data=data, auth=True)

    def place_order(
        self,
        symbol: str,
        size: int,
        side: str,
        order_type: str = "market_order",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        self._assert_trading_allowed("place_order")

        prod = self.get_product(symbol)
        if not prod:
            raise DeltaAPIException(f"Unknown symbol: {symbol}")

        prod_id = prod["id"]
        side_clean = side.lower().strip()
        if side_clean not in ["buy", "sell"]:
            raise ValueError(f"Invalid order side: {side}. Must be 'buy' or 'sell'.")

        order_data: Dict[str, Any] = {
            "product_id": prod_id,
            "size": int(size),
            "side": side_clean,
            "order_type": order_type.lower().strip(),
            "reduce_only": bool(reduce_only),
        }

        if limit_price is not None:
            order_data["limit_price"] = str(limit_price)

        if stop_price is not None:
            order_data["stop_price"] = str(stop_price)

        if client_order_id:
            order_data["client_order_id"] = str(client_order_id)

        logger.info(
            f"[TESTNET ORDER] Submitting {side_clean.upper()} {size} contracts of {symbol} "
            f"(Type: {order_type}, Limit: {limit_price}, Stop: {stop_price}, ClientID: {client_order_id})"
        )

        res = self._request("POST", "/v2/orders", data=order_data, auth=True)
        return res.get("result", res) if isinstance(res, dict) else res

    def cancel_order(self, order_id: Union[int, str], product_id: Optional[int] = None) -> Dict[str, Any]:
        self._assert_trading_allowed("cancel_order")
        data: Dict[str, Any] = {"id": order_id}
        if product_id is not None:
            data["product_id"] = product_id

        res = self._request("DELETE", "/v2/orders", data=data, auth=True)
        return res.get("result", res) if isinstance(res, dict) else res

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        self._assert_trading_allowed("close_position")
        pos = self.get_open_position(symbol)
        if not pos or pos["size"] == 0:
            logger.info(f"No open position found to close for {symbol}.")
            return None

        size = abs(int(pos["size"]))
        exit_side = "sell" if pos["size"] > 0 else "buy"
        logger.info(f"[TESTNET CLOSE] Closing active {pos['direction']} position of {size} contracts on {symbol}...")

        return self.place_order(
            symbol=symbol,
            size=size,
            side=exit_side,
            order_type="market_order",
            reduce_only=True,
        )
