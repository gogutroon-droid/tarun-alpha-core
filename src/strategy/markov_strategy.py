from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


class MarkovScalpingEngine:

    def __init__(
        self,
        markov_window: int = 10,
        markov_thresh: float = 0.02,
        ema_fast: int = 20,
        ema_slow: int = 50,
        atr_period: int = 14,
        sl_atr_mult: float = 2.0,
        rr_ratio: float = 15.0,
        max_hold_bars: int = 120,
        min_rvol: float = 1.20,
        min_wick_ratio: float = 0.15,
        rvol_ma_len: int = 20,
        pullback_tolerance_pct: float = 0.002,
    ):
        self.markov_window = markov_window
        self.markov_thresh = markov_thresh
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_period = atr_period
        self.sl_atr_mult = sl_atr_mult
        self.rr_ratio = rr_ratio
        self.max_hold_bars = max_hold_bars
        self.min_rvol = min_rvol
        self.min_wick_ratio = min_wick_ratio
        self.rvol_ma_len = rvol_ma_len
        self.pullback_tolerance_pct = pullback_tolerance_pct

    def compute_daily_regime(self, df_1d: pd.DataFrame) -> Dict[str, Any]:
        if len(df_1d) <= self.markov_window:
            return {"regime": "NEUTRAL", "daily_return": 0.0, "is_bull": False, "is_bear": False, "ret": 0.0}

        c_curr = float(df_1d["close"].iloc[-1])
        c_lag = float(df_1d["close"].iloc[-(self.markov_window + 1)])
        daily_ret = (c_curr - c_lag) / c_lag if c_lag > 0 else 0.0

        is_bull = daily_ret > self.markov_thresh
        is_bear = daily_ret < -self.markov_thresh

        if is_bull:
            regime = "BULL"
        elif is_bear:
            regime = "BEAR"
        else:
            regime = "CONSOLIDATION"

        return {
            "regime": regime,
            "daily_return": daily_ret,
            "ret": daily_ret,
            "is_bull": is_bull,
            "is_bear": is_bear,
            "c_curr": c_curr,
            "c_lag": c_lag,
        }

    def enrich_15m_data(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        df = df_15m.copy()

        df["ema20"] = df["close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=self.ema_slow, adjust=False).mean()

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(window=self.atr_period, min_periods=1).mean()

        df["vol_ma"] = df["volume"].rolling(window=self.rvol_ma_len, min_periods=1).mean()
        df["rvol"] = np.where(df["vol_ma"] > 0, df["volume"] / df["vol_ma"], 1.0)

        candle_range = df["high"] - df["low"]
        lower_wick = np.where(df["close"] >= df["open"], df["open"] - df["low"], df["close"] - df["low"])
        upper_wick = np.where(df["close"] >= df["open"], df["high"] - df["close"], df["high"] - df["open"])
        df["lower_wick_ratio"] = lower_wick / np.where(candle_range > 0, candle_range, 1.0)
        df["upper_wick_ratio"] = upper_wick / np.where(candle_range > 0, candle_range, 1.0)

        return df

    def evaluate_signal(
        self,
        df_1d: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> Optional[Dict[str, Any]]:
        if len(df_15m) < max(self.ema_slow, self.rvol_ma_len) + 5 or len(df_1d) <= self.markov_window:
            return None

        regime_info = self.compute_daily_regime(df_1d)
        if not regime_info["is_bull"] and not regime_info["is_bear"]:
            return None

        df_15m_enriched = self.enrich_15m_data(df_15m)
        last_row = df_15m_enriched.iloc[-1]

        c_close = float(last_row["close"])
        c_open = float(last_row["open"])
        c_high = float(last_row["high"])
        c_low = float(last_row["low"])
        c_ema20 = float(last_row["ema20"])
        c_ema50 = float(last_row["ema50"])
        c_atr = float(last_row["atr14"])
        c_rvol = float(last_row["rvol"])
        l_wick = float(last_row["lower_wick_ratio"])
        u_wick = float(last_row["upper_wick_ratio"])

        c_time = int(last_row["timestamp"]) if "timestamp" in last_row else int(pd.to_datetime(last_row["datetime"]).timestamp() * 1000)
        dt_str = str(last_row["datetime"]) if "datetime" in last_row else str(pd.to_datetime(c_time, unit="ms"))

        if c_rvol < self.min_rvol:
            return None

        if regime_info["is_bull"]:
            ema_aligned = c_ema20 > c_ema50
            pullback_ok = c_low <= (c_ema20 * (1.0 + self.pullback_tolerance_pct))
            rejection_ok = (l_wick >= self.min_wick_ratio) or (c_close > c_open)

            if ema_aligned and pullback_ok and rejection_ok:
                sl_dist = self.sl_atr_mult * c_atr
                if sl_dist <= 0:
                    return None

                sl_price = c_close - sl_dist
                tp_price = c_close + (sl_dist * self.rr_ratio)

                return {
                    "action": "ENTER_LONG",
                    "strategy": "MODEL_B_1to15",
                    "setup_name": "MARKOV_15M_BULL_PULLBACK",
                    "timestamp": c_time,
                    "datetime": dt_str,
                    "direction": "LONG",
                    "dir": "LONG",
                    "entry_price": c_close,
                    "entry": c_close,
                    "stop_loss": sl_price,
                    "sl": sl_price,
                    "stop_dist": sl_dist,
                    "tp_price": tp_price,
                    "tp": tp_price,
                    "tp1_price": tp_price,
                    "tp2_price": tp_price,
                    "rr_ratio": self.rr_ratio,
                    "sl_atr_mult": self.sl_atr_mult,
                    "atr": c_atr,
                    "max_bars": self.max_hold_bars,
                    "max_hold_bars": self.max_hold_bars,
                    "rvol": c_rvol,
                    "daily_return": regime_info["daily_return"],
                }

        elif regime_info["is_bear"]:
            ema_aligned = c_ema20 < c_ema50
            pullback_ok = c_high >= (c_ema20 * (1.0 - self.pullback_tolerance_pct))
            rejection_ok = (u_wick >= self.min_wick_ratio) or (c_close < c_open)

            if ema_aligned and pullback_ok and rejection_ok:
                sl_dist = self.sl_atr_mult * c_atr
                if sl_dist <= 0:
                    return None

                sl_price = c_close + sl_dist
                tp_price = c_close - (sl_dist * self.rr_ratio)

                return {
                    "action": "ENTER_SHORT",
                    "strategy": "MODEL_B_1to15",
                    "setup_name": "MARKOV_15M_BEAR_PULLBACK",
                    "timestamp": c_time,
                    "datetime": dt_str,
                    "direction": "SHORT",
                    "dir": "SHORT",
                    "entry_price": c_close,
                    "entry": c_close,
                    "stop_loss": sl_price,
                    "sl": sl_price,
                    "stop_dist": sl_dist,
                    "tp_price": tp_price,
                    "tp": tp_price,
                    "tp1_price": tp_price,
                    "tp2_price": tp_price,
                    "rr_ratio": self.rr_ratio,
                    "sl_atr_mult": self.sl_atr_mult,
                    "atr": c_atr,
                    "max_bars": self.max_hold_bars,
                    "max_hold_bars": self.max_hold_bars,
                    "rvol": c_rvol,
                    "daily_return": regime_info["daily_return"],
                }

        return None


ModelBMarkovEngine = MarkovScalpingEngine
