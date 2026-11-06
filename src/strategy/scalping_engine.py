from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..indicators.price_action_patterns import PriceActionPatterns
from ..indicators.trend_engine import TrendEngine
from ..indicators.key_levels_sr import KeyLevelsEngine
from ..indicators.fibonacci_retracements import FibonacciRetracementEngine


class ScalpingEngine:

    def __init__(
        self,
        min_rvol: float = 1.60,
        min_wick_ratio: float = 0.25,
        tp1_r: float = 1.60,
        tp2_r: float = 3.00,
        tp1_share: float = 0.50,
        be_trigger_r: float = 1.10,
        be_offset_r: float = 0.15,
        trail_trigger_r: float = 2.30,
        trail_lock_r: float = 1.00,
        sl_atr_buf: float = 0.20,
        risk_pct: float = 0.050,
        require_trend_align: bool = True,
        filter_counter_trend_shorts: bool = True,
        enable_fib_gp: bool = True,
        enable_short_rejections: bool = True
    ):
        self.min_rvol = min_rvol
        self.min_wick_ratio = min_wick_ratio
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.tp1_share = tp1_share
        self.be_trigger_r = be_trigger_r
        self.be_offset_r = be_offset_r
        self.trail_trigger_r = trail_trigger_r
        self.trail_lock_r = trail_lock_r
        self.sl_atr_buf = sl_atr_buf
        self.risk_pct = risk_pct
        self.require_trend_align = require_trend_align
        self.filter_counter_trend_shorts = filter_counter_trend_shorts
        self.enable_fib_gp = enable_fib_gp
        self.enable_short_rejections = enable_short_rejections

    def generate_monthly_signals(self, month_data: Dict[str, pd.DataFrame]) -> List[dict]:
        df_1d_full = month_data.get("full_1d", month_data["1d"])
        df_4h_full = month_data.get("full_4h", month_data["4h"])
        df_15m = month_data["15m"].copy()
        df_5m = month_data["5m"].copy()

        if len(df_15m) < 20 or len(df_5m) < 50:
            return []

        pit_daily = KeyLevelsEngine.extract_point_in_time_pdh_pdl(df_1d_full)

        ts_15m = df_15m["timestamp"].values
        dt_15m = df_15m["datetime"].values
        highs_15m = df_15m["high"].values
        lows_15m = df_15m["low"].values
        closes_15m = df_15m["close"].values
        rvols_15m = df_15m["rvol"].values
        u_wicks_15m = df_15m["upper_wick_pct"].values
        l_wicks_15m = df_15m["lower_wick_pct"].values
        has_bull_pa = df_15m["has_bullish_pa"].values
        has_bear_pa = df_15m["has_bearish_pa"].values
        ema50_15m = df_15m["ema50"].values

        ts_4h = df_4h_full["timestamp"].values
        closes_4h = df_4h_full["close"].values
        ema50_4h = df_4h_full["ema50"].values

        ts_1d = df_1d_full["timestamp"].values
        closes_1d = df_1d_full["close"].values
        ema50_1d = df_1d_full["ema50"].values

        ts_5m = df_5m["timestamp"].values
        dt_5m = df_5m["datetime"].values
        opens_5m = df_5m["open"].values
        highs_5m = df_5m["high"].values
        lows_5m = df_5m["low"].values
        closes_5m = df_5m["close"].values
        atrs_5m = df_5m["atr14"].values

        n_15m = len(df_15m)
        n_5m = len(df_5m)

        sweeps = []

        for i in range(20, n_15m):
            t_open = int(ts_15m[i])
            t_close = t_open + (15 * 60 * 1000) - 1
            c_rvol = rvols_15m[i]

            if c_rvol < self.min_rvol:
                continue

            c_high, c_low, c_close = highs_15m[i], lows_15m[i], closes_15m[i]

            idx_4h = np.searchsorted(ts_4h, t_close) - 1
            idx_4h = max(0, min(idx_4h, len(ts_4h) - 1))
            trend_4h_bull = closes_4h[idx_4h] >= ema50_4h[idx_4h]
            trend_4h_bear = closes_4h[idx_4h] <= ema50_4h[idx_4h]

            idx_1d = np.searchsorted(ts_1d, t_close) - 1
            idx_1d = max(0, min(idx_1d, len(ts_1d) - 1))
            trend_1d_bear = closes_1d[idx_1d] <= ema50_1d[idx_1d]

            pdh, pdl, pdc = KeyLevelsEngine.get_pdh_pdl_at_time(pit_daily, t_close)

            if c_low < pdl and c_close > pdl and (not self.require_trend_align or trend_4h_bull):
                if l_wicks_15m[i] >= self.min_wick_ratio or has_bull_pa[i]:
                    sweeps.append({
                        "type": "LONG",
                        "close_time": t_close,
                        "dt": dt_15m[i],
                        "extreme": c_low,
                        "setup_name": "PDL_SWEEP",
                        "rvol": c_rvol
                    })
                    continue

            if trend_4h_bull and c_low <= ema50_15m[i] and c_close > ema50_15m[i]:
                if l_wicks_15m[i] >= (self.min_wick_ratio * 1.1) or has_bull_pa[i]:
                    sweeps.append({
                        "type": "LONG",
                        "close_time": t_close,
                        "dt": dt_15m[i],
                        "extreme": c_low,
                        "setup_name": "EMA50_BOUNCE",
                        "rvol": c_rvol
                    })
                    continue

            if self.enable_fib_gp and trend_4h_bull:
                sw_low = np.min(lows_15m[i - 20 : i])
                sw_high = np.max(highs_15m[i - 20 : i])
                if (sw_high - sw_low) > (c_close * 0.008):
                    fibs = FibonacciRetracementEngine.calculate_bullish_fibs(sw_low, sw_high)
                    if fibs["golden_pocket_bottom"] <= c_low <= fibs["golden_pocket_top"] and c_close > fibs["golden_pocket_bottom"]:
                        if l_wicks_15m[i] >= self.min_wick_ratio or has_bull_pa[i]:
                            sweeps.append({
                                "type": "LONG",
                                "close_time": t_close,
                                "dt": dt_15m[i],
                                "extreme": c_low,
                                "setup_name": "FIB_GP_LONG",
                                "rvol": c_rvol
                            })
                            continue

            if self.enable_short_rejections and (trend_4h_bear and trend_1d_bear) and c_high >= ema50_15m[i] and c_close < ema50_15m[i]:
                if u_wicks_15m[i] >= (self.min_wick_ratio * 1.05) or has_bear_pa[i]:
                    sweeps.append({
                        "type": "SHORT",
                        "close_time": t_close,
                        "dt": dt_15m[i],
                        "extreme": c_high,
                        "setup_name": "EMA50_REJECTION",
                        "rvol": c_rvol
                    })
                    continue

        candidates = []
        for sw in sweeps:
            idx_start = int(np.searchsorted(ts_5m, sw["close_time"]))
            if idx_start >= n_5m:
                continue

            max_bars = min(12, n_5m - idx_start - 1)

            if sw["type"] == "LONG":
                local_high = highs_5m[idx_start]
                for off in range(1, max_bars):
                    curr = idx_start + off
                    if lows_5m[curr] <= sw["extreme"]:
                        break
                    if closes_5m[curr] > local_high:
                        exec_idx = curr + 1
                        if exec_idx >= n_5m:
                            break
                        entry_p = float(opens_5m[exec_idx])
                        atr_val = float(atrs_5m[curr]) if not np.isnan(atrs_5m[curr]) else float(entry_p * 0.005)
                        sl_p = float(sw["extreme"] - (self.sl_atr_buf * atr_val))
                        stop_dist = entry_p - sl_p
                        if stop_dist > 0:
                            candidates.append({
                                "exec_idx": exec_idx,
                                "exec_time": int(ts_5m[exec_idx]),
                                "dt": str(dt_5m[exec_idx]),
                                "dir": "LONG",
                                "entry": entry_p,
                                "sl": sl_p,
                                "stop_dist": stop_dist,
                                "setup_name": sw["setup_name"],
                                "rvol": sw["rvol"]
                            })
                        break
                    else:
                        local_high = max(local_high, highs_5m[curr])

            elif sw["type"] == "SHORT":
                local_low = lows_5m[idx_start]
                for off in range(1, max_bars):
                    curr = idx_start + off
                    if highs_5m[curr] >= sw["extreme"]:
                        break
                    if closes_5m[curr] < local_low:
                        exec_idx = curr + 1
                        if exec_idx >= n_5m:
                            break
                        entry_p = float(opens_5m[exec_idx])
                        atr_val = float(atrs_5m[curr]) if not np.isnan(atrs_5m[curr]) else float(entry_p * 0.005)
                        sl_p = float(sw["extreme"] + (self.sl_atr_buf * atr_val))
                        stop_dist = sl_p - entry_p
                        if stop_dist > 0:
                            candidates.append({
                                "exec_idx": exec_idx,
                                "exec_time": int(ts_5m[exec_idx]),
                                "dt": str(dt_5m[exec_idx]),
                                "dir": "SHORT",
                                "entry": entry_p,
                                "sl": sl_p,
                                "stop_dist": stop_dist,
                                "setup_name": sw["setup_name"],
                                "rvol": sw["rvol"]
                            })
                        break
                    else:
                        local_low = min(local_low, lows_5m[curr])

        candidates.sort(key=lambda c: c["exec_time"])
        return candidates
