from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class KeyLevelsEngine:

    @staticmethod
    def extract_point_in_time_pdh_pdl(df_1d: pd.DataFrame) -> Dict[str, np.ndarray]:
        ts = df_1d["timestamp"].values
        highs = df_1d["high"].values
        lows = df_1d["low"].values
        closes = df_1d["close"].values

        return {
            "ts": ts,
            "highs": highs,
            "lows": lows,
            "closes": closes,
        }

    @staticmethod
    def get_pdh_pdl_at_time(pit_dict: Dict[str, np.ndarray], current_ts: int) -> Tuple[float, float, float]:
        ts = pit_dict["ts"]
        idx = np.searchsorted(ts, current_ts) - 1
        if idx > 0:
            prev_idx = idx - 1 if current_ts >= ts[idx] else idx - 2
            prev_idx = max(0, prev_idx)
        else:
            prev_idx = 0
        return pit_dict["highs"][prev_idx], pit_dict["lows"][prev_idx], pit_dict["closes"][prev_idx]

    @staticmethod
    def find_major_sr_pivots(df: pd.DataFrame, left: int = 4, right: int = 4) -> Tuple[List[dict], List[dict]]:
        highs = df["high"].values
        lows = df["low"].values
        ts = df["timestamp"].values
        n = len(df)

        support_levels = []
        resistance_levels = []

        for i in range(left, n - right):
            if highs[i] == np.max(highs[i - left : i + right + 1]):
                confirmation_time = int(ts[i + right])
                resistance_levels.append({
                    "price": float(highs[i]),
                    "pivot_time": int(ts[i]),
                    "confirmation_time": confirmation_time,
                    "type": "RESISTANCE"
                })

            if lows[i] == np.min(lows[i - left : i + right + 1]):
                confirmation_time = int(ts[i + right])
                support_levels.append({
                    "price": float(lows[i]),
                    "pivot_time": int(ts[i]),
                    "confirmation_time": confirmation_time,
                    "type": "SUPPORT"
                })

        return support_levels, resistance_levels
