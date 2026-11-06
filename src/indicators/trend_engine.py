import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class TrendEngine:

    @staticmethod
    def calculate_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        closes = df["close"]

        df["ema20"] = closes.ewm(span=20, adjust=False).mean()
        df["ema50"] = closes.ewm(span=50, adjust=False).mean()
        df["sma100"] = closes.rolling(window=100, min_periods=20).mean()
        df["sma200"] = closes.rolling(window=200, min_periods=50).mean()

        df["sma100"] = df["sma100"].fillna(df["ema50"])
        df["sma200"] = df["sma200"].fillna(df["sma100"])

        bullish_bias = (df["close"] > df["ema50"]) & (df["ema20"] >= df["ema50"])
        bearish_bias = (df["close"] < df["ema50"]) & (df["ema20"] <= df["ema50"])

        df["trend_bias"] = np.where(bullish_bias, 1, np.where(bearish_bias, -1, 0))

        df["ema50_slope"] = (df["ema50"] - df["ema50"].shift(3)) / np.maximum(df["close"], 1e-8) * 100.0
        df["ema50_slope"] = df["ema50_slope"].fillna(0.0)

        return df

    @staticmethod
    def determine_market_structure(highs: np.ndarray, lows: np.ndarray, lookback: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        n = len(highs)
        is_swing_high = np.zeros(n, dtype=bool)
        is_swing_low = np.zeros(n, dtype=bool)

        for i in range(lookback, n - lookback):
            if highs[i] == np.max(highs[i - lookback : i + lookback + 1]):
                is_swing_high[i] = True
            if lows[i] == np.min(lows[i - lookback : i + lookback + 1]):
                is_swing_low[i] = True

        return is_swing_high, is_swing_low
