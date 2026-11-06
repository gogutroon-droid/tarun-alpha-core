import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PriceActionPatterns:

    @staticmethod
    def compute_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        total_range = np.maximum(highs - lows, 1e-8)
        body = np.abs(closes - opens)
        body_pct = body / total_range

        upper_wick = highs - np.maximum(opens, closes)
        lower_wick = np.minimum(opens, closes) - lows

        upper_wick_pct = upper_wick / total_range
        lower_wick_pct = lower_wick / total_range

        is_bullish = closes >= opens
        is_bearish = closes < opens

        df["total_range"] = total_range
        df["body"] = body
        df["body_pct"] = body_pct
        df["upper_wick"] = upper_wick
        df["lower_wick"] = lower_wick
        df["upper_wick_pct"] = upper_wick_pct
        df["lower_wick_pct"] = lower_wick_pct
        df["is_bullish"] = is_bullish
        df["is_bearish"] = is_bearish

        if "volume" in df.columns:
            vol_sma = df["volume"].rolling(window=20, min_periods=1).mean()
            df["rvol"] = df["volume"] / np.maximum(vol_sma, 1e-8)
        else:
            df["rvol"] = 1.0

        return df

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        df = PriceActionPatterns.compute_candle_metrics(df)
        n = len(df)

        bullish_pin = np.zeros(n, dtype=bool)
        bearish_pin = np.zeros(n, dtype=bool)
        bullish_engulfing = np.zeros(n, dtype=bool)
        bearish_engulfing = np.zeros(n, dtype=bool)
        morning_star = np.zeros(n, dtype=bool)
        evening_star = np.zeros(n, dtype=bool)

        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        body_pct = df["body_pct"].values
        upper_wick_pct = df["upper_wick_pct"].values
        lower_wick_pct = df["lower_wick_pct"].values
        is_bull = df["is_bullish"].values
        is_bear = df["is_bearish"].values

        for i in range(2, n):
            if lower_wick_pct[i] >= 0.50 and upper_wick_pct[i] <= 0.25 and body_pct[i] <= 0.35:
                bullish_pin[i] = True

            if upper_wick_pct[i] >= 0.50 and lower_wick_pct[i] <= 0.25 and body_pct[i] <= 0.35:
                bearish_pin[i] = True

            if is_bull[i] and is_bear[i - 1]:
                if closes[i] >= opens[i - 1] and opens[i] <= closes[i - 1] and body_pct[i] >= 0.50:
                    bullish_engulfing[i] = True

            if is_bear[i] and is_bull[i - 1]:
                if closes[i] <= opens[i - 1] and opens[i] >= closes[i - 1] and body_pct[i] >= 0.50:
                    bearish_engulfing[i] = True

            if is_bear[i - 2] and body_pct[i - 2] >= 0.40:
                if body_pct[i - 1] <= 0.30:
                    if is_bull[i] and closes[i] > (opens[i - 2] + closes[i - 2]) / 2.0:
                        morning_star[i] = True

            if is_bull[i - 2] and body_pct[i - 2] >= 0.40:
                if body_pct[i - 1] <= 0.30:
                    if is_bear[i] and closes[i] < (opens[i - 2] + closes[i - 2]) / 2.0:
                        evening_star[i] = True

        df["pattern_bullish_pin"] = bullish_pin
        df["pattern_bearish_pin"] = bearish_pin
        df["pattern_bullish_engulfing"] = bullish_engulfing
        df["pattern_bearish_engulfing"] = bearish_engulfing
        df["pattern_morning_star"] = morning_star
        df["pattern_evening_star"] = evening_star

        df["has_bullish_pa"] = bullish_pin | bullish_engulfing | morning_star | (df["lower_wick_pct"] >= 0.40)
        df["has_bearish_pa"] = bearish_pin | bearish_engulfing | evening_star | (df["upper_wick_pct"] >= 0.40)

        return df
