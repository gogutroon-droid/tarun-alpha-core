import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .indicators.price_action_patterns import PriceActionPatterns
from .indicators.trend_engine import TrendEngine
from .indicators.key_levels_sr import KeyLevelsEngine


class DataLoader:

    MONTH_NAMES = [
        "January", "February", "March", "April", "May",
        "June", "July", "August", "September", "October",
        "November", "December",
    ]

    def __init__(self, data_dir: str = "data", year: int = 2026):
        self.data_dir = data_dir
        self.year = year

    def find_data_file(self, tf: str) -> str:
        candidates = [
            os.path.join(self.data_dir, f"BTCUSDT_{tf}.csv"),
            os.path.join("data", f"BTCUSDT_{tf}.csv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", f"BTCUSDT_{tf}.csv"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", f"BTCUSDT_{tf}.csv"),
            os.path.join("..", "data", f"BTCUSDT_{tf}.csv"),
            os.path.join("..", "..", "rebtcusdt", "data", f"BTCUSDT_{tf}.csv"),
            os.path.join("..", "rebtcusdt", "data", f"BTCUSDT_{tf}.csv"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Could not locate dataset for timeframe: {tf}")

    def load_full_enriched_data(self) -> Dict[str, pd.DataFrame]:
        raw_dfs = {}
        for tf in ["1d", "4h", "15m", "5m"]:
            path = self.find_data_file(tf)
            df = pd.read_csv(path)
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            raw_dfs[tf] = df

        df_1d = TrendEngine.calculate_trend_indicators(raw_dfs["1d"])
        df_1d = PriceActionPatterns.compute_candle_metrics(df_1d)

        df_4h = TrendEngine.calculate_trend_indicators(raw_dfs["4h"])
        df_4h = PriceActionPatterns.compute_candle_metrics(df_4h)

        df_15m = TrendEngine.calculate_trend_indicators(raw_dfs["15m"])
        df_15m = PriceActionPatterns.detect_patterns(df_15m)

        df_5m = PriceActionPatterns.detect_patterns(raw_dfs["5m"])
        tr_5m = np.maximum(
            df_5m["high"] - df_5m["low"],
            np.maximum(
                np.abs(df_5m["high"] - df_5m["close"].shift(1)),
                np.abs(df_5m["low"] - df_5m["close"].shift(1)),
            ),
        )
        df_5m["atr14"] = tr_5m.rolling(window=14, min_periods=1).mean()

        for tf, df_enriched in [("1d", df_1d), ("4h", df_4h), ("15m", df_15m), ("5m", df_5m)]:
            df_enriched["year"] = df_enriched["datetime"].dt.year
            df_enriched["month"] = df_enriched["datetime"].dt.month
            df_enriched["month_name"] = df_enriched["datetime"].dt.month_name()

        return {
            "1d": df_1d,
            "4h": df_4h,
            "15m": df_15m,
            "5m": df_5m,
            "full_1d": df_1d,
            "full_4h": df_4h,
        }

    def load_all_2026_data(self) -> Dict[str, pd.DataFrame]:
        return self.load_year_data(self.year)

    def load_year_data(self, year: int) -> Dict[str, pd.DataFrame]:
        full_data = self.load_full_enriched_data()
        datasets = {}
        for tf in ["1d", "4h", "15m", "5m"]:
            df_enriched = full_data[tf]
            df_year = df_enriched[df_enriched["year"] == year].copy().reset_index(drop=True)
            datasets[tf] = df_year

        datasets["full_1d"] = full_data["1d"]
        datasets["full_4h"] = full_data["4h"]
        return datasets

    def partition_by_month(self, datasets: Dict[str, pd.DataFrame]) -> Dict[int, Dict[str, pd.DataFrame]]:
        monthly_data = {}
        months = sorted(datasets["5m"]["month"].unique())

        for m in months:
            monthly_data[m] = {
                "month_num": m,
                "month_name": self.MONTH_NAMES[m - 1] if 1 <= m <= len(self.MONTH_NAMES) else f"Month_{m}",
                "1d": datasets["1d"][datasets["1d"]["month"] == m].copy().reset_index(drop=True),
                "4h": datasets["4h"][datasets["4h"]["month"] == m].copy().reset_index(drop=True),
                "15m": datasets["15m"][datasets["15m"]["month"] == m].copy().reset_index(drop=True),
                "5m": datasets["5m"][datasets["5m"]["month"] == m].copy().reset_index(drop=True),
                "full_1d": datasets.get("full_1d", datasets["1d"]),
                "full_4h": datasets.get("full_4h", datasets["4h"]),
            }

        return monthly_data

    def partition_by_year(self, full_datasets: Dict[str, pd.DataFrame]) -> Dict[int, Dict[str, pd.DataFrame]]:
        years = sorted(full_datasets["5m"]["year"].unique())
        yearly_data = {}
        for y in years:
            yearly_data[y] = {
                "1d": full_datasets["1d"][full_datasets["1d"]["year"] == y].copy().reset_index(drop=True),
                "4h": full_datasets["4h"][full_datasets["4h"]["year"] == y].copy().reset_index(drop=True),
                "15m": full_datasets["15m"][full_datasets["15m"]["year"] == y].copy().reset_index(drop=True),
                "5m": full_datasets["5m"][full_datasets["5m"]["year"] == y].copy().reset_index(drop=True),
                "full_1d": full_datasets["1d"],
                "full_4h": full_datasets["4h"],
            }
        return yearly_data
