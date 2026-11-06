import os
import io
import time
import zipfile
import datetime
import concurrent.futures
from typing import List, Optional
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
TIMEFRAMES = ["1d", "4h", "1h", "15m", "5m", "3m"]

BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT"


def generate_month_list(start_year=2017, start_month=8, end_year=2026, end_month=9) -> List[str]:
    months = []
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def download_binance_vision_month(tf: str, ym: str) -> Optional[pd.DataFrame]:
    url = f"{BINANCE_VISION_BASE}/{tf}/BTCUSDT-{tf}-{ym}.zip"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                csv_name = z.namelist()[0]
                df = pd.read_csv(z.open(csv_name), header=None)
                if isinstance(df.iloc[0, 0], str) and not df.iloc[0, 0].replace(".", "").isdigit():
                    df = df.iloc[1:].copy()
                df = df.iloc[:, :6].copy()
                df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                mask_micro = df["timestamp"] > 1e14
                if mask_micro.any():
                    df.loc[mask_micro, "timestamp"] = df.loc[mask_micro, "timestamp"] / 1000.0

                df["timestamp"] = df["timestamp"].astype(np.int64)
                return df
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None


def fetch_binance_vision_all(tf: str, max_workers: int = 12) -> pd.DataFrame:
    months = generate_month_list(2017, 8, 2026, 9)
    print(f"[{tf}] Downloading {len(months)} monthly archives from Binance Vision...")
    dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda ym: download_binance_vision_month(tf, ym), months))

    for res in results:
        if res is not None and not res.empty:
            dfs.append(res)

    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return full_df
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def synthesize_sub_candles_from_15m(df_15m_slice: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
    rows = []
    ratio = 15 // target_minutes
    sub_ms = target_minutes * 60 * 1000
    for _, row in df_15m_slice.iterrows():
        t = int(row["timestamp"])
        o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]
        sub_vol = v / ratio
        if ratio == 3:
            if c >= o:
                rows.append([t, o, max(o, (o + h) / 2), min(o, (o + l) / 2), (o + h) / 2, sub_vol])
                rows.append([t + sub_ms, (o + h) / 2, h, l, (l + c) / 2, sub_vol])
                rows.append([t + 2 * sub_ms, (l + c) / 2, max(c, (l + c) / 2), min(c, (l + c) / 2), c, sub_vol])
            else:
                rows.append([t, o, max(o, (o + h) / 2), min(o, (o + l) / 2), (o + l) / 2, sub_vol])
                rows.append([t + sub_ms, (o + l) / 2, h, l, (h + c) / 2, sub_vol])
                rows.append([t + 2 * sub_ms, (h + c) / 2, max(c, (h + c) / 2), min(c, (h + c) / 2), c, sub_vol])
        elif ratio == 5:
            step_price = (c - o) / 5.0
            for i in range(5):
                cur_t = t + i * sub_ms
                sub_o = o + i * step_price
                sub_c = o + (i + 1) * step_price
                sub_h = h if i == 2 else max(sub_o, sub_c)
                sub_l = l if i == 1 else min(sub_o, sub_c)
                rows.append([cur_t, sub_o, sub_h, sub_l, sub_c, sub_vol])
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def load_candles(tf: str) -> pd.DataFrame:
    candidates = [
        os.path.join(DATA_DIR, f"BTCUSDT_{tf}.parquet"),
        os.path.join(DATA_DIR, f"BTCUSDT_{tf}.csv"),
        os.path.join("data", f"BTCUSDT_{tf}.parquet"),
        os.path.join("data", f"BTCUSDT_{tf}.csv"),
        os.path.join("..", "data", f"BTCUSDT_{tf}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            if p.endswith(".parquet"):
                return pd.read_parquet(p)
            return pd.read_csv(p)
    raise FileNotFoundError(f"No cached data found for timeframe {tf}")
