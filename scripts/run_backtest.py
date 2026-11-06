import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.execution.markov_simulator import MarkovSimulator
from src.data_loader import DataLoader

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backtest_results"))


def locate_data_file(filename: str) -> str:
    search_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "data"),
        os.path.join(os.path.dirname(__file__), "..", "..", "data"),
        os.path.join(os.path.dirname(__file__), "..", "..", "rebtcusdt", "data"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "rebtcusdt", "data"),
    ]
    for d in search_dirs:
        candidate = os.path.join(d, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Dataset not found: {filename}")


def run_master_backtests(capital: float = 10000.0, risk_pct: float = 0.030):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 80)
    print("MODEL B MARKOV 1:15.0 R:R MICRO-SCALPER AUDIT")
    print("DYNAMIC INTRA-MONTH COMPOUNDING | MONTHLY RESET | 24/7 EXECUTION")
    print("=" * 80)

    try:
        path_1d = locate_data_file("BTCUSDT_1d.csv")
        path_15m = locate_data_file("BTCUSDT_15m.csv")
        df_1d = pd.read_csv(path_1d)
        df_15m = pd.read_csv(path_15m)
    except FileNotFoundError as err:
        print(f"[!] Warning: {err}")
        print("[!] Generating synthetic demonstration candles for pipeline verification...")
        dates_1d = pd.date_range("2024-01-01", periods=1000, freq="1D")
        df_1d = pd.DataFrame({
            "timestamp": [int(d.timestamp() * 1000) for d in dates_1d],
            "datetime": dates_1d.astype(str),
            "open": np.linspace(40000, 65000, len(dates_1d)),
            "high": np.linspace(40500, 65500, len(dates_1d)),
            "low": np.linspace(39500, 64500, len(dates_1d)),
            "close": np.linspace(40200, 65200, len(dates_1d)),
            "volume": np.random.uniform(500, 2500, len(dates_1d)),
        })
        dates_15m = pd.date_range("2026-01-01", periods=2000, freq="15min")
        df_15m = pd.DataFrame({
            "timestamp": [int(d.timestamp() * 1000) for d in dates_15m],
            "datetime": dates_15m.astype(str),
            "open": np.linspace(42000, 48000, len(dates_15m)),
            "high": np.linspace(42200, 48200, len(dates_15m)),
            "low": np.linspace(41800, 47800, len(dates_15m)),
            "close": np.linspace(42100, 48100, len(dates_15m)),
            "volume": np.random.uniform(50, 200, len(dates_15m)),
        })

    df_15m["datetime"] = pd.to_datetime(df_15m["datetime"], format="mixed")
    df_15m["year"] = df_15m["datetime"].dt.year
    df_15m["month"] = df_15m["datetime"].dt.month
    df_15m["year_month"] = df_15m["datetime"].dt.strftime("%Y-%m")

    print(f"Data Summary: 1D ({len(df_1d):,} bars) | 15M ({len(df_15m):,} bars)")
    print(f"Horizon: {df_15m['datetime'].iloc[0]} to {df_15m['datetime'].iloc[-1]}")

    sim = MarkovSimulator(initial_capital=capital, risk_pct=risk_pct)
    monthly_records = []
    cap_trades = []

    for ym, df_m in df_15m.groupby("year_month"):
        df_m_clean = df_m.copy().reset_index(drop=True)
        metrics, tdf, _ = sim.run_backtest(df_1d=df_1d, df_15m=df_m_clean, starting_equity=capital)

        yr = int(ym.split("-")[0])
        mo = int(ym.split("-")[1])

        if not tdf.empty:
            tdf["year"] = yr
            tdf["month"] = mo
            tdf["year_month"] = ym
            tdf["baseline_capital"] = capital
            tdf["pnl_r"] = tdf["net_pnl"] / (capital * risk_pct)
            cap_trades.append(tdf)

        monthly_records.append({
            "year": yr,
            "month": mo,
            "year_month": ym,
            "initial_capital": capital,
            "start_equity": capital,
            "end_equity": metrics["final_equity"],
            "net_profit": metrics["net_pnl"],
            "roi_pct": metrics["return_pct"],
            "trades": metrics["total_trades"],
            "wins": metrics["wins"],
            "losses": metrics["losses"],
            "tp_hits": metrics["tp_hits"],
            "timeout_exits": metrics["timeouts"],
            "win_rate_pct": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "max_dd_pct": -metrics["max_drawdown"],
        })

    df_monthly = pd.DataFrame(monthly_records)
    df_monthly.to_csv(os.path.join(RESULTS_DIR, "monthly_ledger.csv"), index=False)

    if cap_trades:
        master_trade_log = pd.concat(cap_trades, ignore_index=True)
        master_trade_log.to_csv(os.path.join(RESULTS_DIR, "trade_log.csv"), index=False)

    print("\nPerformance Summary:")
    print(f"  Total Trades:     {df_monthly['trades'].sum()}")
    print(f"  Total Net Profit: ${df_monthly['net_profit'].sum():,.2f}")
    print(f"  Avg Monthly ROI:  {df_monthly['roi_pct'].mean():.2f}%")
    print(f"  Elapsed:          {time.time() - t0:.2f}s")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Model B Markov Backtest Runner")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USD/INR")
    parser.add_argument("--risk", type=float, default=0.030, help="Risk per trade percentage")
    args = parser.parse_args()

    run_master_backtests(capital=args.capital, risk_pct=args.risk)


if __name__ == "__main__":
    main()
