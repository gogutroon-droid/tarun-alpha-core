import os
import sys
import time
import json
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.execution.markov_simulator import MarkovSimulator

RESULTS_DIR = os.path.join(CURRENT_DIR, "backtest_results")


def locate_data_file(filename: str) -> str:
    path = os.path.join(CURRENT_DIR, "data", filename)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Dataset not found: {filename}")


def run_image2_master_backtests():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("================================================================================")
    print("      MODEL B MARKOV 1:15.0 R:R MICRO-SCALPER 10-YEAR AUDIT")
    print("      DYNAMIC INTRA-MONTH COMPOUNDING | MONTHLY PROFIT RESET | 24/7 UNCONSTRAINED")
    print("================================================================================")

    print("\n[1/4] Loading 10-year historical candles (1D and 15M)...")
    path_1d = locate_data_file("BTCUSDT_1d.csv")
    path_15m = locate_data_file("BTCUSDT_15m.csv")

    df_1d = pd.read_csv(path_1d)
    df_15m = pd.read_csv(path_15m)

    df_15m["datetime"] = pd.to_datetime(df_15m["datetime"], format="mixed")
    df_15m["year"] = df_15m["datetime"].dt.year
    df_15m["month"] = df_15m["datetime"].dt.month
    df_15m["year_month"] = df_15m["datetime"].dt.strftime("%Y-%m")

    print(f"Loaded: 1D ({len(df_1d):,} bars) | 15M ({len(df_15m):,} bars)")
    print(f"Historical Horizon: {df_15m['datetime'].iloc[0]} to {df_15m['datetime'].iloc[-1]}")

    capitals = [2000, 5000, 10000, 20000]
    all_capital_annual = []

    print("\n[2/4] Running 10-Year Simulation across all 4 Capital Baselines...")

    all_monthly_dfs = {}
    master_trade_log = []

    for cap in capitals:
        cap_str = f"{cap // 1000}k"
        print(f"\n---> Simulating Baseline ${cap:,} ({cap_str})...")

        sim = MarkovSimulator(initial_capital=float(cap), risk_pct=0.030)
        monthly_records = []
        cap_trades = []

        for ym, df_m in df_15m.groupby("year_month"):
            df_m_clean = df_m.copy().reset_index(drop=True)
            metrics, tdf, _ = sim.run_backtest(df_1d=df_1d, df_15m=df_m_clean, starting_equity=float(cap))

            yr = int(ym.split("-")[0])
            mo = int(ym.split("-")[1])

            if not tdf.empty:
                tdf["year"] = yr
                tdf["month"] = mo
                tdf["year_month"] = ym
                tdf["baseline_capital"] = cap
                tdf["pnl_r"] = tdf["net_pnl"] / (float(cap) * 0.030)
                cap_trades.append(tdf)

            monthly_records.append({
                "year": yr,
                "month": mo,
                "year_month": ym,
                "initial_capital": cap,
                "start_equity": float(cap),
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
        df_monthly.to_csv(os.path.join(RESULTS_DIR, f"monthly_ledger_{cap_str}.csv"), index=False)
        all_monthly_dfs[cap] = df_monthly

        if cap == 10000 and cap_trades:
            master_trade_log = pd.concat(cap_trades, ignore_index=True)
            master_trade_log["exit_reason"] = master_trade_log["outcome"]
            master_trade_log.to_csv(os.path.join(RESULTS_DIR, "trade_log.csv"), index=False)

        years = sorted(df_monthly["year"].unique())
        ann_recs = []
        for y in years:
            sub_m = df_monthly[df_monthly["year"] == y]
            net_yr = sub_m["net_profit"].sum()
            t_cnt = sub_m["trades"].sum()
            w_cnt = sub_m["wins"].sum()
            tp_cnt = sub_m["tp_hits"].sum()
            win_r = (w_cnt / t_cnt * 100.0) if t_cnt > 0 else 0.0
            gross_w = sub_m[sub_m["net_profit"] > 0]["net_profit"].sum()
            gross_l = abs(sub_m[sub_m["net_profit"] < 0]["net_profit"].sum())
            pf_yr = (gross_w / gross_l) if gross_l > 0 else 99.0
            max_dd_yr = sub_m["max_dd_pct"].min()

            ann_recs.append({
                "year": y,
                "initial_capital": cap,
                "end_equity": cap + net_yr,
                "net_profit": net_yr,
                "roi_pct": (net_yr / cap) * 100.0,
                "trades": t_cnt,
                "tp_hits": tp_cnt,
                "win_rate_pct": win_r,
                "profit_factor": pf_yr,
                "max_dd_pct": max_dd_yr,
            })

        df_ann = pd.DataFrame(ann_recs)
        all_capital_annual.append(df_ann)
        tot_prof = df_ann["net_profit"].sum()
        print(f"  [+] 10-Year Cumulative Cash Extracted for ${cap:,}: ${tot_prof:,.2f} | Avg Annual ROI: {df_ann['roi_pct'].mean():.1f}%")

    df_all_ann = pd.concat(all_capital_annual, ignore_index=True)
    df_all_ann.to_csv(os.path.join(RESULTS_DIR, "annual_summary_all_capitals.csv"), index=False)

    print("\n[3/4] Generating Native INR Ledger...")
    df_10k_m = all_monthly_dfs[10000].copy()
    df_inr_m = df_10k_m.copy()
    df_inr_m.to_csv(os.path.join(RESULTS_DIR, "monthly_ledger_inr_10k.csv"), index=False)
    print(f"  [+] Total 10-Year Extracted Net Cash in INR: Rs {df_inr_m['net_profit'].sum():+,.2f}")

    print("\n[4/4] 2026 Verification Audit:")
    df_2026 = df_10k_m[df_10k_m["year"] == 2026]
    print(f"  • 2026 Total Trades:   {df_2026['trades'].sum()}")
    print(f"  • 2026 Total ROI:      +{df_2026['roi_pct'].sum():.2f}%")
    print(f"  • 2026 Net PnL ($10k): +${df_2026['net_profit'].sum():,.2f}")

    print(f"\nCompleted Master 10-Year Simulations in {time.time() - t0:.2f} seconds!")
    return df_all_ann


if __name__ == "__main__":
    run_image2_master_backtests()
