from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class MonthlyMetricsEngine:

    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        if trades_df.empty:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "win_loss_ratio": 0.0,
                "avg_bars_held": 0.0,
                "long_trades": 0,
                "short_trades": 0,
                "final_equity": initial_capital
            }

        df = trades_df.copy()
        if "net_pnl" not in df.columns and "net_pnl_inr" in df.columns:
            df["net_pnl"] = df["net_pnl_inr"]
        if "gross_pnl" not in df.columns and "gross_pnl_inr" in df.columns:
            df["gross_pnl"] = df["gross_pnl_inr"]
        if "equity" not in df.columns and "equity_inr" in df.columns:
            df["equity"] = df["equity_inr"]

        n_trades = len(df)
        winners = df[df["net_pnl"] > 0]
        losers = df[df["net_pnl"] <= 0]

        win_rate = (len(winners) / n_trades) * 100.0 if n_trades > 0 else 0.0
        tot_net_pnl = df["net_pnl"].sum()
        final_equity = initial_capital + tot_net_pnl
        ret_pct = (tot_net_pnl / initial_capital) * 100.0

        gross_win = winners["gross_pnl"].sum() if not winners.empty else 0.0
        gross_loss = abs(losers["gross_pnl"].sum()) if not losers.empty else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0

        eq_curve = [initial_capital] + list(df["equity"].values)
        eq_series = pd.Series(eq_curve)
        peak_series = eq_series.cummax()
        dd_series = (eq_series - peak_series) / peak_series * 100.0
        max_dd = abs(dd_series.min())

        avg_win = winners["net_pnl"].mean() if not winners.empty else 0.0
        avg_loss = abs(losers["net_pnl"].mean()) if not losers.empty else 0.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

        avg_bars_held = df["bars_held"].mean() if "bars_held" in df.columns else 0.0
        long_trades = len(df[df["direction"] == "LONG"])
        short_trades = len(df[df["direction"] == "SHORT"])

        r_returns = df["r_realized"].values if "r_realized" in df.columns else np.zeros(n_trades)
        sharpe = (np.mean(r_returns) / np.std(r_returns) * np.sqrt(252)) if len(r_returns) > 1 and np.std(r_returns) > 0 else 0.0

        return {
            "total_trades": n_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "net_pnl": tot_net_pnl,
            "return_pct": ret_pct,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "win_loss_ratio": win_loss_ratio,
            "avg_bars_held": avg_bars_held,
            "long_trades": long_trades,
            "short_trades": short_trades,
            "final_equity": final_equity
        }
