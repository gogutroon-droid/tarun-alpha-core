from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class MarkovSimulator:

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_pct: float = 0.030,
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
        maker_fee: float = 0.00020,
        taker_fee: float = 0.00030,
        slippage_bps: float = 1.0,
        currency: str = "INR",
    ):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
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
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_rate = slippage_bps / 10000.0
        self.currency = currency

    def run_backtest(
        self,
        df_1d: pd.DataFrame,
        df_15m: pd.DataFrame,
        starting_equity: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]]:
        equity = starting_equity if starting_equity is not None else self.initial_capital
        peak_equity = equity

        df_1d_sorted = df_1d.copy()
        df_1d_sorted["dt"] = pd.to_datetime(df_1d_sorted["datetime"])
        df_1d_sorted.sort_values("dt", inplace=True)
        df_1d_sorted["daily_return_10d"] = (df_1d_sorted["close"] - df_1d_sorted["close"].shift(self.markov_window)) / df_1d_sorted["close"].shift(self.markov_window)
        df_1d_sorted["is_bull"] = df_1d_sorted["daily_return_10d"] > self.markov_thresh
        df_1d_sorted["is_bear"] = df_1d_sorted["daily_return_10d"] < -self.markov_thresh

        df_15m_sorted = df_15m.copy()
        df_15m_sorted["dt"] = pd.to_datetime(df_15m_sorted["datetime"])
        df_15m_sorted.sort_values("dt", inplace=True)

        df = pd.merge_asof(
            df_15m_sorted,
            df_1d_sorted[["dt", "is_bull", "is_bear", "daily_return_10d"]],
            on="dt",
            direction="backward",
        )

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
        df["lower_wick"] = lower_wick / np.where(candle_range > 0, candle_range, 1.0)
        df["upper_wick"] = upper_wick / np.where(candle_range > 0, candle_range, 1.0)

        trades = []
        in_pos = False
        pos_dir = 0
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        entry_idx = 0
        trade_qty = 0.0
        notional_val = 0.0
        risk_budget = 0.0

        n_bars = len(df)
        start_idx = max(50, self.rvol_ma_len, self.ema_slow)

        for i in range(start_idx, n_bars):
            row = df.iloc[i]

            if in_pos:
                bars_held = i - entry_idx

                if pos_dir == 1:
                    if row["low"] <= sl_price:
                        exit_price = sl_price
                        gross_pnl = (exit_price - entry_price) * trade_qty
                        fee = (entry_price + exit_price) * trade_qty * ((self.maker_fee + self.taker_fee) / 2.0 + self.slippage_rate)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (equity - peak_equity) / peak_equity * 100.0

                        trades.append({
                            "entry_time": df.iloc[entry_idx]["datetime"],
                            "exit_time": row["datetime"],
                            "direction": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "outcome": "SL",
                            "reason": "SL_FULL (-1.0R Stop Loss)",
                            "gross_pnl": gross_pnl,
                            "friction": fee,
                            "net_pnl": net_pnl,
                            "equity": equity,
                            "drawdown_pct": dd,
                            "bars_held": bars_held,
                        })
                        in_pos = False

                    elif row["high"] >= tp_price:
                        exit_price = tp_price
                        gross_pnl = (exit_price - entry_price) * trade_qty
                        fee = (entry_price + exit_price) * trade_qty * ((self.maker_fee + self.taker_fee) / 2.0 + self.slippage_rate)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (equity - peak_equity) / peak_equity * 100.0

                        trades.append({
                            "entry_time": df.iloc[entry_idx]["datetime"],
                            "exit_time": row["datetime"],
                            "direction": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "outcome": "TP",
                            "reason": f"TP_FULL (+{self.rr_ratio:.1f}R Target)",
                            "gross_pnl": gross_pnl,
                            "friction": fee,
                            "net_pnl": net_pnl,
                            "equity": equity,
                            "drawdown_pct": dd,
                            "bars_held": bars_held,
                        })
                        in_pos = False

                    elif bars_held >= self.max_hold_bars:
                        exit_price = row["close"]
                        gross_pnl = (exit_price - entry_price) * trade_qty
                        fee = (entry_price + exit_price) * trade_qty * ((self.maker_fee + self.taker_fee) / 2.0 + self.slippage_rate)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (equity - peak_equity) / peak_equity * 100.0

                        trades.append({
                            "entry_time": df.iloc[entry_idx]["datetime"],
                            "exit_time": row["datetime"],
                            "direction": "LONG",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "outcome": "TIMEOUT",
                            "reason": f"TIMEOUT_EXIT ({bars_held} Bars / 30H)",
                            "gross_pnl": gross_pnl,
                            "friction": fee,
                            "net_pnl": net_pnl,
                            "equity": equity,
                            "drawdown_pct": dd,
                            "bars_held": bars_held,
                        })
                        in_pos = False

                elif pos_dir == -1:
                    if row["high"] >= sl_price:
                        exit_price = sl_price
                        gross_pnl = (entry_price - exit_price) * trade_qty
                        fee = (entry_price + exit_price) * trade_qty * ((self.maker_fee + self.taker_fee) / 2.0 + self.slippage_rate)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (equity - peak_equity) / peak_equity * 100.0

                        trades.append({
                            "entry_time": df.iloc[entry_idx]["datetime"],
                            "exit_time": row["datetime"],
                            "direction": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "outcome": "SL",
                            "reason": "SL_FULL (-1.0R Stop Loss)",
                            "gross_pnl": gross_pnl,
                            "friction": fee,
                            "net_pnl": net_pnl,
                            "equity": equity,
                            "drawdown_pct": dd,
                            "bars_held": bars_held,
                        })
                        in_pos = False

                    elif row["low"] <= tp_price:
                        exit_price = tp_price
                        gross_pnl = (entry_price - exit_price) * trade_qty
                        fee = (entry_price + exit_price) * trade_qty * ((self.maker_fee + self.taker_fee) / 2.0 + self.slippage_rate)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (equity - peak_equity) / peak_equity * 100.0

                        trades.append({
                            "entry_time": df.iloc[entry_idx]["datetime"],
                            "exit_time": row["datetime"],
                            "direction": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "outcome": "TP",
                            "reason": f"TP_FULL (+{self.rr_ratio:.1f}R Target)",
                            "gross_pnl": gross_pnl,
                            "friction": fee,
                            "net_pnl": net_pnl,
                            "equity": equity,
                            "drawdown_pct": dd,
                            "bars_held": bars_held,
                        })
                        in_pos = False

                    elif bars_held >= self.max_hold_bars:
                        exit_price = row["close"]
                        gross_pnl = (entry_price - exit_price) * trade_qty
                        fee = (entry_price + exit_price) * trade_qty * ((self.maker_fee + self.taker_fee) / 2.0 + self.slippage_rate)
                        net_pnl = gross_pnl - fee
                        equity += net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (equity - peak_equity) / peak_equity * 100.0

                        trades.append({
                            "entry_time": df.iloc[entry_idx]["datetime"],
                            "exit_time": row["datetime"],
                            "direction": "SHORT",
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "outcome": "TIMEOUT",
                            "reason": f"TIMEOUT_EXIT ({bars_held} Bars / 30H)",
                            "gross_pnl": gross_pnl,
                            "friction": fee,
                            "net_pnl": net_pnl,
                            "equity": equity,
                            "drawdown_pct": dd,
                            "bars_held": bars_held,
                        })
                        in_pos = False

            if not in_pos:
                is_bull = bool(row.get("is_bull", False))
                is_bear = bool(row.get("is_bear", False))
                rvol_ok = float(row.get("rvol", 1.0)) >= self.min_rvol

                if is_bull and rvol_ok:
                    ema_aligned = row["ema20"] > row["ema50"]
                    pullback_ok = row["low"] <= (row["ema20"] * (1.0 + self.pullback_tolerance_pct))
                    rejection_ok = (row["lower_wick"] >= self.min_wick_ratio) or (row["close"] > row["open"])

                    if ema_aligned and pullback_ok and rejection_ok:
                        sl_dist = self.sl_atr_mult * row["atr14"]
                        if sl_dist > 0:
                            stop_pct = sl_dist / row["close"]
                            risk_budget = equity * self.risk_pct
                            notional_val = risk_budget / stop_pct
                            trade_qty = notional_val / row["close"]
                            entry_price = row["close"]
                            sl_price = entry_price - sl_dist
                            tp_price = entry_price + (sl_dist * self.rr_ratio)
                            entry_idx = i
                            pos_dir = 1
                            in_pos = True

                elif is_bear and rvol_ok:
                    ema_aligned = row["ema20"] < row["ema50"]
                    pullback_ok = row["high"] >= (row["ema20"] * (1.0 - self.pullback_tolerance_pct))
                    rejection_ok = (row["upper_wick"] >= self.min_wick_ratio) or (row["close"] < row["open"])

                    if ema_aligned and pullback_ok and rejection_ok:
                        sl_dist = self.sl_atr_mult * row["atr14"]
                        if sl_dist > 0:
                            stop_pct = sl_dist / row["close"]
                            risk_budget = equity * self.risk_pct
                            notional_val = risk_budget / stop_pct
                            trade_qty = notional_val / row["close"]
                            entry_price = row["close"]
                            sl_price = entry_price + sl_dist
                            tp_price = entry_price - (sl_dist * self.rr_ratio)
                            entry_idx = i
                            pos_dir = -1
                            in_pos = True

        tdf = pd.DataFrame(trades)
        monthly_summary = self._build_monthly_summary(tdf, starting_equity if starting_equity is not None else self.initial_capital)
        overall_metrics = self._build_overall_metrics(tdf, starting_equity if starting_equity is not None else self.initial_capital, equity)

        return overall_metrics, tdf, monthly_summary

    def _build_monthly_summary(self, tdf: pd.DataFrame, initial_capital: float) -> List[Dict[str, Any]]:
        if len(tdf) == 0:
            return []

        tdf_copy = tdf.copy()
        tdf_copy["entry_dt"] = pd.to_datetime(tdf_copy["entry_time"])
        tdf_copy["month_year"] = tdf_copy["entry_dt"].dt.strftime("%B %Y")
        tdf_copy["month_idx"] = tdf_copy["entry_dt"].dt.month

        monthly_records = []
        prev_equity = initial_capital

        for m_name in tdf_copy["month_year"].unique():
            m_df = tdf_copy[tdf_copy["month_year"] == m_name]
            n_trades = len(m_df)
            wins = (m_df["net_pnl"] > 0).sum()
            losses = (m_df["net_pnl"] <= 0).sum()
            tp_hits = (m_df["outcome"] == "TP").sum()
            sl_hits = (m_df["outcome"] == "SL").sum()
            timeouts = (m_df["outcome"] == "TIMEOUT").sum()

            gross_wins = m_df[m_df["net_pnl"] > 0]["net_pnl"].sum()
            gross_losses = abs(m_df[m_df["net_pnl"] <= 0]["net_pnl"].sum())
            pf = (gross_wins / gross_losses) if gross_losses > 0 else 99.0

            net_pnl = m_df["net_pnl"].sum()
            end_eq = m_df["equity"].iloc[-1]
            pct_ret = (net_pnl / prev_equity) * 100.0 if prev_equity > 0 else 0.0
            win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
            max_dd = m_df["drawdown_pct"].min() if not m_df.empty else 0.0

            monthly_records.append({
                "month_name": m_name,
                "trades": int(n_trades),
                "wins": int(wins),
                "losses": int(losses),
                "tp_hits": int(tp_hits),
                "sl_hits": int(sl_hits),
                "timeouts": int(timeouts),
                "win_rate": round(win_rate, 2),
                "net_pnl": round(net_pnl, 2),
                "return_pct": round(pct_ret, 2),
                "profit_factor": round(pf, 2),
                "max_drawdown": abs(round(max_dd, 2)),
                "ending_equity": round(end_eq, 2),
            })
            prev_equity = end_eq

        return monthly_records

    def _build_overall_metrics(self, tdf: pd.DataFrame, start_eq: float, end_eq: float) -> Dict[str, Any]:
        if len(tdf) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "final_equity": start_eq,
            }

        n_trades = len(tdf)
        wins = (tdf["net_pnl"] > 0).sum()
        win_rate = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
        total_pnl = tdf["net_pnl"].sum()
        tot_return = ((end_eq - start_eq) / start_eq) * 100.0

        gross_wins = tdf[tdf["net_pnl"] > 0]["net_pnl"].sum()
        gross_losses = abs(tdf[tdf["net_pnl"] <= 0]["net_pnl"].sum())
        pf = (gross_wins / gross_losses) if gross_losses > 0 else 99.0
        max_dd = abs(tdf["drawdown_pct"].min())

        return {
            "total_trades": int(n_trades),
            "wins": int(wins),
            "losses": int(n_trades - wins),
            "tp_hits": int((tdf["outcome"] == "TP").sum()),
            "sl_hits": int((tdf["outcome"] == "SL").sum()),
            "timeouts": int((tdf["outcome"] == "TIMEOUT").sum()),
            "win_rate": round(win_rate, 2),
            "net_pnl": round(total_pnl, 2),
            "return_pct": round(tot_return, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(max_dd, 2),
            "final_equity": round(end_eq, 2),
        }
