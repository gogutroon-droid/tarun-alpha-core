from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class ScalpingSimulator:

    def __init__(
        self,
        initial_capital: float = 10000.0,
        risk_pct: float = 0.050,
        tp1_r: float = 1.60,
        tp2_r: float = 3.00,
        tp1_share: float = 0.50,
        be_trigger_r: float = 1.10,
        be_offset_r: float = 0.15,
        trail_trigger_r: float = 2.30,
        trail_lock_r: float = 1.00,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0003,
        slippage_bps: float = 1.0,
        currency: str = "INR"
    ):
        self.initial_capital = initial_capital
        self.risk_pct = risk_pct
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.tp1_share = tp1_share
        self.be_trigger_r = be_trigger_r
        self.be_offset_r = be_offset_r
        self.trail_trigger_r = trail_trigger_r
        self.trail_lock_r = trail_lock_r
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage_rate = slippage_bps / 10000.0
        self.currency = currency

    def simulate_trades(
        self,
        candidates: List[dict],
        df_5m: pd.DataFrame,
        starting_equity: Optional[float] = None
    ) -> Tuple[List[dict], pd.DataFrame]:
        if starting_equity is None:
            equity = self.initial_capital
        else:
            equity = starting_equity

        highs = df_5m["high"].values
        lows = df_5m["low"].values
        closes = df_5m["close"].values
        dts = df_5m["datetime"].values
        n_bars = len(df_5m)

        trades = []
        last_exit_idx = -1

        for c in candidates:
            exec_idx = c["exec_idx"]
            if exec_idx <= last_exit_idx or exec_idx >= n_bars:
                continue

            entry_p = c["entry"]
            sl_p = c["sl"]
            stop_dist = c["stop_dist"]
            dir_val = c["dir"]

            risk_amt = equity * self.risk_pct
            stop_dist_pct = stop_dist / entry_p
            notional_val = risk_amt / stop_dist_pct

            if dir_val == "LONG":
                tp1_p = entry_p + (self.tp1_r * stop_dist)
                tp2_p = entry_p + (self.tp2_r * stop_dist)
                be_trig_p = entry_p + (self.be_trigger_r * stop_dist)
                be_sl_p = entry_p + (self.be_offset_r * stop_dist)
                trail_trig_p = entry_p + (self.trail_trigger_r * stop_dist)
                trail_sl_p = entry_p + (self.trail_lock_r * stop_dist)
            else:
                tp1_p = entry_p - (self.tp1_r * stop_dist)
                tp2_p = entry_p - (self.tp2_r * stop_dist)
                be_trig_p = entry_p - (self.be_trigger_r * stop_dist)
                be_sl_p = entry_p - (self.be_offset_r * stop_dist)
                trail_trig_p = entry_p - (self.trail_trigger_r * stop_dist)
                trail_sl_p = entry_p - (self.trail_lock_r * stop_dist)

            tp1_hit = False
            trail_hit = False
            curr_sl = sl_p
            trade_closed = False

            for bar_i in range(exec_idx, n_bars):
                b_high = highs[bar_i]
                b_low = lows[bar_i]

                if dir_val == "LONG":
                    if not tp1_hit and b_high >= be_trig_p:
                        tp1_hit = True
                        curr_sl = be_sl_p

                    if tp1_hit and not trail_hit and b_high >= trail_trig_p:
                        trail_hit = True
                        curr_sl = trail_sl_p

                    sl_hit = b_low <= curr_sl
                    tp2_hit = b_high >= tp2_p

                    if sl_hit or tp2_hit:
                        last_exit_idx = bar_i
                        if tp2_hit and not sl_hit:
                            gross_val = (self.tp1_share * self.tp1_r * risk_amt) + ((1.0 - self.tp1_share) * self.tp2_r * risk_amt)
                            reason = "TP2_FULL (+3.0R Full Runner)"
                            exit_p = tp2_p
                            realized_r = (self.tp1_share * self.tp1_r) + ((1.0 - self.tp1_share) * self.tp2_r)
                        elif tp1_hit and sl_hit:
                            realized_runner_r = (curr_sl - entry_p) / stop_dist
                            gross_val = (self.tp1_share * self.tp1_r * risk_amt) + ((1.0 - self.tp1_share) * realized_runner_r * risk_amt)
                            reason = "TRAIL_PROFIT (Trailed +1.0R SL Hit)" if trail_hit else "TP1_BE_EXIT (50% TP1 + BE)"
                            exit_p = curr_sl
                            realized_r = (self.tp1_share * self.tp1_r) + ((1.0 - self.tp1_share) * realized_runner_r)
                        else:
                            gross_val = -1.0 * risk_amt
                            reason = "SL_FULL (-1.0R Stop Out)"
                            exit_p = curr_sl
                            realized_r = -1.0

                        entry_fee = notional_val * self.maker_fee
                        exit_notional = notional_val * (exit_p / entry_p)
                        exit_fee = exit_notional * self.taker_fee
                        slip_cost = (notional_val + exit_notional) * (self.slippage_rate / 2.0)
                        tot_cost = entry_fee + exit_fee + slip_cost
                        net_pnl = gross_val - tot_cost
                        equity += net_pnl

                        trades.append({
                            "entry_time": c["dt"],
                            "exit_time": str(dts[bar_i]),
                            "bars_held": bar_i - exec_idx + 1,
                            "direction": dir_val,
                            "setup": c["setup_name"],
                            "entry_price": entry_p,
                            "exit_price": exit_p,
                            "stop_loss": sl_p,
                            "final_trailed_sl": curr_sl,
                            "risk_amount": risk_amt,
                            "notional_val": notional_val,
                            "gross_pnl": gross_val,
                            "friction": tot_cost,
                            "net_pnl": net_pnl,
                            "outcome": reason,
                            "r_realized": realized_r,
                            "equity": equity
                        })
                        trade_closed = True
                        break

                elif dir_val == "SHORT":
                    if not tp1_hit and b_low <= be_trig_p:
                        tp1_hit = True
                        curr_sl = be_sl_p

                    if tp1_hit and not trail_hit and b_low <= trail_trig_p:
                        trail_hit = True
                        curr_sl = trail_sl_p

                    sl_hit = b_high >= curr_sl
                    tp2_hit = b_low <= tp2_p

                    if sl_hit or tp2_hit:
                        last_exit_idx = bar_i
                        if tp2_hit and not sl_hit:
                            gross_val = (self.tp1_share * self.tp1_r * risk_amt) + ((1.0 - self.tp1_share) * self.tp2_r * risk_amt)
                            reason = "TP2_FULL (+3.0R Full Runner)"
                            exit_p = tp2_p
                            realized_r = (self.tp1_share * self.tp1_r) + ((1.0 - self.tp1_share) * self.tp2_r)
                        elif tp1_hit and sl_hit:
                            realized_runner_r = (entry_p - curr_sl) / stop_dist
                            gross_val = (self.tp1_share * self.tp1_r * risk_amt) + ((1.0 - self.tp1_share) * realized_runner_r * risk_amt)
                            reason = "TRAIL_PROFIT (Trailed +1.0R SL Hit)" if trail_hit else "TP1_BE_EXIT (50% TP1 + BE)"
                            exit_p = curr_sl
                            realized_r = (self.tp1_share * self.tp1_r) + ((1.0 - self.tp1_share) * realized_runner_r)
                        else:
                            gross_val = -1.0 * risk_amt
                            reason = "SL_FULL (-1.0R Stop Out)"
                            exit_p = curr_sl
                            realized_r = -1.0

                        entry_fee = notional_val * self.maker_fee
                        exit_notional = notional_val * (entry_p / exit_p)
                        exit_fee = exit_notional * self.taker_fee
                        slip_cost = (notional_val + exit_notional) * (self.slippage_rate / 2.0)
                        tot_cost = entry_fee + exit_fee + slip_cost
                        net_pnl = gross_val - tot_cost
                        equity += net_pnl

                        trades.append({
                            "entry_time": c["dt"],
                            "exit_time": str(dts[bar_i]),
                            "bars_held": bar_i - exec_idx + 1,
                            "direction": dir_val,
                            "setup": c["setup_name"],
                            "entry_price": entry_p,
                            "exit_price": exit_p,
                            "stop_loss": sl_p,
                            "final_trailed_sl": curr_sl,
                            "risk_amount": risk_amt,
                            "notional_val": notional_val,
                            "gross_pnl": gross_val,
                            "friction": tot_cost,
                            "net_pnl": net_pnl,
                            "outcome": reason,
                            "r_realized": realized_r,
                            "equity": equity
                        })
                        trade_closed = True
                        break

            if not trade_closed:
                last_exit_idx = n_bars

        tdf = pd.DataFrame(trades)
        return trades, tdf
