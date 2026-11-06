from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from ..data_loader import DataLoader
from ..strategy.scalping_engine import ScalpingEngine
from ..execution.scalping_simulator import ScalpingSimulator


@dataclass
class WFAWindowResult:
    window_id: int
    train_months: List[str]
    test_month: str
    best_params: Dict[str, Any]
    is_trades: int
    is_win_rate: float
    is_net_pnl: float
    is_profit_factor: float
    oos_trades: int
    oos_win_rate: float
    oos_net_pnl: float
    oos_profit_factor: float
    wfe_efficiency_ratio: float


class WalkForwardAnalysisEngine:

    def __init__(
        self,
        is_window_size: int = 2,
        oos_window_size: int = 1,
        starting_capital: float = 10000.0,
        risk_pct: float = 0.040,
        currency: str = "INR"
    ):
        self.is_window_size = is_window_size
        self.oos_window_size = oos_window_size
        self.starting_capital = starting_capital
        self.risk_pct = risk_pct
        self.currency = currency

        self.param_grid = {
            "tp1_r": [1.40, 1.60, 1.80],
            "trail_trigger_r": [2.00, 2.30, 2.60],
            "min_rvol": [1.40, 1.60, 1.80]
        }

    def _generate_param_combinations(self) -> List[Dict[str, Any]]:
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in product(*values)]

    def _evaluate_params_on_months(
        self,
        months_data: List[Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
        starting_equity: float
    ) -> Tuple[List[dict], float, float, float]:
        engine = ScalpingEngine(
            min_rvol=params["min_rvol"],
            tp1_r=params["tp1_r"],
            trail_trigger_r=params["trail_trigger_r"],
            risk_pct=self.risk_pct
        )

        all_trades = []
        current_eq = starting_equity

        for m_data in months_data:
            signals = engine.generate_monthly_signals(m_data)
            if not signals:
                continue

            sim = ScalpingSimulator(
                initial_capital=self.starting_capital,
                risk_pct=self.risk_pct,
                tp1_r=params["tp1_r"],
                trail_trigger_r=params["trail_trigger_r"],
                currency=self.currency
            )
            m_trades, _ = sim.simulate_trades(
                candidates=signals,
                df_5m=m_data["5m"],
                starting_equity=current_eq
            )
            all_trades.extend(m_trades)
            if m_trades:
                current_eq = m_trades[-1].get("equity", m_trades[-1].get("account_equity", current_eq))

        if not all_trades:
            return [], 0.0, 0.0, 0.0

        n_trades = len(all_trades)
        wins = [t for t in all_trades if t["net_pnl"] > 0]
        losses = [t for t in all_trades if t["net_pnl"] < 0]
        win_rate = (len(wins) / n_trades) * 100.0 if n_trades > 0 else 0.0

        tot_win = sum(t["net_pnl"] for t in wins)
        tot_loss = abs(sum(t["net_pnl"] for t in losses))
        profit_factor = (tot_win / tot_loss) if tot_loss > 0 else (tot_win if tot_win > 0 else 1.0)
        net_pnl = sum(t["net_pnl"] for t in all_trades)

        return all_trades, win_rate, profit_factor, net_pnl

    def run_walk_forward(
        self,
        monthly_data: Dict[int, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        month_keys = sorted(monthly_data.keys())
        total_months = len(month_keys)

        param_combos = self._generate_param_combinations()
        window_results: List[WFAWindowResult] = []
        oos_all_trades: List[dict] = []

        running_oos_equity = self.starting_capital

        for i in range(total_months - self.is_window_size):
            is_month_nums = month_keys[i : i + self.is_window_size]
            oos_month_num = month_keys[i + self.is_window_size]

            is_datasets = [monthly_data[m] for m in is_month_nums]
            oos_dataset = [monthly_data[oos_month_num]]

            is_names = [monthly_data[m]["month_name"] for m in is_month_nums]
            oos_name = monthly_data[oos_month_num]["month_name"]

            best_p = None
            best_score = -1e9
            best_is_stats = (0, 0.0, 0.0, 0.0)

            for p in param_combos:
                trades, wr, pf, net_pnl = self._evaluate_params_on_months(
                    is_datasets, p, starting_equity=self.starting_capital
                )
                score = pf * (len(trades) ** 0.5) if trades else -1e9
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_is_stats = (len(trades), wr, pf, net_pnl)

            oos_trades, oos_wr, oos_pf, oos_pnl = self._evaluate_params_on_months(
                oos_dataset, best_p, starting_equity=running_oos_equity
            )

            if oos_trades:
                running_oos_equity = oos_trades[-1].get("equity", oos_trades[-1].get("account_equity", running_oos_equity))
                oos_all_trades.extend(oos_trades)

            is_pf = best_is_stats[2]
            wfe = (oos_pf / is_pf) if is_pf > 0 else 0.0

            res = WFAWindowResult(
                window_id=i + 1,
                train_months=is_names,
                test_month=oos_name,
                best_params=best_p,
                is_trades=best_is_stats[0],
                is_win_rate=best_is_stats[1],
                is_profit_factor=best_is_stats[2],
                is_net_pnl=best_is_stats[3],
                oos_trades=len(oos_trades),
                oos_win_rate=oos_wr,
                oos_profit_factor=oos_pf,
                oos_net_pnl=oos_pnl,
                wfe_efficiency_ratio=wfe
            )
            window_results.append(res)

        tot_oos_trades = len(oos_all_trades)
        oos_wins = [t for t in oos_all_trades if t["net_pnl"] > 0]
        oos_losses = [t for t in oos_all_trades if t["net_pnl"] < 0]
        overall_oos_wr = (len(oos_wins) / tot_oos_trades * 100.0) if tot_oos_trades > 0 else 0.0
        tot_oos_win_pnl = sum(t["net_pnl"] for t in oos_wins)
        tot_oos_loss_pnl = abs(sum(t["net_pnl"] for t in oos_losses))
        overall_oos_pf = (tot_oos_win_pnl / tot_oos_loss_pnl) if tot_oos_loss_pnl > 0 else 1.0
        avg_wfe = np.mean([w.wfe_efficiency_ratio for w in window_results]) if window_results else 0.0

        return {
            "windows": window_results,
            "total_oos_trades": tot_oos_trades,
            "overall_oos_win_rate": overall_oos_wr,
            "overall_oos_profit_factor": overall_oos_pf,
            "final_oos_equity": running_oos_equity,
            "total_oos_net_profit": running_oos_equity - self.starting_capital,
            "average_wfe": avg_wfe,
            "oos_trades_log": oos_all_trades
        }
