from itertools import combinations, product
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from ..strategy.scalping_engine import ScalpingEngine
from ..execution.scalping_simulator import ScalpingSimulator


class CombinatorialPurgedCVEngine:

    def __init__(
        self,
        n_splits: int = 6,
        k_test_groups: int = 2,
        embargo_pct: float = 0.01,
        starting_capital: float = 10000.0,
        risk_pct: float = 0.040,
        currency: str = "INR"
    ):
        self.n_splits = n_splits
        self.k_test_groups = k_test_groups
        self.embargo_pct = embargo_pct
        self.starting_capital = starting_capital
        self.risk_pct = risk_pct
        self.currency = currency

        self.param_grid = [
            {"name": "P1 (Default)", "min_rvol": 1.60, "tp1_r": 1.60, "trail_trigger_r": 2.30},
            {"name": "P2 (Aggressive TP)", "min_rvol": 1.50, "tp1_r": 1.40, "trail_trigger_r": 2.00},
            {"name": "P3 (Conservative TP)", "min_rvol": 1.70, "tp1_r": 1.80, "trail_trigger_r": 2.50},
            {"name": "P4 (Tight Volume)", "min_rvol": 1.80, "tp1_r": 1.60, "trail_trigger_r": 2.20},
            {"name": "P5 (Early Trail)", "min_rvol": 1.60, "tp1_r": 1.50, "trail_trigger_r": 2.10},
            {"name": "P6 (Late Trail)", "min_rvol": 1.60, "tp1_r": 1.70, "trail_trigger_r": 2.60},
            {"name": "P7 (High RVOL Scale)", "min_rvol": 1.90, "tp1_r": 1.60, "trail_trigger_r": 2.40},
            {"name": "P8 (Loose RVOL)", "min_rvol": 1.40, "tp1_r": 1.60, "trail_trigger_r": 2.30},
        ]

    def _partition_dataset(
        self,
        monthly_data: Dict[int, Dict[str, pd.DataFrame]]
    ) -> List[Dict[str, Any]]:
        month_keys = sorted(monthly_data.keys())
        total_months = len(month_keys)

        groups: List[Dict[str, Any]] = []
        chunk_size = max(1, total_months // self.n_splits)

        for i in range(self.n_splits):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.n_splits - 1 else total_months
            group_months = month_keys[start_idx:end_idx]
            if not group_months:
                continue

            groups.append({
                "group_id": i,
                "month_keys": group_months,
                "month_names": [monthly_data[m]["month_name"] for m in group_months],
                "datasets": [monthly_data[m] for m in group_months]
            })

        return groups

    def _evaluate_param_on_groups(
        self,
        groups: List[Dict[str, Any]],
        params: Dict[str, Any]
    ) -> float:
        engine = ScalpingEngine(
            min_rvol=params["min_rvol"],
            tp1_r=params["tp1_r"],
            trail_trigger_r=params["trail_trigger_r"],
            risk_pct=self.risk_pct
        )

        all_trades = []
        current_eq = self.starting_capital

        for g in groups:
            for m_data in g["datasets"]:
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

        if len(all_trades) < 2:
            return 0.0

        pnls = np.array([t["net_pnl"] for t in all_trades])
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1)

        if std_pnl == 0 or np.isnan(std_pnl):
            return 0.0

        sharpe = (mean_pnl / std_pnl) * np.sqrt(min(252, len(pnls)))
        return float(sharpe)

    def run_cpcv(
        self,
        monthly_data: Dict[int, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Any]:
        groups = self._partition_dataset(monthly_data)
        actual_n = len(groups)
        all_group_indices = list(range(actual_n))

        test_combos = list(combinations(all_group_indices, self.k_test_groups))
        total_splits = len(test_combos)

        split_records = []
        is_best_oos_ranks = []
        logits = []

        for split_idx, test_indices in enumerate(test_combos):
            train_indices = [idx for idx in all_group_indices if idx not in test_indices]

            train_groups = [groups[idx] for idx in train_indices]
            test_groups = [groups[idx] for idx in test_indices]

            is_sharpes = []
            oos_sharpes = []

            for p in self.param_grid:
                is_s = self._evaluate_param_on_groups(train_groups, p)
                oos_s = self._evaluate_param_on_groups(test_groups, p)
                is_sharpes.append(is_s)
                oos_sharpes.append(oos_s)

            best_is_idx = int(np.argmax(is_sharpes))
            best_is_param = self.param_grid[best_is_idx]["name"]
            best_is_sharpe = is_sharpes[best_is_idx]
            best_oos_sharpe = oos_sharpes[best_is_idx]

            sorted_oos = sorted(oos_sharpes)
            rank_idx = sorted_oos.index(best_oos_sharpe)
            rel_rank = (rank_idx + 1) / len(sorted_oos)
            is_best_oos_ranks.append(rel_rank)

            p_val = max(1e-4, min(0.9999, rel_rank))
            logit = np.log(p_val / (1.0 - p_val))
            logits.append(logit)

            split_records.append({
                "split_id": split_idx + 1,
                "train_groups": train_indices,
                "test_groups": list(test_indices),
                "is_best_param": best_is_param,
                "is_best_sharpe": best_is_sharpe,
                "oos_sharpe": best_oos_sharpe,
                "oos_percentile_rank": rel_rank * 100.0,
                "underperformed_median": rel_rank < 0.50
            })

        underperformed_count = sum(1 for r in is_best_oos_ranks if r < 0.50)
        pbo = (underperformed_count / total_splits) if total_splits > 0 else 0.0

        avg_is_sharpe = np.mean([r["is_best_sharpe"] for r in split_records])
        avg_oos_sharpe = np.mean([r["oos_sharpe"] for r in split_records])
        sharpe_decay_pct = ((avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe * 100.0) if avg_is_sharpe != 0 else 0.0

        return {
            "total_splits": total_splits,
            "n_partitions": actual_n,
            "k_test_groups": self.k_test_groups,
            "pbo_probability": pbo,
            "avg_is_sharpe": avg_is_sharpe,
            "avg_oos_sharpe": avg_oos_sharpe,
            "sharpe_decay_pct": sharpe_decay_pct,
            "is_best_ranks": is_best_oos_ranks,
            "logits": logits,
            "splits": split_records
        }
