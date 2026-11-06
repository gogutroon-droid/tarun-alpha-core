from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class MLMCLevelResult:
    level: int
    n_samples: int
    step_size: float
    mean_diff: float
    var_diff: float
    cost_per_sample: float
    total_level_cost: float


@dataclass
class MLMCOverallResult:
    target_accuracy: float
    levels_evaluated: int
    mlmc_value: float
    mlmc_variance: float
    mlmc_standard_error: float
    total_computational_cost_mlmc: float
    equivalent_standard_mc_cost: float
    efficiency_speedup_factor: float
    level_breakdown: List[MLMCLevelResult]


class MultilevelMonteCarloEngine:

    def __init__(
        self,
        mu: float = 0.0001,
        sigma: float = 0.0080,
        t_horizon_bars: int = 127,
        m_refinement: int = 2,
        l_max: int = 5,
        starting_equity: float = 10000.0,
        risk_pct: float = 0.040,
        tp1_r: float = 1.60,
        tp2_r: float = 3.00,
        sl_r: float = 1.00
    ):
        self.mu = mu
        self.sigma = sigma
        self.t_horizon_bars = t_horizon_bars
        self.m_refinement = m_refinement
        self.l_max = l_max
        self.starting_equity = starting_equity
        self.risk_pct = risk_pct
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.sl_r = sl_r

    def _simulate_path_payoff(self, n_steps: int, d_w: np.ndarray) -> float:
        dt = 1.0 / n_steps
        s = 1.0
        r_current = 0.0
        be_hit = False
        tp1_hit = False
        trail_hit = False
        curr_sl = -self.sl_r

        for step_i in range(n_steps):
            dw_val = d_w[step_i]
            s = s * np.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * dw_val)
            r_gain = (s - 1.0) / (self.sigma * 2.0)

            if not be_hit and r_gain >= 1.10:
                be_hit = True
                curr_sl = max(curr_sl, 0.15)
            if not tp1_hit and r_gain >= self.tp1_r:
                tp1_hit = True
                be_hit = True
                curr_sl = max(curr_sl, 0.15)
            if tp1_hit and not trail_hit and r_gain >= 2.30:
                trail_hit = True
                curr_sl = max(curr_sl, 1.00)

            if r_gain <= curr_sl:
                if trail_hit:
                    return 0.5 * self.tp1_r + 0.5 * 1.00
                elif tp1_hit:
                    return 0.5 * self.tp1_r + 0.5 * 0.15
                elif be_hit:
                    return 0.15
                else:
                    return -self.sl_r
            if r_gain >= self.tp2_r:
                return 0.5 * self.tp1_r + 0.5 * self.tp2_r

        return r_gain

    def run_mlmc(self, target_accuracy: float = 0.02) -> MLMCOverallResult:
        np.random.seed(42)
        eps = target_accuracy
        levels = list(range(self.l_max + 1))
        level_results: List[MLMCLevelResult] = []

        n_pilot = 500
        mean_diffs = []
        var_diffs = []
        costs = []

        for l in levels:
            n_fine = (self.m_refinement ** l) * 10
            n_coarse = (self.m_refinement ** (l - 1)) * 10 if l > 0 else 0
            cost_l = (n_fine + n_coarse) if l > 0 else n_fine
            costs.append(cost_l)

            diffs = []
            for _ in range(n_pilot):
                if l == 0:
                    dw_fine = np.random.normal(0, 1, size=n_fine)
                    p_fine = self._simulate_path_payoff(n_fine, dw_fine)
                    diffs.append(p_fine)
                else:
                    dw_fine = np.random.normal(0, 1, size=n_fine)
                    m = self.m_refinement
                    dw_coarse = np.sum(dw_fine.reshape(-1, m), axis=1) / np.sqrt(m)

                    p_fine = self._simulate_path_payoff(n_fine, dw_fine)
                    p_coarse = self._simulate_path_payoff(n_coarse, dw_coarse)
                    diffs.append(p_fine - p_coarse)

            diffs = np.array(diffs)
            mean_diffs.append(np.mean(diffs))
            var_diffs.append(np.var(diffs))

        sqrt_vc_sum = sum(np.sqrt(var_diffs[k] * costs[k]) for k in levels)
        n_optimal = []
        for l in levels:
            nl = int(np.ceil(2.0 * (eps**-2) * np.sqrt(var_diffs[l] / costs[l]) * sqrt_vc_sum))
            n_optimal.append(max(n_pilot, min(nl, 50000)))

        mlmc_sum = 0.0
        mlmc_var = 0.0
        total_cost = 0.0

        for l in levels:
            nl = n_optimal[l]
            n_fine = (self.m_refinement ** l) * 10
            n_coarse = (self.m_refinement ** (l - 1)) * 10 if l > 0 else 0

            diffs = []
            for _ in range(nl):
                if l == 0:
                    dw_fine = np.random.normal(0, 1, size=n_fine)
                    p_fine = self._simulate_path_payoff(n_fine, dw_fine)
                    diffs.append(p_fine)
                else:
                    dw_fine = np.random.normal(0, 1, size=n_fine)
                    m = self.m_refinement
                    dw_coarse = np.sum(dw_fine.reshape(-1, m), axis=1) / np.sqrt(m)
                    p_fine = self._simulate_path_payoff(n_fine, dw_fine)
                    p_coarse = self._simulate_path_payoff(n_coarse, dw_coarse)
                    diffs.append(p_fine - p_coarse)

            diffs = np.array(diffs)
            m_l = float(np.mean(diffs))
            v_l = float(np.var(diffs))
            c_l = float(costs[l])

            mlmc_sum += m_l
            mlmc_var += v_l / nl
            total_cost += nl * c_l

            level_results.append(MLMCLevelResult(
                level=l,
                n_samples=nl,
                step_size=1.0 / n_fine,
                mean_diff=m_l,
                var_diff=v_l,
                cost_per_sample=c_l,
                total_level_cost=float(nl * c_l)
            ))

        var_pl = var_diffs[0]
        n_std = int(np.ceil(var_pl / (eps**2)))
        std_cost = float(n_std * costs[-1])
        speedup = std_cost / total_cost if total_cost > 0 else 1.0

        return MLMCOverallResult(
            target_accuracy=eps,
            levels_evaluated=len(levels),
            mlmc_value=float(mlmc_sum),
            mlmc_variance=float(mlmc_var),
            mlmc_standard_error=float(np.sqrt(mlmc_var)),
            total_computational_cost_mlmc=total_cost,
            equivalent_standard_mc_cost=std_cost,
            efficiency_speedup_factor=float(speedup),
            level_breakdown=level_results
        )
