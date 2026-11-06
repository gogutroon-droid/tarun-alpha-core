from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class JumpDiffusionMonteCarloEngine:

    def __init__(
        self,
        n_simulations: int = 1000,
        n_trades_per_path: int = 127,
        starting_capital: float = 10000.0,
        risk_pct: float = 0.040,
        currency: str = "INR"
    ):
        self.n_simulations = n_simulations
        self.n_trades_per_path = n_trades_per_path
        self.starting_capital = starting_capital
        self.risk_pct = risk_pct
        self.currency = currency

    def fit_empirical_parameters(self, df_5m: pd.DataFrame) -> Dict[str, float]:
        closes = df_5m["close"].values
        log_returns = np.diff(np.log(closes))

        mu = np.mean(log_returns)
        sigma = np.std(log_returns)

        jump_mask = np.abs(log_returns - mu) > (3.0 * sigma)
        jumps = log_returns[jump_mask]

        lambda_jump = len(jumps) / len(log_returns) if len(log_returns) > 0 else 0.005
        mu_jump = np.mean(jumps) if len(jumps) > 0 else 0.0
        sigma_jump = np.std(jumps) if len(jumps) > 0 else sigma * 2.0

        return {
            "mu": float(mu),
            "sigma": float(sigma),
            "lambda_jump": float(lambda_jump),
            "mu_jump": float(mu_jump),
            "sigma_jump": float(sigma_jump)
        }

    def run_monte_carlo(
        self,
        historical_trades: List[Dict[str, Any]],
        df_5m: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        np.random.seed(42)

        if historical_trades:
            r_multiples = []
            for t in historical_trades:
                risk_b = t.get("risk_budget", t.get("risk_amt", 400.0))
                net_pnl = t.get("net_pnl", 0.0)
                r_mult = net_pnl / risk_b if risk_b > 0 else 0.0
                r_multiples.append(r_mult)
            r_multiples = np.array(r_multiples)
        else:
            r_multiples = np.array(
                [-1.0] * 42 + [0.15] * 15 + [0.875] * 10 + [1.80] * 8 + [3.0] * 25
            )

        n_trades = self.n_trades_per_path
        jump_prob = 0.035
        jump_adverse_impact_r = -0.50

        all_final_equities = []
        all_max_drawdowns = []
        all_equity_curves = []
        circuit_breaker_breaches = 0

        for sim_i in range(self.n_simulations):
            sampled_r = np.random.choice(r_multiples, size=n_trades, replace=True)

            jump_events = np.random.binomial(1, jump_prob, size=n_trades)
            shocks = jump_events * np.random.normal(jump_adverse_impact_r, 0.25, size=n_trades)
            effective_r = sampled_r + shocks

            equity = self.starting_capital
            peak_equity = equity
            max_dd = 0.0
            daily_start_equity = equity
            hit_circuit_breaker = False
            curve = [equity]

            for trade_idx, r_val in enumerate(effective_r):
                if trade_idx % 15 == 0:
                    daily_start_equity = equity

                risk_budget = equity * self.risk_pct
                trade_pnl = r_val * risk_budget
                equity += trade_pnl
                curve.append(equity)

                if equity > peak_equity:
                    peak_equity = equity

                dd = (peak_equity - equity) / peak_equity * 100.0
                if dd > max_dd:
                    max_dd = dd

                if daily_start_equity > 0:
                    daily_loss = (daily_start_equity - equity) / daily_start_equity
                    if daily_loss >= 0.10:
                        hit_circuit_breaker = True

            all_final_equities.append(equity)
            all_max_drawdowns.append(max_dd)
            all_equity_curves.append(curve)

            if hit_circuit_breaker:
                circuit_breaker_breaches += 1

        all_final_equities = np.array(all_final_equities)
        all_max_drawdowns = np.array(all_max_drawdowns)

        eq_p5 = np.percentile(all_final_equities, 5)
        eq_p25 = np.percentile(all_final_equities, 25)
        eq_p50 = np.percentile(all_final_equities, 50)
        eq_p75 = np.percentile(all_final_equities, 75)
        eq_p95 = np.percentile(all_final_equities, 95)

        dd_median = np.median(all_max_drawdowns)
        dd_var_95 = np.percentile(all_max_drawdowns, 95)
        dd_var_99 = np.percentile(all_max_drawdowns, 99)
        dd_cvar_95 = np.mean(all_max_drawdowns[all_max_drawdowns >= dd_var_95])

        risk_of_ruin_pct = (np.sum(all_final_equities < (self.starting_capital * 0.50)) / self.n_simulations) * 100.0
        profitable_paths_pct = (np.sum(all_final_equities > self.starting_capital) / self.n_simulations) * 100.0
        circuit_breaker_paths_pct = (circuit_breaker_breaches / self.n_simulations) * 100.0

        return {
            "n_simulations": self.n_simulations,
            "n_trades_per_path": n_trades,
            "starting_capital": self.starting_capital,
            "equity_median": float(eq_p50),
            "equity_p5": float(eq_p5),
            "equity_p25": float(eq_p25),
            "equity_p75": float(eq_p75),
            "equity_p95": float(eq_p95),
            "max_dd_median": float(dd_median),
            "max_dd_var_95": float(dd_var_95),
            "max_dd_var_99": float(dd_var_99),
            "max_dd_cvar_95": float(dd_cvar_95),
            "risk_of_ruin_pct": float(risk_of_ruin_pct),
            "circuit_breaker_paths_pct": float(circuit_breaker_paths_pct),
            "profitable_paths_pct": float(profitable_paths_pct),
            "sample_curves": all_equity_curves[:10]
        }
