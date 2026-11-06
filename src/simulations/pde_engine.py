from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.linalg as la


@dataclass
class PDEMatrixResult:
    x_grid: np.ndarray
    price_grid: np.ndarray
    tau_grid: np.ndarray
    tp1_hit_probability_surface: np.ndarray
    sl_hit_probability_surface: np.ndarray
    entry_tp1_probability: float
    entry_tp2_probability: float
    entry_sl_probability: float
    expected_r_theoretical: float
    stability_condition_met: bool


class CrankNicolsonPDEEngine:

    def __init__(
        self,
        mu: float = 0.0001,
        sigma: float = 0.0080,
        t_max_bars: int = 100,
        n_space_steps: int = 200,
        n_time_steps: int = 100,
        s0: float = 60000.0,
        sl_r: float = 1.00,
        tp1_r: float = 1.60,
        tp2_r: float = 3.00,
        atr_stop_pct: float = 0.008
    ):
        self.mu = mu
        self.sigma = sigma
        self.t_max_bars = t_max_bars
        self.n_space_steps = n_space_steps
        self.n_time_steps = n_time_steps
        self.s0 = s0
        self.sl_r = sl_r
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.atr_stop_pct = atr_stop_pct

    def solve_barrier_pde(self) -> PDEMatrixResult:
        r_unit = self.atr_stop_pct
        x_min = - (self.sl_r * r_unit * 1.5)
        x_max = + (self.tp2_r * r_unit * 1.5)

        x_grid = np.linspace(x_min, x_max, self.n_space_steps)
        dx = x_grid[1] - x_grid[0]

        t_total = float(self.t_max_bars)
        tau_grid = np.linspace(0, t_total, self.n_time_steps)
        d_tau = tau_grid[1] - tau_grid[0]

        a = 0.5 * (self.sigma ** 2)
        b = self.mu - 0.5 * (self.sigma ** 2)

        alpha = a * d_tau / (2.0 * dx**2)
        beta = b * d_tau / (4.0 * dx)

        sub_diag_lhs = - alpha + beta
        main_diag_lhs = 1.0 + 2.0 * alpha
        sup_diag_lhs = - alpha - beta

        sub_diag_rhs = alpha - beta
        main_diag_rhs = 1.0 - 2.0 * alpha
        sup_diag_rhs = alpha + beta

        m = self.n_space_steps - 2

        ab_lhs = np.zeros((3, m))
        ab_lhs[0, 1:] = sup_diag_lhs
        ab_lhs[1, :] = main_diag_lhs
        ab_lhs[2, :-1] = sub_diag_lhs

        x_tp1 = self.tp1_r * r_unit
        u_tp1 = np.where(x_grid >= x_tp1, 1.0, 0.0)

        tp1_surface = np.zeros((self.n_time_steps, self.n_space_steps))
        tp1_surface[0, :] = u_tp1

        u_curr = u_tp1.copy()
        for t_idx in range(1, self.n_time_steps):
            rhs = np.zeros(m)
            for i in range(m):
                idx = i + 1
                rhs[i] = (
                    sub_diag_rhs * u_curr[idx - 1] +
                    main_diag_rhs * u_curr[idx] +
                    sup_diag_rhs * u_curr[idx + 1]
                )

            rhs[0] -= sub_diag_lhs * 0.0
            rhs[-1] -= sup_diag_lhs * 1.0

            u_internal = la.solve_banded((1, 1), ab_lhs, rhs)
            u_curr[1:-1] = u_internal
            u_curr[0] = 0.0
            u_curr[-1] = 1.0
            tp1_surface[t_idx, :] = u_curr

        sl_surface = 1.0 - tp1_surface

        idx_entry = np.argmin(np.abs(x_grid - 0.0))
        p_tp1_entry = float(tp1_surface[-1, idx_entry])
        p_sl_entry = float(sl_surface[-1, idx_entry])

        p_tp2_entry = float(p_tp1_entry * 0.45)

        e_r = (p_tp2_entry * 2.30) + ((p_tp1_entry - p_tp2_entry) * 0.875) - (p_sl_entry * 1.00)

        return PDEMatrixResult(
            x_grid=x_grid,
            price_grid=self.s0 * np.exp(x_grid),
            tau_grid=tau_grid,
            tp1_hit_probability_surface=tp1_surface,
            sl_hit_probability_surface=sl_surface,
            entry_tp1_probability=p_tp1_entry,
            entry_tp2_probability=p_tp2_entry,
            entry_sl_probability=p_sl_entry,
            expected_r_theoretical=float(e_r),
            stability_condition_met=bool(alpha > 0)
        )
