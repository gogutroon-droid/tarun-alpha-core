from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TreeSimulationResult:
    tree_type: str
    n_steps: int
    expected_value_r: float
    win_probability: float
    tp1_knockout_prob: float
    tp2_knockout_prob: float
    sl_knockout_prob: float
    up_step_factor: float
    down_step_factor: float
    step_dt: float


class LatticeTreeEngine:

    def __init__(
        self,
        mu: float = 0.0001,
        sigma: float = 0.0080,
        t_bars: int = 50,
        s0: float = 60000.0,
        sl_r: float = 1.00,
        tp1_r: float = 1.60,
        tp2_r: float = 3.00,
        atr_stop_pct: float = 0.008
    ):
        self.mu = mu
        self.sigma = sigma
        self.t_bars = t_bars
        self.s0 = s0
        self.sl_r = sl_r
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.atr_stop_pct = atr_stop_pct

    def run_crr_binomial_tree(self) -> TreeSimulationResult:
        n = self.t_bars
        dt = 1.0 / n
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1.0 / u
        p_up = (np.exp(self.mu * dt) - d) / (u - d)
        p_up = np.clip(p_up, 0.01, 0.99)
        p_down = 1.0 - p_up

        stop_dist = self.s0 * self.atr_stop_pct
        sl_barrier = self.s0 - (self.sl_r * stop_dist)
        tp1_barrier = self.s0 + (self.tp1_r * stop_dist)
        tp2_barrier = self.s0 + (self.tp2_r * stop_dist)

        v = np.zeros(n + 1)
        for j in range(n + 1):
            s_node = self.s0 * (u ** j) * (d ** (n - j))
            r_val = (s_node - self.s0) / stop_dist
            v[j] = np.clip(r_val, -self.sl_r, self.tp2_r)

        for i in range(n - 1, -1, -1):
            v_next = np.zeros(i + 1)
            for j in range(i + 1):
                s_node = self.s0 * (u ** j) * (d ** (i - j))
                if s_node >= tp2_barrier:
                    v_next[j] = (0.5 * self.tp1_r) + (0.5 * self.tp2_r)
                elif s_node <= sl_barrier:
                    v_next[j] = -self.sl_r
                else:
                    exp_val = (p_up * v[j + 1]) + (p_down * v[j])
                    v_next[j] = exp_val
            v = v_next

        expected_r = float(v[0])

        p_win = float(1.0 / (1.0 + np.exp(- 2.0 * self.mu * float(n) / (self.sigma ** 2 + 1e-6))))
        p_win = np.clip(p_win, 0.45, 0.65)
        p_tp1 = float(p_win * 0.90)
        p_tp2 = float(p_win * 0.42)
        p_sl = float(1.0 - p_win)

        return TreeSimulationResult(
            tree_type="Cox-Ross-Rubinstein (CRR) Binomial",
            n_steps=n,
            expected_value_r=expected_r,
            win_probability=p_win,
            tp1_knockout_prob=p_tp1,
            tp2_knockout_prob=p_tp2,
            sl_knockout_prob=p_sl,
            up_step_factor=float(u),
            down_step_factor=float(d),
            step_dt=float(dt)
        )

    def run_trinomial_tree(self) -> TreeSimulationResult:
        n = self.t_bars
        dt = 1.0 / n
        lambda_param = 1.22474
        u = np.exp(lambda_param * self.sigma * np.sqrt(dt))
        d = 1.0 / u

        mu_term = self.mu * dt
        p_up = 1.0 / (2.0 * (lambda_param ** 2)) + (mu_term / (2.0 * lambda_param * self.sigma * np.sqrt(dt)))
        p_down = 1.0 / (2.0 * (lambda_param ** 2)) - (mu_term / (2.0 * lambda_param * self.sigma * np.sqrt(dt)))
        p_mid = 1.0 - p_up - p_down

        p_up = np.clip(p_up, 0.01, 0.98)
        p_down = np.clip(p_down, 0.01, 0.98)
        p_mid = np.clip(p_mid, 0.01, 0.98)

        stop_dist = self.s0 * self.atr_stop_pct
        sl_barrier = self.s0 - (self.sl_r * stop_dist)
        tp1_barrier = self.s0 + (self.tp1_r * stop_dist)
        tp2_barrier = self.s0 + (self.tp2_r * stop_dist)

        v = np.zeros(2 * n + 1)
        for k in range(-n, n + 1):
            idx = k + n
            s_node = self.s0 * (u ** k)
            r_val = (s_node - self.s0) / stop_dist
            v[idx] = np.clip(r_val, -self.sl_r, self.tp2_r)

        for i in range(n - 1, -1, -1):
            v_next = np.zeros(2 * i + 1)
            for k in range(-i, i + 1):
                idx = k + i
                s_node = self.s0 * (u ** k)
                if s_node >= tp2_barrier:
                    v_next[idx] = (0.5 * self.tp1_r) + (0.5 * self.tp2_r)
                elif s_node <= sl_barrier:
                    v_next[idx] = -self.sl_r
                else:
                    idx_prev_up = (k + 1) + (i + 1)
                    idx_prev_mid = k + (i + 1)
                    idx_prev_down = (k - 1) + (i + 1)
                    exp_val = (p_up * v[idx_prev_up]) + (p_mid * v[idx_prev_mid]) + (p_down * v[idx_prev_down])
                    v_next[idx] = exp_val
            v = v_next

        expected_r = float(v[0])

        p_win = float(1.0 / (1.0 + np.exp(- 2.0 * self.mu * float(n) / (self.sigma ** 2 + 1e-6))))
        p_win = np.clip(p_win, 0.46, 0.66)
        p_tp1 = float(p_win * 0.92)
        p_tp2 = float(p_win * 0.44)
        p_sl = float(1.0 - p_win)

        return TreeSimulationResult(
            tree_type="Kamrad-Ritchken Trinomial",
            n_steps=n,
            expected_value_r=expected_r,
            win_probability=p_win,
            tp1_knockout_prob=p_tp1,
            tp2_knockout_prob=p_tp2,
            sl_knockout_prob=p_sl,
            up_step_factor=float(u),
            down_step_factor=float(d),
            step_dt=float(dt)
        )
