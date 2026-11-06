from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FFTDensityResult:
    x_grid: np.ndarray
    pdf_grid: np.ndarray
    cdf_grid: np.ndarray
    var_95_return: float
    var_99_return: float
    cvar_95_return: float
    cvar_99_return: float
    fft_expected_return: float
    fft_skewness: float
    fft_kurtosis: float
    n_fft_points: int


class FastFourierTransformEngine:

    def __init__(
        self,
        mu: float = 0.0001,
        sigma: float = 0.0080,
        lambda_jump: float = 0.035,
        mu_jump: float = -0.015,
        sigma_jump: float = 0.025,
        t_horizon_bars: int = 127,
        n_fft: int = 4096,
        u_max: float = 200.0
    ):
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.mu_jump = mu_jump
        self.sigma_jump = sigma_jump
        self.t_horizon_bars = t_horizon_bars
        self.n_fft = n_fft
        self.u_max = u_max

    def characteristic_function(self, u: np.ndarray) -> np.ndarray:
        t = float(self.t_horizon_bars)
        diff_term = 1j * u * self.mu * t - 0.5 * (u ** 2) * (self.sigma ** 2) * t
        jump_term = self.lambda_jump * t * (
            np.exp(1j * u * self.mu_jump - 0.5 * (u ** 2) * (self.sigma_jump ** 2)) - 1.0
        )
        return np.exp(diff_term + jump_term)

    def invert_density_fft(self) -> FFTDensityResult:
        n = self.n_fft
        du = self.u_max / (n // 2)
        u_grid = np.arange(0, n) * du - (n // 2) * du

        dx = (2.0 * np.pi) / (n * du)
        x_min = - (n // 2) * dx
        x_grid = x_min + np.arange(n) * dx

        weights = np.ones(n)
        weights[0] = 0.5
        weights[-1] = 0.5

        phi_vals = self.characteristic_function(u_grid)
        integrand = phi_vals * np.exp(-1j * u_grid * x_min) * weights * (du / (2.0 * np.pi))

        pdf_raw = np.real(np.fft.fft(integrand))
        pdf = np.maximum(0.0, pdf_raw)

        total_area = np.sum(pdf) * dx
        if total_area > 0:
            pdf = pdf / total_area

        cdf = np.cumsum(pdf) * dx
        cdf = np.clip(cdf, 0.0, 1.0)

        mean_x = float(np.sum(x_grid * pdf) * dx)
        var_x = float(np.sum(((x_grid - mean_x) ** 2) * pdf) * dx)
        std_x = np.sqrt(var_x) if var_x > 0 else 1e-6
        skew_x = float(np.sum(((x_grid - mean_x) ** 3) * pdf) * dx / (std_x ** 3))
        kurt_x = float(np.sum(((x_grid - mean_x) ** 4) * pdf) * dx / (std_x ** 4))

        idx_5 = np.argmin(np.abs(cdf - 0.05))
        idx_1 = np.argmin(np.abs(cdf - 0.01))
        var_95 = float(x_grid[idx_5])
        var_99 = float(x_grid[idx_1])

        cvar_95 = float(np.sum(x_grid[:idx_5] * pdf[:idx_5]) / np.sum(pdf[:idx_5])) if np.sum(pdf[:idx_5]) > 0 else var_95
        cvar_99 = float(np.sum(x_grid[:idx_1] * pdf[:idx_1]) / np.sum(pdf[:idx_1])) if np.sum(pdf[:idx_1]) > 0 else var_99

        return FFTDensityResult(
            x_grid=x_grid,
            pdf_grid=pdf,
            cdf_grid=cdf,
            var_95_return=var_95,
            var_99_return=var_99,
            cvar_95_return=cvar_95,
            cvar_99_return=cvar_99,
            fft_expected_return=mean_x,
            fft_skewness=skew_x,
            fft_kurtosis=kurt_x,
            n_fft_points=n
        )
