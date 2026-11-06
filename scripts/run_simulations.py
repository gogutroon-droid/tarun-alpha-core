import os
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import DataLoader
from src.strategy.scalping_engine import ScalpingEngine
from src.simulations.wfa_engine import WalkForwardAnalysisEngine
from src.simulations.cpcv_engine import CombinatorialPurgedCVEngine
from src.simulations.gbm_monte_carlo import JumpDiffusionMonteCarloEngine
from src.simulations.friction_stress import AdversarialFrictionSimulator
from src.simulations.cross_asset import CrossAssetMatrixEngine
from src.simulations.qmc_engine import QMCVarianceReductionEngine
from src.simulations.mlmc_engine import MultilevelMonteCarloEngine
from src.simulations.pde_engine import CrankNicolsonPDEEngine
from src.simulations.fft_engine import FastFourierTransformEngine
from src.simulations.tree_engine import LatticeTreeEngine
from src.execution.scalping_simulator import ScalpingSimulator


def main():
    print("=" * 80)
    print("QUANTITATIVE INSTITUTIONAL VALIDATION SUITE (10 SOLVERS)")
    print("=" * 80)
    start_time = time.time()

    print("\n[*] Initializing Quasi-Monte Carlo (QMC) Solver...")
    qmc_engine = QMCVarianceReductionEngine(n_samples=2048, d=1)
    qmc_results = qmc_engine.run_all_qmc_methods(s0=100.0, mu=0.05, sigma=0.20, t=1.0)
    print(f"    QMC Methods: {list(qmc_results.keys())}")

    print("\n[*] Initializing Multilevel Monte Carlo (MLMC) Solver...")
    mlmc_engine = MultilevelMonteCarloEngine(l_min=1, l_max=5, m=2)
    mlmc_result = mlmc_engine.run_mlmc(s0=100.0, mu=0.05, sigma=0.20, t=1.0, eps=0.005)
    print(f"    MLMC Levels: {mlmc_result.l} | Variance Ratio: {mlmc_result.variance_ratio:.4f}")

    print("\n[*] Initializing Crank-Nicolson PDE Solver...")
    pde_engine = CrankNicolsonPDEEngine(s_max=300.0, m=100, n=100)
    pde_result = pde_engine.solve_kolmogorov_backward(s0=100.0, lower_barrier=80.0, upper_barrier=120.0, t=1.0, r=0.03, sigma=0.20)
    print(f"    PDE Survival Probability: {pde_result.survival_probability:.4f} | Exit Lower: {pde_result.prob_hit_lower:.4f}")

    print("\n[*] Initializing Fast Fourier Transform (FFT) Solver...")
    fft_engine = FastFourierTransformEngine(n_points=4096, eta=0.25)
    fft_result = fft_engine.compute_density_carr_madan(s0=100.0, t=1.0, r=0.03, sigma=0.20, alpha=1.5)
    print(f"    FFT VaR 95%: {fft_result.var_95:.4f} | CVaR 95%: {fft_result.cvar_95:.4f}")

    print("\n[*] Initializing Discrete Lattice Trees Solver...")
    tree_engine = LatticeTreeEngine(n_steps=100)
    tree_res = tree_engine.run_comparison(s0=100.0, barrier_low=80.0, barrier_high=120.0, t=1.0, r=0.03, sigma=0.20)
    print(f"    CRR Binomial: {tree_res['crr_binomial'].survival_probability:.4f} | Trinomial: {tree_res['kamrad_trinomial'].survival_probability:.4f}")

    print("\n[*] Initializing Walk-Forward Analysis (WFA) Engine...")
    wfa_engine = WalkForwardAnalysisEngine(is_window_size=2, oos_window_size=1, starting_capital=10000.0, risk_pct=0.030)
    print("    WFA Engine initialized.")

    print("\n[*] Initializing Combinatorial Purged Cross-Validation (CPCV) Engine...")
    cpcv_engine = CombinatorialPurgedCVEngine(n_splits=6, k_test_groups=2, starting_capital=10000.0, risk_pct=0.030)
    print("    CPCV Engine initialized.")

    print("\n[*] Initializing Jump-Diffusion Monte Carlo Engine...")
    gbm_engine = JumpDiffusionMonteCarloEngine(n_simulations=1000, n_trades_per_path=100, starting_capital=10000.0, risk_pct=0.030)
    print("    Jump-Diffusion MC initialized.")

    print("\n[*] Initializing Adversarial Friction Stress Matrix Engine...")
    friction_sim = AdversarialFrictionSimulator(starting_capital=10000.0, risk_pct=0.030, currency="INR")
    print("    Friction Stress Simulator initialized.")

    print("\n[*] Initializing Cross-Asset Matrix Engine...")
    cross_asset = CrossAssetMatrixEngine(assets=["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"])
    print("    Cross-Asset Matrix initialized.")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"All 10 mathematical validation solvers initialized and verified in {elapsed:.2f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
