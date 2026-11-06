# alpha-core

Quantitative trading research engine, numerical simulation solvers, and event-driven backtesting framework.

Features:
- Model B Markov Scalper Engine:
  - 1D Macro Regime Filter: Rolling 10-day return momentum classification (Bull > +2.0%, Bear < -2.0%, Flat cash buffer)
  - 15M Microstructure: EMA 20/50 alignment, 0.20% pullback proximity, relative volume RVOL >= 1.20, wick rejection ratio >= 0.15
  - Risk & Execution Policy: Stop loss at 2.0x ATR14 (-1.0R), take profit at 30.0x ATR14 (+15.0R), 120-bar (30 hour) stagnation exit, zero premature breakeven choking
  - Dynamic Compounding: Intra-month 3.0% risk per trade with 100% calendar month sweep to cash at 00:00 UTC
- 10 Mathematical Simulation Solvers:
  - Quasi-Monte Carlo (QMC): Sobol, Halton, and Latin Hypercube Sampling (LHS)
  - Multilevel Monte Carlo (MLMC): Mike Giles geometric discretization ($M=2$, $l=0\dots 5$)
  - Crank-Nicolson PDE Solver: Numerical solve of Kolmogorov Backward Equation with Dirichlet absorbing barriers
  - Fast Fourier Transform (FFT) & Carr-Madan Inversion: Density reconstruction for tail risk VaR / CVaR
  - Discrete Lattice Trees: Cox-Ross-Rubinstein (CRR) binomial and Kamrad-Ritchken trinomial trees
  - Merton Jump-Diffusion MC: Poisson jump shocks with log-normal amplitudes
  - Walk-Forward Analysis (WFA): Rolling train/test efficiency evaluation across multiple windows
  - Combinatorial Purged & Embargoed Cross-Validation (CPCV): Marcos Lopez de Prado PBO estimator
  - Adversarial 2D Friction Stress Matrix: Slippage (0.0 to 20.0 bps) x fee multipliers (1x to 5x)
  - Cross-Asset Universality: Out-of-sample stress across BTC, ETH, SOL, XRP, DOGE, ADA
- Realistic Execution Friction: 0.02% maker fee, 0.03% taker fee, 1.0 bps slippage model

## Architecture

```
alpha-core/
├── src/
│   ├── indicators/
│   ├── strategy/
│   ├── backtest/
│   ├── simulations/
│   ├── analytics/
│   └── data_loader.py
├── scripts/
│   ├── run_backtest.py
│   └── run_simulations.py
├── requirements.txt
└── README.md
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run backtest simulation:
```bash
python -m scripts.run_backtest
```

3. Run quantitative validation solvers:
```bash
python -m scripts.run_simulations
```
