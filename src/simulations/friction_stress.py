from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from ..strategy.scalping_engine import ScalpingEngine
from ..execution.scalping_simulator import ScalpingSimulator


@dataclass
class FrictionMatrixCell:
    fee_multiplier: float
    maker_fee_pct: float
    taker_fee_pct: float
    slippage_bps: float
    final_equity: float
    net_profit: float
    total_friction_cost: float
    profit_factor: float
    win_rate: float
    is_profitable: bool


class AdversarialFrictionSimulator:

    def __init__(
        self,
        base_maker_fee: float = 0.00020,
        base_taker_fee: float = 0.00030,
        starting_capital: float = 10000.0,
        risk_pct: float = 0.040,
        currency: str = "INR"
    ):
        self.base_maker_fee = base_maker_fee
        self.base_taker_fee = base_taker_fee
        self.starting_capital = starting_capital
        self.risk_pct = risk_pct
        self.currency = currency

        self.fee_multipliers = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        self.slippage_bps_levels = [0.0, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]

    def run_stress_matrix(
        self,
        monthly_data: Dict[int, Dict[str, pd.DataFrame]],
        strategy_engine: ScalpingEngine
    ) -> Dict[str, Any]:
        cached_signals = {}
        for m_num, m_data in monthly_data.items():
            cached_signals[m_num] = strategy_engine.generate_monthly_signals(m_data)

        matrix_results: List[FrictionMatrixCell] = []

        for fee_mult in self.fee_multipliers:
            maker_f = self.base_maker_fee * fee_mult
            taker_f = self.base_taker_fee * fee_mult

            for slip_bps in self.slippage_bps_levels:
                sim = ScalpingSimulator(
                    initial_capital=self.starting_capital,
                    risk_pct=self.risk_pct,
                    tp1_r=strategy_engine.tp1_r,
                    tp2_r=strategy_engine.tp2_r,
                    tp1_share=strategy_engine.tp1_share,
                    be_trigger_r=strategy_engine.be_trigger_r,
                    be_offset_r=strategy_engine.be_offset_r,
                    trail_trigger_r=strategy_engine.trail_trigger_r,
                    trail_lock_r=strategy_engine.trail_lock_r,
                    maker_fee=maker_f,
                    taker_fee=taker_f,
                    slippage_bps=slip_bps,
                    currency=self.currency
                )

                all_trades = []
                current_eq = self.starting_capital

                for m_num, m_data in monthly_data.items():
                    signals = cached_signals[m_num]
                    if not signals:
                        continue
                    m_trades, _ = sim.simulate_trades(
                        candidates=signals,
                        df_5m=m_data["5m"],
                        starting_equity=current_eq
                    )
                    all_trades.extend(m_trades)
                    if m_trades:
                        current_eq = m_trades[-1].get("equity", m_trades[-1].get("account_equity", current_eq))

                tot_friction = sum(t.get("friction", t.get("total_friction", 0.0)) for t in all_trades)
                net_pnl = current_eq - self.starting_capital
                wins = [t for t in all_trades if t["net_pnl"] > 0]
                losses = [t for t in all_trades if t["net_pnl"] < 0]
                wr = (len(wins) / len(all_trades) * 100.0) if all_trades else 0.0
                win_sum = sum(t["net_pnl"] for t in wins)
                loss_sum = abs(sum(t["net_pnl"] for t in losses))
                pf = (win_sum / loss_sum) if loss_sum > 0 else 1.0

                cell = FrictionMatrixCell(
                    fee_multiplier=fee_mult,
                    maker_fee_pct=maker_f * 100.0,
                    taker_fee_pct=taker_f * 100.0,
                    slippage_bps=slip_bps,
                    final_equity=current_eq,
                    net_profit=net_pnl,
                    total_friction_cost=tot_friction,
                    profit_factor=pf,
                    win_rate=wr,
                    is_profitable=net_pnl > 0
                )
                matrix_results.append(cell)

        be_slip = 0.0
        for cell in matrix_results:
            if cell.fee_multiplier == 1.0:
                if cell.is_profitable and cell.slippage_bps > be_slip:
                    be_slip = cell.slippage_bps

        slip_1x_cells = [c for c in matrix_results if c.fee_multiplier == 1.0]
        if len(slip_1x_cells) >= 2:
            base_pnl = slip_1x_cells[0].net_profit
            high_slip_pnl = slip_1x_cells[-1].net_profit
            delta_slip = slip_1x_cells[-1].slippage_bps - slip_1x_cells[0].slippage_bps
            elasticity_per_bps = (base_pnl - high_slip_pnl) / delta_slip if delta_slip > 0 else 0.0
        else:
            elasticity_per_bps = 0.0

        return {
            "matrix": matrix_results,
            "break_even_slippage_bps_at_1x": be_slip,
            "profit_decay_per_bps_inr": elasticity_per_bps,
            "extreme_5x_15bps_pnl": next((c.net_profit for c in matrix_results if c.fee_multiplier == 5.0 and c.slippage_bps == 15.0), 0.0),
            "extreme_5x_15bps_pf": next((c.profit_factor for c in matrix_results if c.fee_multiplier == 5.0 and c.slippage_bps == 15.0), 0.0)
        }
