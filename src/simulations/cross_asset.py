from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..exchange.delta_client import DeltaExchangeClient
from ..indicators.price_action_patterns import PriceActionPatterns
from ..indicators.trend_engine import TrendEngine
from ..indicators.key_levels_sr import KeyLevelsEngine
from ..strategy.scalping_engine import ScalpingEngine
from ..execution.scalping_simulator import ScalpingSimulator

logger = logging.getLogger("Trader.CrossAsset")


@dataclass
class TokenMatrixResult:
    symbol: str
    total_trades: int
    win_rate: float
    net_profit_base: float
    return_pct: float
    profit_factor: float
    trailed_wins_count: int
    tp2_runners_count: int
    long_trades: int
    short_trades: int
    average_trade_duration_mins: float
    edge_confirmed: bool


class CrossAssetMatrixEngine:

    TARGET_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "DOGEUSD", "ADAUSD"]

    def __init__(
        self,
        client: Optional[DeltaExchangeClient] = None,
        starting_capital: float = 10000.0,
        risk_pct: float = 0.040,
        currency: str = "INR"
    ):
        self.client = client or DeltaExchangeClient(environment="testnet", trading_enabled=False)
        self.starting_capital = starting_capital
        self.risk_pct = risk_pct
        self.currency = currency

    def fetch_token_dataset(self, symbol: str, limit_5m: int = 2000) -> Optional[Dict[str, pd.DataFrame]]:
        try:
            logger.info(f"Fetching multi-timeframe candles for {symbol} from Delta Exchange...")
            df_1d = self.client.get_candles(symbol=symbol, resolution="1d", limit=180)
            if df_1d.empty or len(df_1d) < 20:
                logger.warning(f"Insufficient 1D data for {symbol}")
                return None
            df_1d = TrendEngine.calculate_trend_indicators(df_1d)
            df_1d = PriceActionPatterns.compute_candle_metrics(df_1d)

            df_4h = self.client.get_candles(symbol=symbol, resolution="4h", limit=300)
            if df_4h.empty or len(df_4h) < 20:
                logger.warning(f"Insufficient 4H data for {symbol}")
                return None
            df_4h = TrendEngine.calculate_trend_indicators(df_4h)
            df_4h = PriceActionPatterns.compute_candle_metrics(df_4h)

            df_15m = self.client.get_candles(symbol=symbol, resolution="15m", limit=800)
            if df_15m.empty or len(df_15m) < 20:
                logger.warning(f"Insufficient 15M data for {symbol}")
                return None
            df_15m = TrendEngine.calculate_trend_indicators(df_15m)
            df_15m = PriceActionPatterns.detect_patterns(df_15m)

            df_5m = self.client.get_candles(symbol=symbol, resolution="5m", limit=limit_5m)
            if df_5m.empty or len(df_5m) < 50:
                logger.warning(f"Insufficient 5M data for {symbol}")
                return None
            df_5m = PriceActionPatterns.detect_patterns(df_5m)
            tr_5m = np.maximum(
                df_5m["high"] - df_5m["low"],
                np.maximum(
                    np.abs(df_5m["high"] - df_5m["close"].shift(1)),
                    np.abs(df_5m["low"] - df_5m["close"].shift(1))
                )
            )
            df_5m["atr14"] = tr_5m.rolling(window=14, min_periods=1).mean()

            return {
                "1d": df_1d,
                "4h": df_4h,
                "15m": df_15m,
                "5m": df_5m,
                "full_1d": df_1d,
                "full_4h": df_4h
            }
        except Exception as e:
            logger.error(f"Failed to fetch dataset for {symbol}: {e}")
            return None

    def evaluate_token(
        self,
        symbol: str,
        dataset: Dict[str, pd.DataFrame],
        strategy_engine: Optional[ScalpingEngine] = None
    ) -> TokenMatrixResult:
        if strategy_engine is None:
            strategy_engine = ScalpingEngine(risk_pct=self.risk_pct)

        signals = strategy_engine.generate_monthly_signals(dataset)
        if not signals:
            return TokenMatrixResult(
                symbol=symbol,
                total_trades=0,
                win_rate=0.0,
                net_profit_base=0.0,
                return_pct=0.0,
                profit_factor=1.0,
                trailed_wins_count=0,
                tp2_runners_count=0,
                long_trades=0,
                short_trades=0,
                average_trade_duration_mins=0.0,
                edge_confirmed=False
            )

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
            currency=self.currency
        )

        trades, _ = sim.simulate_trades(
            candidates=signals,
            df_5m=dataset["5m"],
            starting_equity=self.starting_capital
        )

        if not trades:
            return TokenMatrixResult(
                symbol=symbol,
                total_trades=0,
                win_rate=0.0,
                net_profit_base=0.0,
                return_pct=0.0,
                profit_factor=1.0,
                trailed_wins_count=0,
                tp2_runners_count=0,
                long_trades=0,
                short_trades=0,
                average_trade_duration_mins=0.0,
                edge_confirmed=False
            )

        n_trades = len(trades)
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] < 0]
        wr = (len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0

        win_sum = sum(t["net_pnl"] for t in wins)
        loss_sum = abs(sum(t["net_pnl"] for t in losses))
        pf = (win_sum / loss_sum) if loss_sum > 0 else (win_sum if win_sum > 0 else 1.0)

        final_eq = trades[-1].get("equity", trades[-1].get("account_equity", self.starting_capital))
        net_profit = final_eq - self.starting_capital
        ret_pct = (net_profit / self.starting_capital) * 100.0

        trailed_cnt = sum(1 for t in trades if "TRAIL" in str(t.get("outcome", t.get("exit_reason", ""))))
        tp2_cnt = sum(1 for t in trades if "TP2" in str(t.get("outcome", t.get("exit_reason", ""))))
        longs = sum(1 for t in trades if t.get("direction") == "LONG")
        shorts = sum(1 for t in trades if t.get("direction") == "SHORT")
        durations = [t.get("bars_held", 1) * 5 for t in trades]
        avg_dur = float(np.mean(durations)) if durations else 0.0

        edge_ok = (wr >= 50.0 and pf >= 1.25) or (net_profit > 0 and pf >= 1.20)

        return TokenMatrixResult(
            symbol=symbol,
            total_trades=n_trades,
            win_rate=wr,
            net_profit_base=net_profit,
            return_pct=ret_pct,
            profit_factor=pf,
            trailed_wins_count=trailed_cnt,
            tp2_runners_count=tp2_cnt,
            long_trades=longs,
            short_trades=shorts,
            average_trade_duration_mins=avg_dur,
            edge_confirmed=edge_ok
        )

    def run_matrix(self, symbols: Optional[List[str]] = None) -> List[TokenMatrixResult]:
        sym_list = symbols or self.TARGET_SYMBOLS
        results = []

        for sym in sym_list:
            ds = self.fetch_token_dataset(sym)
            if ds is not None:
                res = self.evaluate_token(sym, ds)
                results.append(res)
                logger.info(f"[{sym}] Trades: {res.total_trades} | WR: {res.win_rate:.1f}% | PF: {res.profit_factor:.2f} | Net PnL: ₹{res.net_profit_base:+,.2f}")
            else:
                logger.warning(f"Skipping {sym} due to lack of feed.")

        return results
