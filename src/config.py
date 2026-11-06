from dataclasses import dataclass, field
import json
import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AccountConfig:
    starting_capital_inr: float = 10000.0
    starting_capital_usdt: float = 115.60
    risk_per_trade_pct: float = 0.030
    max_daily_loss_pct: float = 0.10
    max_leverage: int = 10
    max_open_positions: int = 1
    reset_frequency: str = "MONTHLY"


@dataclass
class StrategyConfig:
    strategy_name: str = "STRATEGY_4_MARKOV_1TO15"
    markov_window: int = 10
    markov_thresh: float = 0.02
    ema_fast: int = 20
    ema_slow: int = 50
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    rr_ratio: float = 15.0
    max_hold_bars: int = 120
    min_rvol: float = 1.20
    min_wick_ratio: float = 0.15
    rvol_ma_len: int = 20
    pullback_tolerance_pct: float = 0.002


@dataclass
class GridConfig:
    enabled: bool = True
    range_timeframe: str = "4h"
    range_lookback_bars: int = 100
    min_touches: int = 2
    boundary_tolerance_pct: float = 0.015
    min_range_width_atr: float = 2.0
    max_range_width_atr: float = 12.0
    atr_period: int = 14
    spacing_mode: str = "ATR"
    spacing_atr_multiplier: float = 1.0
    spacing_pct: float = 0.010
    max_grid_levels: int = 8
    capital_fraction: float = 0.50
    max_inventory_contracts: int = 20
    max_consecutive_fills: int = 6
    max_grid_drawdown_pct: float = 0.15
    breakout_buffer_atr: float = 0.50
    breakout_confirmation_tf: str = "4h"
    breakout_action: str = "PAUSE"
    range_refresh_bars: int = 6
    session_filter_enabled: bool = True


@dataclass
class RegimeConfig:
    mode: str = "PURE_MARKOV"
    grid_months: List[int] = field(default_factory=lambda: [])
    normal_months: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    transition_policy: str = "DISABLED"


@dataclass
class SessionConfig:
    filter_enabled: bool = True
    london_start_utc: int = 7
    london_end_utc: int = 16
    ny_start_utc: int = 13
    ny_end_utc: int = 22


@dataclass
class FrictionConfig:
    maker_fee_pct: float = 0.00020
    taker_fee_pct: float = 0.00030
    slippage_bps: float = 1.0


@dataclass
class ApiConfig:
    delta_environment: str = "testnet"
    delta_base_url: str = "https://cdn-ind.testnet.deltaex.org"
    delta_pub_wss: str = "wss://socket-ind-pub.testnet.deltaex.org"
    delta_prv_wss: str = "wss://socket-ind.testnet.deltaex.org"
    delta_api_key: str = ""
    delta_api_secret: str = ""
    trading_enabled: bool = False

    binance_testnet: bool = False
    api_key: str = ""
    api_secret: str = ""


@dataclass
class BotConfig:
    mode: str = "PAPER"
    exchange: str = "DELTA"
    symbol: str = "BTCUSD"
    timeframe: str = "15m"
    currency: str = "INR"
    inr_usdt_rate: float = 86.50
    account: AccountConfig = field(default_factory=AccountConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    poll_interval_sec: int = 5
    save_trade_history: bool = True
    trade_history_file: str = "reports/live_trades.csv"

    @classmethod
    def default(cls) -> "BotConfig":
        return cls()

    @classmethod
    def from_json(cls, file_path: str) -> "BotConfig":
        data = {}
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        account = AccountConfig(**data.get("account", {}))
        strategy = StrategyConfig(**data.get("strategy", {}))
        grid = GridConfig(**data.get("grid", {}))
        regime = RegimeConfig(**data.get("regime", {}))
        session = SessionConfig(**data.get("session", {}))
        friction = FrictionConfig(**data.get("friction", {}))
        api = ApiConfig(**data.get("api", {}))

        return cls(
            mode=data.get("mode", "PAPER"),
            exchange=data.get("exchange", "DELTA"),
            symbol=data.get("symbol", "BTCUSD"),
            timeframe=data.get("timeframe", "15m"),
            currency=data.get("currency", "INR"),
            inr_usdt_rate=data.get("inr_usdt_rate", 86.50),
            account=account,
            strategy=strategy,
            grid=grid,
            regime=regime,
            session=session,
            friction=friction,
            api=api,
            poll_interval_sec=data.get("poll_interval_sec", 5),
            save_trade_history=data.get("save_trade_history", True),
            trade_history_file=data.get("trade_history_file", "reports/live_trades.csv"),
        )
