from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class FibonacciRetracementEngine:

    FIB_LEVELS = {
        "fib_236": 0.236,
        "fib_382": 0.382,
        "fib_500": 0.500,
        "fib_618": 0.618,
        "fib_786": 0.786
    }

    @staticmethod
    def calculate_bullish_fibs(swing_low: float, swing_high: float) -> Dict[str, float]:
        diff = swing_high - swing_low
        return {
            "swing_low": swing_low,
            "swing_high": swing_high,
            "fib_382": swing_high - (0.382 * diff),
            "fib_500": swing_high - (0.500 * diff),
            "fib_618": swing_high - (0.618 * diff),
            "fib_786": swing_high - (0.786 * diff),
            "golden_pocket_top": swing_high - (0.500 * diff),
            "golden_pocket_bottom": swing_high - (0.618 * diff),
        }

    @staticmethod
    def calculate_bearish_fibs(swing_high: float, swing_low: float) -> Dict[str, float]:
        diff = swing_high - swing_low
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "fib_382": swing_low + (0.382 * diff),
            "fib_500": swing_low + (0.500 * diff),
            "fib_618": swing_low + (0.618 * diff),
            "fib_786": swing_low + (0.786 * diff),
            "golden_pocket_bottom": swing_low + (0.500 * diff),
            "golden_pocket_top": swing_low + (0.618 * diff),
        }

    @staticmethod
    def is_in_golden_pocket(price: float, fib_dict: Dict[str, float], direction: str) -> bool:
        if direction in ["LONG", "BUY"]:
            top = fib_dict["golden_pocket_top"]
            bot = fib_dict["golden_pocket_bottom"]
            return bot <= price <= top
        elif direction in ["SHORT", "SELL"]:
            top = fib_dict["golden_pocket_top"]
            bot = fib_dict["golden_pocket_bottom"]
            return bot <= price <= top
        return False
