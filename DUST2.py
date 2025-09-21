"""
Advanced SMC/MMC Trading Bot
Enhanced with multi-timeframe analysis, sophisticated risk management,
advanced order handling, and comprehensive error management.
"""

import argparse
import datetime as dt
import time
import math
import sys
import logging
import json
import os
from typing import List, Dict, Any, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set up logging with a writable directory
log_dir = Path(os.path.expanduser("~")) / "SMC_MMC_Logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "trading_bot.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Attempt to import MetaTrader5 (optional)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not available. Install it for live trading.")

# Telegram (for onboarding and notifications)
try:
    from telegram import Bot
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    Bot = None
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not available. Install it for Telegram notifications.")

# -----------------------
# Data Structures
# -----------------------
class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"

class MarketRegime(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNCLEAR = "UNCLEAR"

@dataclass
class OrderBlock:
    type: str  # 'demand' or 'supply'
    time: dt.datetime
    zone: Tuple[float, float]
    strength: float
    volume: float
    timeframe: str
    validity: bool = True

@dataclass
class TradingSignal:
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    rationale: str
    timestamp: dt.datetime
    order_blocks: List[OrderBlock] = None
    timeframe: str = "M1"

@dataclass
class Position:
    id: int
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    entry_time: dt.datetime
    risk_reward_ratio: float
    status: str = "OPEN"  # OPEN, CLOSED, CANCELLED

# -----------------------
# Configuration Manager
# -----------------------
class ConfigManager:
    def __init__(self, config_file="config.json"):
        self.config_file = config_file
        self.default_config = {
            "risk_management": {
                "max_risk_per_trade": 0.02,  # 2% of account per trade
                "max_daily_risk": 0.05,  # 5% of account daily
                "max_open_trades": 3,
                "min_risk_reward": 1.5,
                "volatility_multiplier": 1.5,
                "equity_protection_stop": 0.8  # Stop trading if equity drops to 80% of starting
            },
            "trading": {
                "allowed_symbols": ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
                "default_timeframe": "M15",
                "analysis_timeframes": ["M5", "M15", "H1"],
                "trading_hours": {
                    "start": "00:00",
                    "end": "23:59",
                    "timezone": "UTC"
                },
                "news_filter": True,
                "high_impact_news_buffer": 30  # minutes before/after high impact news to avoid trading
            },
            "mt5": {
                "server": "Default",
                "login": 123456,
                "password": "password",
                "timeout": 10000,
                "portable": False
            },
            "telegram": {
                "enabled": False,
                "token": "your_telegram_token",
                "chat_id": 123456789,
                "notify_on_signal": True,
                "notify_on_entry": True,
                "notify_on_exit": True,
                "notify_on_error": True
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using default config.")
                return self.default_config
        else:
            logger.info("Config file not found. Using default config.")
            return self.default_config
    
    def save_config(self):
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Config saved successfully.")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value

# -----------------------
# Data Manager
# -----------------------
class DataManager:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.symbols = config.get("trading.allowed_symbols", ["XAUUSD"])
        self.timeframes = config.get("trading.analysis_timeframes", ["M15"])
        self.data = {}
        
    def fetch_mt5_data(self, symbol: str, timeframe: str, bars: int = 1000) -> Optional[pd.DataFrame]:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 not available")
            return None
            
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        if timeframe not in tf_map:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None
            
        try:
            if not mt5.initialize():
                logger.error("MT5 initialize failed")
                return None
                
            rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, bars)
            mt5.shutdown()
            
            if rates is None:
                logger.error(f"Failed to fetch data for {symbol} {timeframe}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'tick_volume': 'volume'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching MT5 data: {e}")
            return None
    
    def fetch_all_data(self):
        for symbol in self.symbols:
            self.data[symbol] = {}
            for tf in self.timeframes:
                self.data[symbol][tf] = self.fetch_mt5_data(symbol, tf)
                if self.data[symbol][tf] is not None:
                    logger.info(f"Fetched {symbol} {tf} data: {len(self.data[symbol][tf])} bars")
                else:
                    logger.warning(f"Failed to fetch {symbol} {tf} data")
    
    def get_symbol_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        if symbol in self.data and timeframe in self.data[symbol]:
            return self.data[symbol][timeframe]
        return None

# -----------------------
# Advanced Technical Indicators
# -----------------------
class AdvancedIndicators:
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum.reduce([high_low, high_close, low_close])
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(df: pd.Series, period: int) -> pd.Series:
        return df.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = df.rolling(window=period).mean()
        std = df.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def calculate_rsi(df: pd.Series, period: int = 14) -> pd.Series:
        delta = df.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        
        tr = np.maximum.reduce([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ])
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_macd(df: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = df.ewm(span=fast, adjust=False).mean()
        ema_slow = df.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap

# -----------------------
# Market Regime Detection
# -----------------------
class MarketRegimeDetector:
    @staticmethod
    def detect_regime(df: pd.DataFrame) -> MarketRegime:
        if len(df) < 50:
            return MarketRegime.UNCLEAR
            
        # Calculate indicators
        adx = AdvancedIndicators.calculate_adx(df)
        rsi = AdvancedIndicators.calculate_rsi(df['close'])
        atr = AdvancedIndicators.calculate_atr(df)
        ema20 = AdvancedIndicators.calculate_ema(df['close'], 20)
        ema50 = AdvancedIndicators.calculate_ema(df['close'], 50)
        
        # Get current values
        current_adx = adx.iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        avg_atr = atr.mean()
        
        # Trend detection
        price_above_ema20 = df['close'].iloc[-1] > ema20.iloc[-1]
        price_above_ema50 = df['close'].iloc[-1] > ema50.iloc[-1]
        ema20_above_ema50 = ema20.iloc[-1] > ema50.iloc[-1]
        
        # Determine regime
        if current_adx > 25:
            if price_above_ema20 and price_above_ema50 and ema20_above_ema50:
                return MarketRegime.TRENDING_UP
            elif not price_above_ema20 and not price_above_ema50 and not ema20_above_ema50:
                return MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.VOLATILE
        elif current_adx < 20:
            if current_atr > avg_atr * 1.5:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.RANGING
        else:
            return MarketRegime.UNCLEAR

# -----------------------
# Enhanced SMC/MMC Strategy
# -----------------------
class EnhancedSMCStrategy:
    def __init__(self, data_manager: DataManager, config: ConfigManager):
        self.dm = data_manager
        self.config = config
        self.regime_detector = MarketRegimeDetector()
        
    def analyze_multi_timeframe(self, symbol: str) -> List[TradingSignal]:
        signals = []
        primary_tf = self.config.get("trading.default_timeframe", "M15")
        
        # Get market regime from higher timeframe
        h1_data = self.dm.get_symbol_data(symbol, "H1")
        if h1_data is not None:
            regime = self.regime_detector.detect_regime(h1_data)
        else:
            regime = MarketRegime.UNCLEAR
            
        # Analyze primary timeframe
        primary_data = self.dm.get_symbol_data(symbol, primary_tf)
        if primary_data is None:
            return signals
            
        # Get order blocks from higher timeframes
        order_blocks = self.find_order_blocks(symbol, primary_tf)
        
        # Look for liquidity sweeps with confirmations
        signals.extend(self.find_liquidity_sweeps(symbol, primary_tf, order_blocks, regime))
        
        # Look for fair value gap trades
        signals.extend(self.find_fvg_trades(symbol, primary_tf, order_blocks, regime))
        
        # Filter signals by confidence and market regime
        filtered_signals = self.filter_signals(signals, regime)
        
        return filtered_signals
    
    def find_order_blocks(self, symbol: str, timeframe: str) -> List[OrderBlock]:
        df = self.dm.get_symbol_data(symbol, timeframe)
        if df is None:
            return []
            
        order_blocks = []
        lookback = 100  # bars to look back for order blocks
        
        for i in range(2, min(len(df), lookback)):
            # Bullish order block (demand zone)
            if (df['close'].iloc[i] > df['open'].iloc[i] and  # Bullish candle
                df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # Bearish follow-up
                df['low'].iloc[i+1] >= df['low'].iloc[i]):  # Didn't break low
                
                strength = (df['close'].iloc[i] - df['open'].iloc[i]) / df['open'].iloc[i]
                ob = OrderBlock(
                    type='demand',
                    time=df.index[i],
                    zone=(df['low'].iloc[i], df['high'].iloc[i]),
                    strength=strength,
                    volume=df['volume'].iloc[i],
                    timeframe=timeframe
                )
                order_blocks.append(ob)
                
            # Bearish order block (supply zone)
            elif (df['close'].iloc[i] < df['open'].iloc[i] and  # Bearish candle
                  df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # Bullish follow-up
                  df['high'].iloc[i+1] <= df['high'].iloc[i]):  # Didn't break high
                
                strength = (df['open'].iloc[i] - df['close'].iloc[i]) / df['open'].iloc[i]
                ob = OrderBlock(
                    type='supply',
                    time=df.index[i],
                    zone=(df['low'].iloc[i], df['high'].iloc[i]),
                    strength=strength,
                    volume=df['volume'].iloc[i],
                    timeframe=timeframe
                )
                order_blocks.append(ob)
                
        return order_blocks
    
    def find_liquidity_sweeps(self, symbol: str, timeframe: str, 
                             order_blocks: List[OrderBlock], regime: MarketRegime) -> List[TradingSignal]:
        df = self.dm.get_symbol_data(symbol, timeframe)
        if df is None:
            return []
            
        signals = []
        lookback = 50
        
        for i in range(lookback, len(df) - 1):
            # Check for liquidity sweep
            is_sweep, sweep_type = self.is_liquidity_sweep(df, i)
            
            if not is_sweep:
                continue
                
            # Find nearest order block
            nearest_ob = self.find_nearest_order_block(df.index[i], order_blocks)
            
            # Check if sweep is confirmed
            confirmed, confirmation_type = self.is_sweep_confirmed(df, i, sweep_type, nearest_ob)
            
            if confirmed:
                # Generate signal
                signal = self.generate_sweep_signal(df, i, sweep_type, nearest_ob, 
                                                  confirmation_type, regime, timeframe)
                if signal:
                    signals.append(signal)
                    
        return signals
    
    def is_liquidity_sweep(self, df: pd.DataFrame, index: int) -> Tuple[bool, Optional[str]]:
        if index < 20 or index >= len(df) - 1:
            return False, None
            
        current = df.iloc[index]
        prev_high = df['high'].iloc[index-20:index].max()
        prev_low = df['low'].iloc[index-20:index].min()
        
        # Bullish sweep (sweep of lows)
        if (current['low'] < prev_low and 
            current['close'] > (current['high'] + current['low']) / 2 and
            current['volume'] > df['volume'].iloc[index-20:index].mean() * 1.5):
            return True, 'BULLISH'
            
        # Bearish sweep (sweep of highs)
        if (current['high'] > prev_high and 
            current['close'] < (current['high'] + current['low']) / 2 and
            current['volume'] > df['volume'].iloc[index-20:index].mean() * 1.5):
            return True, 'BEARISH'
            
        return False, None
    
    def is_sweep_confirmed(self, df: pd.DataFrame, index: int, sweep_type: str, 
                          order_block: Optional[OrderBlock]) -> Tuple[bool, str]:
        if index >= len(df) - 2:
            return False, "NO_CONFIRMATION"
            
        next_candle = df.iloc[index+1]
        
        if sweep_type == 'BULLISH':
            # Check for bullish confirmation candle
            if (next_candle['close'] > next_candle['open'] and
                next_candle['close'] > df['close'].iloc[index]):
                return True, "BULLISH_CONFIRMATION"
                
            # Check for order block confirmation
            if (order_block and order_block.type == 'demand' and
                next_candle['low'] >= order_block.zone[0] and
                next_candle['high'] <= order_block.zone[1]):
                return True, "ORDER_BLOCK_CONFIRMATION"
                
        elif sweep_type == 'BEARISH':
            # Check for bearish confirmation candle
            if (next_candle['close'] < next_candle['open'] and
                next_candle['close'] < df['close'].iloc[index]):
                return True, "BEARISH_CONFIRMATION"
                
            # Check for order block confirmation
            if (order_block and order_block.type == 'supply' and
                next_candle['low'] >= order_block.zone[0] and
                next_candle['high'] <= order_block.zone[1]):
                return True, "ORDER_BLOCK_CONFIRMATION"
                
        return False, "NO_CONFIRMATION"
    
    def generate_sweep_signal(self, df: pd.DataFrame, index: int, sweep_type: str,
                            order_block: Optional[OrderBlock], confirmation_type: str,
                            regime: MarketRegime, timeframe: str) -> Optional[TradingSignal]:
        entry_candle = df.iloc[index+1]
        atr = AdvancedIndicators.calculate_atr(df.iloc[:index+1]).iloc[-1]
        
        if sweep_type == 'BULLISH':
            direction = TradeDirection.LONG
            entry_price = entry_candle['close']
            stop_loss = entry_candle['low'] - atr * 0.5
            take_profit = entry_price + (entry_price - stop_loss) * 2  # 1:2 RR
            
            # Check if regime is compatible
            if regime == MarketRegime.TRENDING_DOWN:
                confidence = 0.4
            elif regime == MarketRegime.TRENDING_UP:
                confidence = 0.8
            else:
                confidence = 0.6
                
        elif sweep_type == 'BEARISH':
            direction = TradeDirection.SHORT
            entry_price = entry_candle['close']
            stop_loss = entry_candle['high'] + atr * 0.5
            take_profit = entry_price - (stop_loss - entry_price) * 2  # 1:2 RR
            
            # Check if regime is compatible
            if regime == MarketRegime.TRENDING_UP:
                confidence = 0.4
            elif regime == MarketRegime.TRENDING_DOWN:
                confidence = 0.8
            else:
                confidence = 0.6
        else:
            return None
            
        # Adjust confidence based on confirmation type
        if confirmation_type == "ORDER_BLOCK_CONFIRMATION":
            confidence *= 1.2  # Boost confidence for OB confirmation
            
        rationale = f"{sweep_type} liquidity sweep with {confirmation_type} on {timeframe}. Market regime: {regime.value}"
        
        return TradingSignal(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=min(confidence, 0.95),  # Cap at 95%
            rationale=rationale,
            timestamp=df.index[index+1],
            order_blocks=[order_block] if order_block else [],
            timeframe=timeframe
        )
    
    def find_fvg_trades(self, symbol: str, timeframe: str, 
                       order_blocks: List[OrderBlock], regime: MarketRegime) -> List[TradingSignal]:
        # Implementation for FVG trades
        # This would be similar to the sweep detection but for Fair Value Gaps
        return []  # Placeholder
    
    def find_nearest_order_block(self, timestamp: dt.datetime, 
                               order_blocks: List[OrderBlock]) -> Optional[OrderBlock]:
        if not order_blocks:
            return None
            
        nearest = None
        min_diff = float('inf')
        
        for ob in order_blocks:
            diff = abs((timestamp - ob.time).total_seconds())
            if diff < min_diff and diff <= 86400:  # Within 24 hours
                min_diff = diff
                nearest = ob
                
        return nearest
    
    def filter_signals(self, signals: List[TradingSignal], regime: MarketRegime) -> List[TradingSignal]:
        filtered = []
        
        for signal in signals:
            # Filter by confidence
            if signal.confidence < 0.5:
                continue
                
            # Filter by market regime compatibility
            if (regime == MarketRegime.TRENDING_UP and signal.direction == TradeDirection.SHORT):
                signal.confidence *= 0.7  # Reduce confidence for counter-trend
            elif (regime == MarketRegime.TRENDING_DOWN and signal.direction == TradeDirection.LONG):
                signal.confidence *= 0.7  # Reduce confidence for counter-trend
                
            # Filter by risk-reward ratio
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            if risk == 0 or reward/risk < 1.2:  # Minimum 1.2:1 RR
                continue
                
            filtered.append(signal)
            
        return filtered

# -----------------------
# Advanced Risk Manager
# -----------------------
class AdvancedRiskManager:
    def __init__(self, config: ConfigManager, account_balance: float):
        self.config = config
        self.balance = account_balance
        self.equity = account_balance
        self.initial_balance = account_balance
        self.open_positions = []
        self.daily_pnl = 0
        self.last_reset = dt.datetime.now().date()
        
    def update_equity(self, current_equity: float):
        self.equity = current_equity
        
        # Reset daily PNL if it's a new day
        current_date = dt.datetime.now().date()
        if current_date != self.last_reset:
            self.daily_pnl = 0
            self.last_reset = current_date
    
    def can_open_trade(self, symbol: str, risk_amount: float) -> Tuple[bool, str]:
        # Check max open trades
        max_trades = self.config.get("risk_management.max_open_trades", 3)
        if len(self.open_positions) >= max_trades:
            return False, f"Max open trades ({max_trades}) reached"
            
        # Check max risk per trade
        max_risk_per_trade = self.config.get("risk_management.max_risk_per_trade", 0.02)
        if risk_amount > self.equity * max_risk_per_trade:
            return False, f"Risk amount ({risk_amount}) exceeds max per trade ({self.equity * max_risk_per_trade})"
            
        # Check max daily risk
        max_daily_risk = self.config.get("risk_management.max_daily_risk", 0.05)
        if abs(self.daily_pnl) > self.equity * max_daily_risk:
            return False, f"Daily PNL ({self.daily_pnl}) exceeds max daily risk ({self.equity * max_daily_risk})"
            
        # Check equity protection
        equity_stop = self.config.get("risk_management.equity_protection_stop", 0.8)
        if self.equity < self.initial_balance * equity_stop:
            return False, f"Equity ({self.equity}) below protection level ({self.initial_balance * equity_stop})"
            
        # Check if symbol has too much exposure
        symbol_exposure = sum(1 for p in self.open_positions if p.symbol == symbol)
        if symbol_exposure >= 2:  # Max 2 positions per symbol
            return False, f"Max exposure to {symbol} reached"
            
        return True, "OK"
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              symbol: str) -> Tuple[float, float]:
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0, 0
            
        # Calculate max risk amount
        max_risk_pct = self.config.get("risk_management.max_risk_per_trade", 0.02)
        max_risk_amount = self.equity * max_risk_pct
        
        # Calculate position size
        position_size = max_risk_amount / risk_per_share
        
        # Adjust for lot size constraints
        # This is simplified - in reality, you'd need to consider lot sizes, margin, etc.
        position_size = round(position_size, 2)
        
        return position_size, max_risk_amount
    
    def add_position(self, position: Position):
        self.open_positions.append(position)
    
    def remove_position(self, position_id: int, pnl: float = 0):
        self.open_positions = [p for p in self.open_positions if p.id != position_id]
        self.daily_pnl += pnl

# -----------------------
# Advanced MT5 Executor
# -----------------------
class AdvancedMT5Executor:
    def __init__(self, config: ConfigManager, risk_manager: AdvancedRiskManager):
        self.config = config
        self.risk_manager = risk_manager
        self.connected = False
        self.symbol_info_cache = {}
        
    def connect(self) -> bool:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 not available")
            return False
            
        try:
            mt5_config = self.config.get("mt5", {})
            initialized = mt5.initialize(
                login=mt5_config.get("login", 0),
                password=mt5_config.get("password", ""),
                server=mt5_config.get("server", ""),
                timeout=mt5_config.get("timeout", 10000),
                portable=mt5_config.get("portable", False)
            )
            
            if initialized:
                self.connected = True
                logger.info("Connected to MT5")
                return True
            else:
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False
    
    def disconnect(self):
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def execute_trade(self, signal: TradingSignal, symbol: str) -> Optional[Position]:
        if not self.connected:
            logger.error("Not connected to MT5")
            return None
            
        # Calculate position size
        position_size, risk_amount = self.risk_manager.calculate_position_size(
            signal.entry_price, signal.stop_loss, symbol
        )
        
        # Check if we can open the trade
        can_open, reason = self.risk_manager.can_open_trade(symbol, risk_amount)
        if not can_open:
            logger.warning(f"Cannot open trade: {reason}")
            return None
            
        # Prepare order request
        order_type = mt5.ORDER_TYPE_BUY if signal.direction == TradeDirection.LONG else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if signal.direction == TradeDirection.LONG else mt5.symbol_info_tick(symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position_size,
            "type": order_type,
            "price": price,
            "sl": signal.stop_loss,
            "tp": signal.take_profit,
            "deviation": 20,
            "magic": 123456,
            "comment": f"SMC_MMC_{signal.timeframe}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Calculate risk-reward ratio
            risk = abs(signal.entry_price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            position = Position(
                id=result.order,
                symbol=symbol,
                direction=signal.direction,
                entry_price=price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                size=position_size,
                entry_time=dt.datetime.now(),
                risk_reward_ratio=rr_ratio
            )
            
            self.risk_manager.add_position(position)
            logger.info(f"Trade executed: {symbol} {signal.direction.value} {position_size} lots")
            
            return position
        else:
            logger.error(f"Order failed: {result.retcode} - {mt5.last_error()}")
            return None
    
    def check_positions(self):
        if not self.connected:
            return
            
        # Get all open positions
        positions = mt5.positions_get()
        
        for pos in positions:
            # Check if position needs to be updated in risk manager
            # This is a simplified implementation
            pass

# -----------------------
# Telegram Notifier
# -----------------------
class TelegramNotifier:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.bot = None
        self.enabled = config.get("telegram.enabled", False)
        
        if self.enabled and TELEGRAM_AVAILABLE:
            try:
                self.bot = Bot(token=config.get("telegram.token"))
                self.chat_id = config.get("telegram.chat_id")
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.enabled = False
    
    def send_message(self, message: str):
        if not self.enabled or not self.bot:
            return
            
        try:
            self.bot.send_message(chat_id=self.chat_id, text=message)
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
    
    def notify_signal(self, signal: TradingSignal, symbol: str):
        if not self.config.get("telegram.notify_on_signal", True):
            return
            
        message = (
            f"🚨 Trading Signal - {symbol}\n"
            f"Direction: {signal.direction.value}\n"
            f"Entry: {signal.entry_price:.5f}\n"
            f"SL: {signal.stop_loss:.5f}\n"
            f"TP: {signal.take_profit:.5f}\n"
            f"Confidence: {signal.confidence:.2f}\n"
            f"Timeframe: {signal.timeframe}\n"
            f"Rationale: {signal.rationale}\n"
            f"Time: {signal.timestamp}"
        )
        
        self.send_message(message)
    
    def notify_trade(self, position: Position, signal: TradingSignal):
        if not self.config.get("telegram.notify_on_entry", True):
            return
            
        message = (
            f"✅ Trade Executed - {position.symbol}\n"
            f"Direction: {position.direction.value}\n"
            f"Entry: {position.entry_price:.5f}\n"
            f"Size: {position.size:.2f} lots\n"
            f"SL: {position.stop_loss:.5f}\n"
            f"TP: {position.take_profit:.5f}\n"
            f"RR: {position.risk_reward_ratio:.2f}\n"
            f"Time: {position.entry_time}"
        )
        
        self.send_message(message)
    
    def notify_error(self, error: str):
        if not self.config.get("telegram.notify_on_error", True):
            return
            
        message = f"❌ Error: {error}"
        self.send_message(message)

# -----------------------
# Main Trading Bot
# -----------------------
class SMCMMCTradingBot:
    def __init__(self, config_file="config.json"):
        self.config = ConfigManager(config_file)
        self.data_manager = DataManager(self.config)
        self.strategy = EnhancedSMCStrategy(self.data_manager, self.config)
        self.risk_manager = AdvancedRiskManager(self.config, 10000)  # Default balance
        self.mt5_executor = AdvancedMT5Executor(self.config, self.risk_manager)
        self.telegram_notifier = TelegramNotifier(self.config)
        self.running = False
        
    def initialize(self):
        """Initialize the trading bot"""
        logger.info("Initializing trading bot...")
        
        # Connect to MT5
        if not self.mt5_executor.connect():
            logger.error("Failed to connect to MT5")
            return False
            
        # Get account balance
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return False
            
        self.risk_manager = AdvancedRiskManager(self.config, account_info.balance)
        logger.info(f"Account balance: ${account_info.balance:.2f}")
        
        # Fetch market data
        self.data_manager.fetch_all_data()
        
        logger.info("Trading bot initialized successfully")
        return True
    
    def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("Starting trading bot...")
        
        try:
            while self.running:
                # Check if we're within trading hours
                if not self.is_trading_hours():
                    logger.info("Outside trading hours. Sleeping...")
                    time.sleep(300)  # Sleep for 5 minutes
                    continue
                
                # Update market data
                self.data_manager.fetch_all_data()
                
                # Check open positions
                self.mt5_executor.check_positions()
                
                # Generate signals for each symbol
                for symbol in self.config.get("trading.allowed_symbols", ["XAUUSD"]):
                    signals = self.strategy.analyze_multi_timeframe(symbol)
                    
                    for signal in signals:
                        # Notify about signal
                        self.telegram_notifier.notify_signal(signal, symbol)
                        
                        # Execute trade if confidence is high enough
                        if signal.confidence > 0.7:
                            position = self.mt5_executor.execute_trade(signal, symbol)
                            if position:
                                self.telegram_notifier.notify_trade(position, signal)
                
                # Sleep until next cycle
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Shutting down by user request...")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            self.telegram_notifier.notify_error(str(e))
        finally:
            self.shutdown()
    
    def is_trading_hours(self) -> bool:
        """Check if current time is within trading hours"""
        trading_hours = self.config.get("trading.trading_hours", {})
        if not trading_hours:
            return True  # Always trade if no hours specified
            
        try:
            now = dt.datetime.now()
            start_time = dt.datetime.strptime(trading_hours.get("start", "00:00"), "%H:%M").time()
            end_time = dt.datetime.strptime(trading_hours.get("end", "23:59"), "%H:%M").time()
            
            return start_time <= now.time() <= end_time
        except Exception as e:
            logger.error(f"Error checking trading hours: {e}")
            return True
    
    def shutdown(self):
        """Shut down the trading bot"""
        logger.info("Shutting down trading bot...")
        self.running = False
        self.mt5_executor.disconnect()
        logger.info("Trading bot shut down successfully")

# -----------------------
# CLI Interface
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced SMC/MMC Trading Bot")
    parser.add_argument('--mode', choices=['run', 'backtest', 'optimize', 'monitor'], default='run')
    parser.add_argument('--config', default='config.json', help='Path to config file')
    parser.add_argument('--symbol', help='Specific symbol to trade')
    parser.add_argument('--timeframe', help='Specific timeframe to analyze')
    parser.add_argument('--balance', type=float, default=10000, help='Starting balance for backtesting')
    parser.add_argument('--days', type=int, default=30, help='Days of data for backtesting')
    
    # For MT5, handle additional arguments that might be passed
    parser.add_argument('mt5_symbol', nargs='?', help=argparse.SUPPRESS)
    parser.add_argument('mt5_timeframe', nargs='?', help=argparse.SUPPRESS)
    
    # Use parse_known_args to ignore any additional arguments
    args, unknown = parser.parse_known_args()
    
    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")
    
    # Initialize bot
    bot = SMCMMCTradingBot(args.config)
    
    if args.mode == 'run':
        if bot.initialize():
            bot.run()
        else:
            logger.error("Failed to initialize bot")
            sys.exit(1)
    elif args.mode == 'backtest':
        # Implement backtesting functionality
        logger.info("Backtesting mode not yet implemented")
    elif args.mode == 'optimize':
        # Implement optimization functionality
        logger.info("Optimization mode not yet implemented")
    elif args.mode == 'monitor':
        # Implement monitoring functionality
        logger.info("Monitoring mode not yet implemented")

if __name__ == '__main__':
    main()