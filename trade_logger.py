"""Trade Logger Module - Logs signals to CSV"""
import csv
import os
from datetime import datetime
from typing import List
from base_strategy import Signal
from config import LOG_CONFIG


class TradeLogger:
    """Logs trading signals to CSV files."""
    
    def __init__(self):
        self.signals_file = LOG_CONFIG["signals_csv"]
        self._ensure_dir()
        self._init_csv()
    
    def _ensure_dir(self):
        """Ensure log directory exists."""
        log_dir = os.path.dirname(self.signals_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def _init_csv(self):
        """Initialize CSV with headers if not exists."""
        if not os.path.exists(self.signals_file):
            headers = [
                "timestamp", "pair", "direction", "entry_price",
                "stop_loss", "take_profit", "confidence",
                "strategy", "timeframe"
            ]
            with open(self.signals_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_signal(self, signal: Signal):
        """Log a single signal to CSV."""
        row = [
            signal.timestamp,
            signal.pair,
            signal.direction,
            signal.entry_price,
            signal.stop_loss,
            signal.take_profit,
            signal.confidence,
            signal.strategy,
            signal.timeframe
        ]
        
        with open(self.signals_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_signals(self, signals: List[Signal]):
        """Log multiple signals."""
        for signal in signals:
            self.log_signal(signal)
    
    def get_recent_signals(self, count: int = 10) -> List[dict]:
        """Get recent logged signals."""
        if not os.path.exists(self.signals_file):
            return []
        
        signals = []
        with open(self.signals_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                signals.append(row)
        
        return signals[-count:]
