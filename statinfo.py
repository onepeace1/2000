import pandas as pd
import numpy as np

class PatchAnalyzer:
    def __init__(self, patch: pd.DataFrame):
        self.patch = patch
        self.close = patch["Close_z"]
        self.sector = patch["Sector"].iloc[0]
        self.ticker = patch["Ticker"].iloc[0]
        self.start_date = patch["Date"].iloc[0]

    def compute_basic_stats(self):
        self.mean_val = self.close.mean()
        self.max_val = self.close.max()
        self.min_val = self.close.min()

    def compute_moving_averages(self):
        self.sma = self.close.mean()
        weights = np.arange(1, 21)
        self.wma = (self.close.values * weights).sum() / weights.sum()
        self.ema = self.close.ewm(span=30, adjust=False).mean().iloc[-1]

    def compute_rsi(self):
        delta = self.close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(14).mean()
        roll_down = down.rolling(14).mean()
        rs = roll_up / roll_down
        self.rsi_val = (100 - (100 / (1 + rs))).iloc[-1]

    def compute_bollinger_bands(self):
        self.ma_20 = self.close.rolling(20).mean().iloc[-1]
        self.std_20 = self.close.rolling(20).std().iloc[-1]
        self.bb_upper = self.ma_20 + 2 * self.std_20
        self.bb_lower = self.ma_20 - 2 * self.std_20

    def generate_text(self):
        rsi_state = (
            "Overbought condition" if self.rsi_val > 70 else
            "Oversold condition" if self.rsi_val < 30 else
            "neutrality"
        )
        self.text = (
            f"The stock belongs to the {sector} sector and the data was collected on a daily basis. "
            f"This patch covers a 20-day window starting from {start_date}. "
            f"During this period, the average normalized closing price was {mean_val:.2f}, "
            f"with a maximum of {max_val:.2f} and a minimum of {min_val:.2f}. "
            f"Trend indicators were calculated as follows: the Simple Moving Average (SMA) is {sma:.2f}, "
            f"the Weighted Moving Average (WMA) is {wma:.2f}, and the Exponential Moving Average (EMA) is {ema:.2f}. "
            f"Momentum was assessed using the 14-day Relative Strength Index (RSI), which is {rsi_val:.2f}. "
            f"Volatility was evaluated using Bollinger Bands, with a middle band (20-day SMA) at {ma_20:.2f}, "
            f"an upper band at {bb_upper:.2f}, and a lower band at {bb_lower:.2f}. "
            f"These indicators together describe the recent market behavior of the stock, "
            f"providing insight into potential trends, reversals, and volatility conditions."
        )

    def get_summary(self):
        self.compute_basic_stats()
        self.compute_moving_averages()
        self.compute_rsi()
        self.compute_bollinger_bands()
        self.generate_text()

        return {
            "Ticker": self.ticker,
            "Start_Date": self.start_date,
            "Sector": self.sector,
            "Statistical_Text": self.text
        }
