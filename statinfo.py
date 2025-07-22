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
            "과매수 상태" if self.rsi_val > 70 else
            "과매도 상태" if self.rsi_val < 30 else
            "중립 상태"
        )
        self.text = (
            f"이 종목은 {self.sector} 섹터에 속하며, 데이터는 하루 단위로 수집되었다. "
            f"{self.start_date}부터 시작하는 최근 20일간 평균 종가는 {self.mean_val:.2f}이며, "
            f"최고가는 {self.max_val:.2f}, 최저가는 {self.min_val:.2f}이다. "
            f"이동평균선은 단순(SMA) {self.sma:.2f}, 가중(WMA) {self.wma:.2f}, 지수(EMA) {self.ema:.2f}로 계산되었다. "
            f"RSI(14)는 {self.rsi_val:.2f}이며, 이는 {rsi_state}를 시사한다. "
            f"볼린저밴드는 중심선 {self.ma_20:.2f}, 상단선 {self.bb_upper:.2f}, 하단선 {self.bb_lower:.2f}로 구성되며, "
            f"이를 통해 가격의 극단값과 변동성 변화를 판단할 수 있다."
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
