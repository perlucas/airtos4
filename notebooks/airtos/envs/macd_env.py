import pandas_ta as ta
import numpy as np

from .trading_env import TradingEnv


class MacdEnv(TradingEnv):
    """Trading environment designed to use MACD as feature signals
    """

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]

        # Get the actual prices within observed frame
        # Ensure there are at least window_size ticks before the first observed one
        prices = prices[self.frame_bound[0] -
                        self.window_size: self.frame_bound[1]]

        # Z-score normalization function
        def z_score(values):
            mean = np.mean(values)
            std_dev = np.std(values)
            return (values - mean) / std_dev

        macd = self.df.ta.macd().to_numpy()
        macd = np.where(np.isfinite(macd), macd, 0)
        macd = macd[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        macd_z = z_score(macd)

        return prices.astype(np.float32), macd_z.astype(np.float32)
