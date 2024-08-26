import pandas_ta as ta
import numpy as np

from .trading_env import TradingEnv


class RsiEnv(TradingEnv):
    """Trading environment designed to use RSI as feature signals
    """

    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]

        # Get the actual prices within observed frame
        prices = prices[self.frame_bound[0] - self.window_size: self.frame_bound[1]]

        # Z-score normalization function
        def z_score(values):
            mean = np.mean(values)
            std_dev = np.std(values)
            return (values - mean) / std_dev

        rsi = self.df.ta.rsi().to_numpy()
        rsi = np.where(np.isfinite(rsi), rsi, 0)
        rsi = rsi[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        rsi_z = z_score(rsi)
        

        features = np.column_stack((rsi_z,))
        # Return prices and the features (inputs for the model)
        return prices.astype(np.float32), features.astype(np.float32)


    '''Older method
    def _process_data(self):
        prices = self.df.loc[:, 'Close'].to_numpy()

        # validate index (TODO: Improve validation)
        prices[self.frame_bound[0] - self.window_size]

        # Get the actual prices within observed frame
        # Ensure there are at least window_size ticks before the first observed one
        prices = prices[self.frame_bound[0] -
                        self.window_size: self.frame_bound[1]]

        # Generate indicators
        self.df.ta.log_return(cumulative=True, append=True)
        self.df.ta.percent_return(cumulative=True, append=True)

        rsi = self.df.ta.rsi().to_numpy()
        rsi = np.where(np.isfinite(rsi), rsi, 0)
        rsi = np.divide(rsi, 100)
        rsi = np.column_stack((rsi,))

        return prices.astype(np.float32), rsi.astype(np.float32)
        '''
