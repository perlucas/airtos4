import pandas as pd
import pandas_ta as ta
import numpy as np

from .trading_env import TradingEnv


class CombinedEnv(TradingEnv):
    """Trading environment designed to use a set of indicators as feature signals"""

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
            if std_dev == 0:
                return np.zeros_like(values)
            return ((values - mean) / std_dev) / 3 # Divide by 3 to keep values between -1 and 1

        # Compute 1st Indicator: Moving Averages
        ma1 = self.df.ta.ema(length=20).to_numpy()
        ma1 = np.where(np.isfinite(ma1), ma1, 0)
        ma1 = ma1[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        ma1_z = z_score(ma1)

        ma2 = self.df.ta.ema(length=50).to_numpy()
        ma2 = np.where(np.isfinite(ma2), ma2, 0)
        ma2 = ma2[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        ma2_z = z_score(ma2)

        ma3 = self.df.ta.ema(length=100).to_numpy()
        ma3 = np.where(np.isfinite(ma3), ma3, 0)
        ma3 = ma3[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        ma3_z = z_score(ma3)

        # Compute 2nd Indicator: RSI
        rsi = self.df.ta.rsi().to_numpy()
        rsi = np.where(np.isfinite(rsi), rsi, 0)
        rsi = rsi[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        rsi_z = z_score(rsi)

        # Compute 3rd Indicator: ADX
        adx = self.df.ta.adx().to_numpy()
        adx = np.where(np.isfinite(adx), adx, 0)
        adx = adx[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        adx_z = z_score(adx)

        # Compute 4th Indicator: MACD
        macd = self.df.ta.macd().to_numpy()
        macd = np.where(np.isfinite(macd), macd, 0)
        macd = macd[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        macd_z = z_score(macd)

        # Compute 5th Indicator: Volume
        volume = self.df.loc[:, 'Volume'].to_numpy()
        volume = volume[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        volume_z = z_score(volume)

        features = np.column_stack((ma1_z, ma2_z, ma3_z, rsi_z, adx_z, macd_z, volume_z))

        # Compute the custom TP/SL convenience indicators
        # Compute X: percentage change from previous price
        X = self.df.loc[:, 'Close'].pct_change()
        X.iloc[0] = 0   # First value will be 0, since there is no previous value

        # Define parameters for the algorithm
        TP = 0.1      # Take Profit (10%)
        SL = 0.05      # Stop Loss (5%)
        W = 15    # Window size (look ahead W ticks)
        MAX_OFFSET = len(self.df)  # Maximum offset (use the length of the data as the limit)
        
        # Algorithm for calculating the 'B' indicator
        def calculate_B(i, X, TP, SL, W, MAX_OFFSET):
            t = 1
            for j in range(i + 1, min(i + W, MAX_OFFSET)):  # Loop within window size or MAX_OFFSET
                t *= (1 + X[j])
                if t - 1 >= TP or t - 1 <= -SL:
                    return t - 1  # Return the profit/loss if the threshold is met
            return t - 1  # If the loop completes, return the final value of t - 1
        
        # Algorithm for calculating the 'S' indicator
        def calculate_S(i, X, TP, SL, W, MAX_OFFSET):
            t = 1
            for j in range(i + 1, min(i + W, MAX_OFFSET)):  # Loop within window size or MAX_OFFSET
                t *= (1 - X[j])
                if t - 1 >= TP or t - 1 <= -SL:
                    return t - 1  # Return the profit/loss if the threshold is met
            return t - 1  # If the loop completes, return the final value of t - 1

        # Apply the function to calculate the 'B' and 'S' indicators for each index
        B = []
        S = []
        for idx, _ in X.items():
            B.append(calculate_B(idx, X, TP, SL, W, MAX_OFFSET))
            S.append(calculate_S(idx, X, TP, SL, W, MAX_OFFSET))

        # Convert the result into a pandas Series (if needed)
        self.B = pd.Series(B, index=X.index).to_numpy()[self.frame_bound[0] - self.window_size: self.frame_bound[1]]
        self.S = pd.Series(S, index=X.index).to_numpy()[self.frame_bound[0] - self.window_size: self.frame_bound[1]]

        # Return prices and the features (inputs for the model)
        return prices.astype(np.float32), features.astype(np.float32)
    
    def compute_step_reward(self, _, current_tick, action):
        if action == self.ACTION_BUY:
            return self.B[current_tick] * 100
        elif action == self.ACTION_SELL:
            return self.S[current_tick] * 100
        else:
            return 0