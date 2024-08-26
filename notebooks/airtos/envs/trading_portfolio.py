class TradingPortfolio:

    def __init__(self):
        self._shorts = []
        self._longs = []

    def reset(self):
        '''Reset the porftolio removing all longs and shorts
        '''
        self._shorts = []
        self._longs = []

    def open_long(self, price):
        '''Open a new long position or close the oldest short currently active
        Returns the profit made by this operation if any
        '''
        assert len(self._shorts) * len(self._longs) == 0
        if len(self._shorts) > 0:
            oldest_short_price = self._shorts.pop(0) # Remove and close short if any
            return oldest_short_price - price
        # Open long otherwise
        self._longs.append(price)
        return 0

    def open_short(self, price):
        '''Open a new short position or close the oldest long currently active
        Returns the profit made by this operation if any
        '''
        assert len(self._shorts) * len(self._longs) == 0
        if len(self._longs) > 0:
            oldest_long_price = self._longs.pop(0) # Remove and close long if any
            return price - oldest_long_price
        # Open short otherwise
        self._shorts.append(price)
        return 0