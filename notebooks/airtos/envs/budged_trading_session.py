class BudgedTradingSession:

    def __init__(self, fee = 0, initial_budget = 1000, stop_loss = None):
        self._shorts = []
        self._longs = []

        assert initial_budget > 0
        self._initial_budget = initial_budget
        self.budget = initial_budget

        assert fee >= 0
        self._fee = fee

        self._stop_loss = stop_loss

    def reset(self):
        '''Reset the porftolio removing all longs and shorts
        '''
        self._shorts = []
        self._longs = []
        self.budget = self._initial_budget

    def __has_shorts(self):
        return len(self._shorts) > 0

    def __oldest_short(self):
        return self._shorts[0]

    def __remove_oldest_short(self):
        return self._shorts.pop(0)

    def __overwrite_oldest_short(self, values):
        self._shorts[0] = values

    def __add_long(self, values):
        self._longs.append(values)

    def __has_longs(self):
        return len(self._longs) > 0

    def __oldest_long(self):
        return self._longs[0]

    def __remove_oldest_long(self):
        return self._longs.pop(0)

    def __overwrite_oldest_long(self, values):
        self._longs[0] = values

    def __add_short(self, values):
        self._shorts.append(values)

    def open_long(self, price, num_shares):
        '''Open a new long position or close the oldest short currently active that matches the number of shares.
        Returns the profit made by this operation when closing one or more shorts.
        '''
        assert len(self._shorts) * len(self._longs) == 0 # We cannot have both open shorts and longs
        
        remaining_to_buy = min(self.budget // price, num_shares)

        discount = price * remaining_to_buy * (self._fee/100) # Apply fee
        profit = 0

        while remaining_to_buy > 0 and self.__has_shorts():
            short = self.__oldest_short()
            short_price, short_shares = short

            if short_shares == remaining_to_buy:
                # if oldest short has the same num of shares that this long needs, close it and compute profit
                self.__remove_oldest_short()
                profit += (short_price - price) * remaining_to_buy
                remaining_to_buy = 0
                break
            elif short_shares < remaining_to_buy:
                # if oldest short's shares is less than the required, close it and continue with the next
                self.__remove_oldest_short()
                profit =+ (short_price - price) * short_shares
                remaining_to_buy -= short_shares
            else:
                # if oldest short's shares is greater than the required, partially close the short
                new_short_shares = short_shares - remaining_to_buy
                profit += (short_price - price) * remaining_to_buy
                remaining_to_buy = 0
                self.__overwrite_oldest_short((short_price, new_short_shares))
                break

        # add long if couldn't buy all the shares
        if remaining_to_buy > 0:
            self.budget -= price * remaining_to_buy
            self.__add_long((price, remaining_to_buy))
        
        self.budget += profit - discount
        return profit - discount

    def open_short(self, price, num_shares):
        '''Open a new short position or close the oldest long currently active if it meets the required num of shares.
        Returns the profit made by this operation if any when closing the long/s.
        '''
        assert len(self._shorts) * len(self._longs) == 0 # We cannot have both open shorts and longs
        
        remaining_to_sell = num_shares
        discount = 0
        profit = 0

        while remaining_to_sell > 0 and self.__has_longs():
            long = self.__oldest_long()
            long_price, long_shares = long

            if long_shares == remaining_to_sell:
                # if oldest long has the same num of shares that this short needs, close it and compute profit
                self.__remove_oldest_long()
                profit += (price - long_price) * remaining_to_sell
                discount += price * remaining_to_sell * (self._fee/100) # Apply fee
                remaining_to_sell = 0
                break
            elif long_shares < remaining_to_sell:
                # if oldest long's shares is less than the required, close it and continue with the next
                self.__remove_oldest_long()
                profit =+ (price - long_price) * long_shares
                discount += price * long_shares * (self._fee/100) # Apply fee
                remaining_to_sell -= long_shares
            else:
                # if oldest long's shares is greater than the required, partially close the long
                new_long_shares = long_shares - remaining_to_sell
                profit += (price - long_price) * remaining_to_sell
                discount += price * remaining_to_sell * (self._fee/100) # Apply fee
                remaining_to_sell = 0
                self.__overwrite_oldest_long((long_price, new_long_shares))
                break

        # add short if couldn't sell all the shares
        if remaining_to_sell > 0:
            remaining_to_sell = min(remaining_to_sell, self.budget // (price * 0.3))
            discount += price * remaining_to_sell * (self._fee/100)
            self.__add_short((price, remaining_to_sell))
        
        self.budget += profit - discount
        return profit - discount

    def check_stop_loss(self, price):
        if self._stop_loss is None:
            return
        
        for i, values in enumerate(self._longs):
            close_price, shares = values
            if price/close_price < 1 - self._stop_loss:
                self.budget += (price - close_price) * shares * (self._fee/100)
                self._longs.pop(i)

        for i, values in enumerate(self._shorts):
            close_price, shares = values
            if close_price/price < 1 - self._stop_loss:
                self.budget += (close_price - price) * shares * (self._fee/100)
                self._shorts.pop(i)
        

    def end_session(self, price):
        '''End the current trading session at the ending price. Calculate profits based
        on the positions that were open and the ending price to close them.
        '''
        assert len(self._shorts) * len(self._longs) == 0 # We cannot have both open shorts and longs
        
        profit = 0
        discount = self._fee/100 # Apply fee
        
        if self.__has_shorts():
            # Close remaining shorts by doing a long for each
            for short in self._shorts:
                short_price, short_shares = short
                profit += (short_price - price) * short_shares
                profit -= (price * short_shares * discount)
            self._shorts = []

        if self.__has_longs():
            # Close remaining longs by doing a short for each
            for long in self._longs:
                long_price, long_shares = long
                profit += (price - long_price) * long_shares
                profit -= (price * long_shares * discount)
            self._longs = []
        
        self.budget += profit
        return profit

