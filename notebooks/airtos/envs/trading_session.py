class TradingSession:

    def __init__(self, fee = 0):
        self._shorts = []
        self._longs = []

        assert fee >= 0
        self._fee = fee

    def reset(self):
        '''Reset the porftolio removing all longs and shorts
        '''
        self._shorts = []
        self._longs = []

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
        
        discount = price * (self._fee/100) # Apply fee

        remaining_to_buy = num_shares
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
            self.__add_long((price, remaining_to_buy))
        
        return profit - discount

    def open_short(self, price, num_shares):
        '''Open a new short position or close the oldest long currently active if it meets the required num of shares.
        Returns the profit made by this operation if any when closing the long/s.
        '''
        assert len(self._shorts) * len(self._longs) == 0 # We cannot have both open shorts and longs
        
        discount = price * (self._fee/100) # Apply fee

        remaining_to_sell = num_shares
        profit = 0

        while remaining_to_sell > 0 and self.__has_longs():
            long = self.__oldest_long()
            long_price, long_shares = long

            if long_shares == remaining_to_sell:
                # if oldest long has the same num of shares that this short needs, close it and compute profit
                self.__remove_oldest_long()
                profit += (price - long_price) * remaining_to_sell
                remaining_to_sell = 0
                break
            elif long_shares < remaining_to_sell:
                # if oldest long's shares is less than the required, close it and continue with the next
                self.__remove_oldest_long()
                profit =+ (price - long_price) * long_shares
                remaining_to_sell -= long_shares
            else:
                # if oldest long's shares is greater than the required, partially close the long
                new_long_shares = long_shares - remaining_to_sell
                profit += (price - long_price) * remaining_to_sell
                remaining_to_sell = 0
                self.__overwrite_oldest_long((long_price, new_long_shares))
                break

        # add short if couldn't sell all the shares
        if remaining_to_sell > 0:
            self.__add_short((price, remaining_to_sell))
        
        return profit - discount

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
        
        return profit



# Unit tests

def __run_simulation_tests():
    '''Run the TradingSession class unit tests'''

    tests = [
        {
            # more longs than shorts
            'fee': 0,
            'positions': (('l', 5, 1, 0), ('l', 4, 1, 0), ('s', 10, 1, 5)),
            'end_session_price': 5,
            'end_session_profit': 1
        },
        {
            # more shorts than longs
            'fee': 0,
            'positions': (('s', 15, 1, 0), ('l', 5, 1, 10), ('s', 10, 1, 0)),
            'end_session_price': 12,
            'end_session_profit': -2
        },
        {
            # same shorts and longs
            'fee': 0,
            'positions': (('l', 5, 1, 0), ('l', 4, 1, 0), ('s', 10, 1, 5), ('s', 12, 1, 8)),
            'end_session_price': 5,
            'end_session_profit': 0
        },
        {
            # only longs
            'fee': 0,
            'positions': (('l', 5, 1, 0), ('l', 6, 1, 0), ('l', 7, 1, 0)),
            'end_session_price': 10,
            'end_session_profit': 12
        },
        {
            # only shorts
            'fee': 0,
            'positions': (('s', 10, 1, 0), ('s', 20, 1, 0), ('s', 13, 1, 0)),
            'end_session_price': 5,
            'end_session_profit': 28
        },
        {
            # no operations
            'fee': 0,
            'positions': (),
            'end_session_price': 5,
            'end_session_profit': 0
        },
        {
            # (fee) more longs than shorts
            'fee': 2,
            'positions': (('l', 5, 1, -5*0.02), ('l', 4, 1, -4*0.02), ('s', 10, 1, 5 - 10*0.02)),
            'end_session_price': 5,
            'end_session_profit': 1 - 5*0.02
        },
        {
            # (fee) more shorts than longs
            'fee': 2,
            'positions': (('s', 15, 1, -15*0.02), ('l', 5, 1, 10 - 5*0.02), ('s', 10, 1, -10*0.02)),
            'end_session_price': 12,
            'end_session_profit': -2 - 12*0.02
        },
        {
            # (fee) same shorts and longs
            'fee': 2,
            'positions': (('l', 5, 1, -5*0.02), ('l', 4, 1, -4*0.02), ('s', 10, 1, 5-10*0.02), ('s', 12, 1, 8-12*0.02)),
            'end_session_price': 5,
            'end_session_profit': 0
        },
        {
            # (fee) only longs
            'fee': 2,
            'positions': (('l', 5, 1, -5*0.02), ('l', 6, 1, -6*0.02), ('l', 7, 1, -7*0.02)),
            'end_session_price': 10,
            'end_session_profit': 12 - 10*0.02*3
        },
        {
            # (fee) only shorts
            'fee': 2,
            'positions': (('s', 10, 1, -10*0.02), ('s', 20, 1, -20*0.02), ('s', 13, 1, -13*0.02)),
            'end_session_price': 5,
            'end_session_profit': 28 - 5*0.02*3
        },
        {
            # (fee) no operations
            'fee': 2,
            'positions': (),
            'end_session_price': 5,
            'end_session_profit': 0
        },
        {
            # (non-fee) more longs than shorts, same num_shares (a-1)
            'fee': 0,
            'positions': (
                ('l', 5, 2, 0),
                ('l', 4, 2, 0),
                ('s', 10, 2, 10),
            ),
            'end_session_price': 5,
            'end_session_profit': 2
        },
        {
            # (non-fee) more longs than shorts, different num_shares, same N. operations (a-1-1)
            'fee': 0,
            'positions': (
                ('l', 5, 4, 0),
                ('s', 10, 2, 10),
                ('l', 3, 3, 0),
                ('s', 12, 3, 23),
            ),
            'end_session_price': 5,
            'end_session_profit': 4
        },
        {
            # (non-fee) more longs than shorts, different num_shares, different N. operations (a-1-2)
            'fee': 0,
            'positions': (
                ('l', 2, 3, 0),
                ('s', 5, 1, 3),
                ('l', 3, 2, 0),
            ),
            'end_session_price': 10,
            'end_session_profit': 30
        },
        {
            # (non-fee) more longs than shorts, different num_shares, different N. operations (a-1-3)
            'fee': 0,
            'positions': (
                ('l', 3, 10, 0),
                ('s', 8, 2, 10),
                ('s', 9, 2, 12),
            ),
            'end_session_price': 7,
            'end_session_profit': 24
        },
        {
            # (non-fee) more shorts than longs, same num_shares, different N. operations (b-1)
            'fee': 0,
            'positions': (
                ('l', 2, 2, 0),
                ('s', 3, 2, 2),
                ('s', 3.5, 2, 0),
            ),
            'end_session_price': 7,
            'end_session_profit': -7
        },
        {
            # (non-fee) more shorts than longs, different num_shares, same N. operations (b-1-1)
            'fee': 0,
            'positions': (
                ('l', 3, 4, 0),
                ('s', 5, 6, 8),
                ('l', 2, 2, 6),
                ('s', 7, 10, 0),
            ),
            'end_session_price': 5,
            'end_session_profit': 20
        },
        {
            # (non-fee) more shorts than longs, different num_shares, different N. operations (b-1-2)
            'fee': 0,
            'positions': (
                ('s', 12, 5, 0),
                ('l', 6, 2, 12),
                ('s', 8, 3, 0),
            ),
            'end_session_price': 2,
            'end_session_profit': 48
        },
        {
            # (non-fee) more shorts than longs, different num_shares, different N. operations (b-1-3)
            'fee': 0,
            'positions': (
                ('l', 4, 10, 0),
                ('s', 15, 25, 110),
                ('l', 5, 5, 50),
            ),
            'end_session_price': 10,
            'end_session_profit': 50
        },
        {
            # (non-fee) same shorts and longs, same num_shares, same N. operations (c-1)
            'fee': 0,
            'positions': (
                ('s', 16, 3, 0),
                ('l', 5, 3, 33),
                ('s', 10, 3, 0),
                ('l', 4, 3, 18),
            ),
            'end_session_price': 3,
            'end_session_profit': 0
        },
        {
            # (non-fee) only longs, different num_shares (d-1)
            'fee': 0,
            'positions': (
                ('l', 5, 10, 0),
                ('l', 6, 10, 0),
            ),
            'end_session_price': 3,
            'end_session_profit': -50
        },
        {
            # (non-fee) only longs, different num_shares (d-2)
            'fee': 0,
            'positions': (
                ('l', 5, 8, 0),
                ('l', 4, 3, 0),
            ),
            'end_session_price': 10,
            'end_session_profit': 58
        },
        {
            # (non-fee) only shorts, same num_shares (e-1)
            'fee': 0,
            'positions': (
                ('s', 10, 20, 0),
                ('s', 11, 20, 0),
            ),
            'end_session_price': 15,
            'end_session_profit': -180
        },
        {
            # (non-fee) only shorts, different num_shares (e-2)
            'fee': 0,
            'positions': (
                ('s', 13, 5, 0),
                ('s', 15, 8, 0),
            ),
            'end_session_price': 10,
            'end_session_profit': 55
        },
    ]

    def compare_floats(x1, x2, err, msg):
        diff = abs(x1-x2)
        assert diff <= err, msg

    for idx, test in enumerate(tests):
        # Start new trading session and simulate running through these operations
        session = TradingSession(fee = test['fee'])
        for i, p in enumerate(test['positions']):
            operation_type, price, num_shares, expected_profit = p
            actual_profit = session.open_long(price, num_shares) if operation_type == 'l' else session.open_short(price, num_shares)
            compare_floats(
                actual_profit,
                expected_profit,
                1e-10,
                f'Failed test {idx} for position {i}, expected {expected_profit}, got {actual_profit}')

        end_profit = session.end_session(test['end_session_price'])
        expected_end_profit = test['end_session_profit']
        compare_floats(
            end_profit,
            expected_end_profit,
            1e-10,
            f'Failed end_session for test {idx}, expected {expected_end_profit}, got {end_profit}')
    
    print('All good, nice job!')