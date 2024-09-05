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

    def open_long(self, price):
        '''Open a new long position or close the oldest short currently active
        Returns the profit made by this operation if any
        '''
        assert len(self._shorts) * len(self._longs) == 0
        discount = price * (self._fee/100) # Apply fee
        
        if len(self._shorts) > 0:
            oldest_short_price = self._shorts.pop(0) # Remove and close short if any
            return oldest_short_price - price - discount
        
        # Open long otherwise
        self._longs.append(price)
        return -discount

    def open_short(self, price):
        '''Open a new short position or close the oldest long currently active
        Returns the profit made by this operation if any
        '''
        assert len(self._shorts) * len(self._longs) == 0
        discount = price * (self._fee/100) # Apply fee
        
        if len(self._longs) > 0:
            oldest_long_price = self._longs.pop(0) # Remove and close long if any
            return price - oldest_long_price - discount
            
        # Open short otherwise
        self._shorts.append(price)
        return -discount

    def end_session(self, price):
        '''End the current trading session at the ending price. Calculate profits based
        on the positions that were open and the ending price
        '''
        assert len(self._shorts) * len(self._longs) == 0
        profit = 0
        discount = price * (self._fee/100) # Apply fee
        
        if len(self._shorts) > 0:
            # Close remaining shorts by doing a long for each
            for short in self._shorts:
                profit += (short - price - discount)
            self._shorts = []

        if len(self._longs) > 0:
            # Close remaining longs by doing a short for each
            for long in self._longs:
                profit += (price - long - discount)
            self._longs = []
        
        return profit



# Unit tests

def __run_simulation_tests():
    '''Run the TradingSession class unit tests'''

    tests = [
        {
            # more longs than shorts
            'fee': 0,
            'positions': (('l', 5, 0), ('l', 4, 0), ('s', 10, 5)),
            'end_session_price': 5,
            'end_session_profit': 1
        },
        {
            # more shorts than longs
            'fee': 0,
            'positions': (('s', 15, 0), ('l', 5, 10), ('s', 10, 0)),
            'end_session_price': 12,
            'end_session_profit': -2
        },
        {
            # same shorts and longs
            'fee': 0,
            'positions': (('l', 5, 0), ('l', 4, 0), ('s', 10, 5), ('s', 12, 8)),
            'end_session_price': 5,
            'end_session_profit': 0
        },
        {
            # only longs
            'fee': 0,
            'positions': (('l', 5, 0), ('l', 6, 0), ('l', 7, 0)),
            'end_session_price': 10,
            'end_session_profit': 12
        },
        {
            # only shorts
            'fee': 0,
            'positions': (('s', 10, 0), ('s', 20, 0), ('s', 13, 0)),
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
            'positions': (('l', 5, -5*0.02), ('l', 4, -4*0.02), ('s', 10, 5 - 10*0.02)),
            'end_session_price': 5,
            'end_session_profit': 1 - 5*0.02
        },
        {
            # (fee) more shorts than longs
            'fee': 2,
            'positions': (('s', 15, -15*0.02), ('l', 5, 10 - 5*0.02), ('s', 10, -10*0.02)),
            'end_session_price': 12,
            'end_session_profit': -2 - 12*0.02
        },
        {
            # (fee) same shorts and longs
            'fee': 2,
            'positions': (('l', 5, -5*0.02), ('l', 4, -4*0.02), ('s', 10, 5-10*0.02), ('s', 12, 8-12*0.02)),
            'end_session_price': 5,
            'end_session_profit': 0
        },
        {
            # (fee) only longs
            'fee': 2,
            'positions': (('l', 5, -5*0.02), ('l', 6, -6*0.02), ('l', 7, -7*0.02)),
            'end_session_price': 10,
            'end_session_profit': 12 - 10*0.02*3
        },
        {
            # (fee) only shorts
            'fee': 2,
            'positions': (('s', 10, -10*0.02), ('s', 20, -20*0.02), ('s', 13, -13*0.02)),
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
    ]

    def compare_floats(x1, x2, err, msg):
        diff = abs(x1-x2)
        assert diff <= err, msg

    for idx, test in enumerate(tests):
        # Start new trading session and simulate running through these operations
        session = TradingSession(fee = test['fee'])
        for i, p in enumerate(test['positions']):
            operation_type, price, expected_profit = p
            actual_profit = session.open_long(price) if operation_type == 'l' else session.open_short(price)
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