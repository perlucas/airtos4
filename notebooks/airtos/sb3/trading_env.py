import numpy as np
import gymnasium as gym

from gymnasium import spaces

from envs.trading_session import TradingSession


# Possible Actions the agent can choose
ACTION_NOOP = 0
ACTION_BUY_LOW = 1
ACTION_BUY_MEDIUM = 2
ACTION_BUY_HIGH = 3
ACTION_SELL_LOW = 4
ACTION_SELL_MEDIUM = 5
ACTION_SELL_HIGH = 6

_MIN_ACTION = ACTION_NOOP
_MAX_ACTION = ACTION_SELL_HIGH

# Posible number of shares the agent can trade: 5 shares, 10 shares or 20 shares
def extract_action_and_num_shares(code):
    """Extract the action and number of shares from the encoded action"""
    code_action_map = [
        ('noop', 0),  # No op
        ('buy', 5),   # buy low
        ('buy', 10),  # buy medium
        ('buy', 20),  # buy high
        ('sell', 5),  # sell low
        ('sell', 10), # sell medium
        ('sell', 20), # sell high
    ]
    return code_action_map[code]


class TradingEnv(gym.Env):
    """Trading Environment implementing the OpenAI Gym interface"""
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, df, window_size, frame_bound, render_mode=None, no_action_punishment=0):
        """
        Create a new environment for stocks trading.

        Args:
            df: a pandas dataframe holding the stock prices, indexed by date
            window_size: an integer defining the window size taken as each input pattern set
            frame_bound: a tuple (int, int), defining the frame for which prices are taken for this env. frame_bound[0] should be >= window_size
        """
        # Currently we don't support rendering via Gym interface
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Validate frame bound is a 2D tuple => (start, end)
        assert len(frame_bound) == 2
        self.frame_bound = frame_bound

        # Validate and process prices and signals from source data
        assert df.ndim == 2
        assert window_size >= 1
        self.df = df
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data() # implemented in subclasses?

        # Define observation space
        # For each day, we compute M indicator values. Then the total dimensions will be W * M
        num_dimensions = window_size * self.signal_features.shape[1]

        self.observation_space = spaces.Box(
            # Indicators are z-score normalized, theoretically, they should range from -3 to 3
            low=-3e0,
            high=3e0,
            shape=(num_dimensions,), # Reshape into N-dim array for better support
            dtype=np.float32,
        )

        # Define action space
        self.action_space = spaces.Discrete(_MAX_ACTION - _MIN_ACTION + 1)

        # Values needed for initializing episodes

        # First W prices are used as inputs for computing TI values as the agent's 1st input set
        # Price W-1 (zero-indexed) will be the act-on price
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 1
        
        self._current_tick = None
        self._history = None
        self._profit = None
        self._punishment_on_no_action = no_action_punishment
        self._cumulated_punish_counter = 0
        self._session = TradingSession(fee = 2)


    def _get_observation(self):
        """Get the current observation from the environment"""
        num_dimensions = self.window_size * self.signal_features.shape[1]
        # Get an array of window_size features
        features = self.signal_features[(
            self._current_tick-self.window_size+1):self._current_tick+1]
        # Reshape features to match N-dim array (observation space)
        return np.reshape(features, (num_dimensions, ))

    def _get_info(self):
        """Get the current info from the environment"""
        return {
            "profit": self._profit,
            "progress": self._current_tick / self._end_tick,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state"""
        super().reset(seed=seed)

        self._history = []
        self._current_tick = self._start_tick
        self._profit = 0
        self._cumulated_punish_counter = 0
        self._session.reset()

        observation = self._get_observation()
        info = self._get_info()
        return observation, info


    def step(self, action_code):
        """Take a step in the environment"""
        # Current stock price
        current_price = self.prices[ self._current_tick ]

        # Compute step reward and add it to profit
        step_reward = 0
        action, num_shares = extract_action_and_num_shares(action_code)

        # Increase punishment on consecutive no actions
        self._cumulated_punish_counter += 1
        
        if action == 'buy':
            step_reward = self._session.open_long(current_price, num_shares)
            self._cumulated_punish_counter = 0
        elif action == 'sell':
            step_reward = self._session.open_short(current_price, num_shares)
            self._cumulated_punish_counter = 0

        # Add punishment for no action
        punishment = self._cumulated_punish_counter * self._punishment_on_no_action
        step_reward -= punishment
        
        self._profit += step_reward

        # Store history for rendering
        self._history.append(action_code)

        self._current_tick += 1
        
        episode_ended = False
        if self._current_tick == self._end_tick:
            # Finish episode if reached last tick
            episode_ended = True
            self._profit += self._session.end_session(current_price)

        observation = self._get_observation()

        # Move on to next step or finish episode
        return observation, step_reward, episode_ended, False, self._get_info()


    def render(self):
        """Render the environment"""
        raise NotImplementedError

    def close(self):
        """Close the environment"""
        pass


    # TODO: Implement the rest of the methods
    def _process_data(self):
        raise NotImplementedError
