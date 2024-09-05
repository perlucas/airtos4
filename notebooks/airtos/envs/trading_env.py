from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import matplotlib.pyplot as plt
import numpy as np

from .trading_session import TradingSession

# Possible Actions the agent can choose
ACTION_NOOP = 0
ACTION_BUY = 1
ACTION_SELL = 2

_MIN_ACTION = ACTION_NOOP
_MAX_ACTION = ACTION_SELL

# Representative colors for each action, mapped by index. Used for rendering actions/environment
COLOR_CODES = [
    None,       # noop, uncolored
    '#2D8930',  # buy, strong green
    '#C30000',  # sell, strong red
]


class TradingEnv(py_environment.PyEnvironment):
    """Represents a full stocks trading environment for AI training and testing of models.
    This is based on former stable-baselines3's TradingEnv class.
    Former class has been adapted to provide the required methods for TensorFlow environments
    and to make use of the pandas_ta package stock prices indicators.
    """

    def __init__(self, df, window_size, frame_bound):
        """
        Create a new environment for stocks trading.

        Args:
            df: a pandas dataframe holding the stock prices, indexed by date
            window_size: an integer defining the window size taken as each input pattern set
            frame_bound: a tuple (int, int), defining the frame for which prices are taken for this env. frame_bound[0] should be >= window_size
        """

        # Validate frame bound is a 2D tuple => (start, end)
        assert len(frame_bound) == 2
        self.frame_bound = frame_bound

        # Validate and process prices and signals from source data
        assert df.ndim == 2
        assert window_size >= 1
        self.df = df        
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data() # implemented in subclasses

        # Define observation space
        # For each day, we compute M indicator values. Then the total dimensions will be W * M
        num_dimensions = window_size * self.signal_features.shape[1]
        
        self._observation_spec = array_spec.BoundedArraySpec(
            # Reshape into N-dim array for better support
            shape=(num_dimensions,),
            dtype=np.float32,
            minimum=[-5e0] * num_dimensions, # Indicators are z-score normalized, theoretically, they should range from -3 to 3
            maximum=[5e0] * num_dimensions,
            name='observation')

        # Define action space
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=_MIN_ACTION, maximum=_MAX_ACTION, name='action')

        # Values needed for initializing episodes

        # First W prices are used as inputs for computing TI values as the agent's 1st input set
        # Price W-1 (zero-indexed) will be the act-on price
        self._start_tick = self.window_size - 1
        self._end_tick = len(self.prices) - 1
        
        self._current_tick = None
        self._episode_ended = None
        self._history = None
        self._profit = None
        self._session = TradingSession(fee=2)

    def action_spec(self):
        """Get action space specifications
        """
        return self._action_spec

    def observation_spec(self):
        """Get observation space specifications
        """
        return self._observation_spec

    def _prepare_render(self):
        # Clear plot and render prices
        # plt.figure(figsize=(15, 8)) # This should be done from outside
        plt.cla()
        plt.plot(self.prices)

        # Render each position from history
        _tick = self._start_tick
        for past_action in self._history:
            color = COLOR_CODES[past_action]
            if color is not None:
                plt.scatter(_tick, self.prices[_tick], color=color)
            _tick += 1

        # Add info
        plt.suptitle(
            "Total Profit: %.2f" % self._profit
        )

    def render(self, mode="human"):
        """Render the environment and print its outcome

        Args:
            mode: string defining the rendering mode. Currently, only 'human' is supported
        """

        if mode != "human":
            raise NotImplementedError(
                'not human mode has not been implemented')

        self._prepare_render()
        plt.show()

    def save_render(self, filename="trading_env"):
        """Render the environment and save it into a file

        Args:
            filename: string, target filename. Defaults to 'trading_env'
        """
        self._prepare_render()
        plt.savefig(filename)

    def _reset(self):
        self._history = []
        self._current_tick = self._start_tick
        self._episode_ended = False
        self._profit = 0
        self._session.reset()

        observation = self._get_observation()
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        # Current stock price
        current_price = self.prices[ self._current_tick ]

        # Compute step reward and add it to profit
        step_reward = 0
        if action == ACTION_BUY:
            step_reward = self._session.open_long(current_price)
        elif action == ACTION_SELL:
            step_reward = self._session.open_short(current_price)

        # Add daily punishment
        step_reward -= 10
        
        self._profit += step_reward

        # Store history for rendering purposes
        self._history.append(action)

        self._current_tick += 1
        
        if self._current_tick == self._end_tick:
            # Finish episode if reached last tick
            self._episode_ended = True
            self._profit += self._session.end_session(current_price)

        observation = self._get_observation()

        # Finish episode
        if self._episode_ended:
            return ts.termination(np.array(observation, dtype=np.float32), reward=step_reward)

        # Move on to next step
        return ts.transition(np.array(observation, dtype=np.float32), reward=step_reward, discount=1.0)

    def _get_observation(self):
        num_dimensions = self.window_size * self.signal_features.shape[1]
        # Get an array of window_size features
        features = self.signal_features[(
            self._current_tick-self.window_size+1):self._current_tick+1]
        # Reshape features to match N-dim array (observation space)
        # print(features)
        return np.reshape(features, (num_dimensions, ))

    def _process_data(self):
        raise NotImplementedError(
            'has not been implemented, try using a subclass')
