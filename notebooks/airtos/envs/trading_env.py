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

# Possible Actions the agent can choose
ACTION_NOOP = 0
ACTION_WEAK_BUY = 1
ACTION_REGULAR_BUY = 2
ACTION_STRONG_BUY = 3
ACTION_WEAK_SELL = 4
ACTION_REGULAR_SELL = 5
ACTION_STRONG_SELL = 6
_MIN_ACTION = ACTION_NOOP
_MAX_ACTION = ACTION_STRONG_SELL

# Percentages of either budget or shares affected by a given action, ex.: WEAK_BUY means to invest 5% of current budget,
# REGULAR_SELL means to sell 50% of current shares
ACTION_PERCENTAGES = [
    0,      # noop
    0.05,   # weak buy (budget)
    0.1,    # regular buy (budget)
    0.3,    # strong buy (budget)
    0.3,    # weak sell (shares)
    0.5,    # regular sell (shares)
    1,      # strong sell (shares)
]

# Representative colors for each action, mapped by index. Used for rendering actions/environment
COLOR_CODES = [
    None,       # noop, uncolored
    '#6AF26F',  # weak buy, light green
    '#59C05C',  # regular buy, green
    '#2D8930',  # strong buy, strong green
    '#F97373',  # weak sell, light red
    '#F53D3D',  # regular sell, red
    '#C30000',  # strong sell, strong red
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

        # Reverse dataframe to sort prices by date => comes ordered from load_dataset function
        self.df = df
        
        self.window_size = window_size
        self.prices, self.signal_features = self._process_data()

        # Define observation and action space
        num_dimensions = window_size * self.signal_features.shape[1]
        self._observation_spec = array_spec.BoundedArraySpec(
            # Reshape into N-dim array for better support
            shape=(num_dimensions,),
            dtype=np.float32,
            minimum=[-5e0] * num_dimensions,
            maximum=[5e0] * num_dimensions,
            name='observation')

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=_MIN_ACTION, maximum=_MAX_ACTION, name='action')

        # Values needed for initializing episodes
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        # Initial budget is enough money to buy 100 shares
        self._initial_funds = 100 * self.prices[self._start_tick]
        self._final_funds = None
        self._current_tick = None
        self._shares = None
        self._budget = None
        self._episode_ended = None
        self._history = None

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
            "Initial Funds: %.2f" % self._initial_funds + ' ~ ' +
            "Final Funds: %.2f" % self._final_funds
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
        self._shares = 0
        self._budget = self._initial_funds
        self._episode_ended = False

        observation = self._get_observation()
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        self._current_tick += 1

        step_reward = self._update_and_get_reward(action)

        if self._current_tick == self._end_tick or not self._can_still_operate():
            # Finish episode if reached last possible tick or can't operate
            self._episode_ended = True

        observation = self._get_observation()

        if self._episode_ended:
            self._final_funds = self._budget + \
                self._shares * \
                self.prices[self._current_tick]  # This is needed for rendering final budget
            return ts.termination(np.array(observation, dtype=np.float32), reward=step_reward)

        return ts.transition(np.array(observation, dtype=np.float32), reward=step_reward, discount=1.0)

    def _get_observation(self):
        num_dimensions = self.window_size * self.signal_features.shape[1]
        # Get an array of window_size features
        features = self.signal_features[(
            self._current_tick-self.window_size+1):self._current_tick+1]
        # Reshape features to match N-dim array (observation space)
        return np.reshape(features, (num_dimensions, ))

    def _process_data(self):
        raise NotImplementedError(
            'has not been implemented, try using a subclass')

    def _update_and_get_reward(self, action):

        # Compute funds before action was taken
        prev_price = self.prices[self._current_tick - 1]
        prev_funds = self._budget + self._shares * prev_price

        # Apply action to update state: budget and shares
        new_shares, operation_shares = self._compute_new_shares(
            action, price=prev_price)
        new_budget = self._compute_new_budget(
            action, operation_shares=operation_shares, price=prev_price)

        # Save checkpoint in history, save it as NOOP if no real operation was performed
        self._history.append(action if operation_shares > 0 else ACTION_NOOP)

        assert new_budget >= 0, f"New budget is negative: {new_budget}"
        assert new_shares >= 0, f"New shares are negative: {new_shares}"

        self._shares = new_shares
        self._budget = new_budget
        cur_funds = self._budget + self._shares * \
            self.prices[self._current_tick]

        return (cur_funds - prev_funds)

    def _compute_new_shares(self, action, price):
        if action == ACTION_NOOP:
            return (self._shares, 0)

        is_sell = action in [ACTION_WEAK_SELL,
                             ACTION_REGULAR_SELL, ACTION_STRONG_SELL]
        # Cannot sell shares if has not bought any before
        if self._shares == 0 and is_sell:
            return (0, 0)

        if is_sell:
            # Compute operation shares from current shares
            operation_shares = np.floor(
                self._shares * ACTION_PERCENTAGES[action])
            # Cannot sell more than its current holdings
            operation_shares = np.min([operation_shares, self._shares])
            return (self._shares - operation_shares, operation_shares)

        # It's buy
        budget = np.max([0, self._budget])  # In case it's negative
        operation_shares = np.floor(
            (budget * ACTION_PERCENTAGES[action]) / price)  # Number of shares to buy, percentage of budget
        return (self._shares + operation_shares, operation_shares)

    def _compute_new_budget(self, action, operation_shares, price):
        if action == ACTION_NOOP:
            return self._budget

        is_sell = action in [ACTION_WEAK_SELL,
                             ACTION_REGULAR_SELL, ACTION_STRONG_SELL]
        # Increase budget if selling shares, decrease budget if buying
        if is_sell:
            return self._budget + price * operation_shares
        return self._budget - price * operation_shares

    def _can_still_operate(self):
        if self._shares > 0:
            return True

        budget = np.max([0, self._budget])  # In case it's negative
        min_price = np.min(self.prices[self._current_tick:])
        min_shares_to_buy = np.floor(
            budget * ACTION_PERCENTAGES[ACTION_WEAK_BUY] / min_price)  # can it buy at least 1 share?
        return min_shares_to_buy > 0
