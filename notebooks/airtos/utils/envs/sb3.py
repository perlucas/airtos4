from os import path

import numpy as np

from sb3.combined_env import CombinedEnv
from . import load_dataset

# ====================================== Utils functions ======================================

def create_env(env_type: str, df, window_size, frame_bound):
    '''Create an environment based on the type of environment (only 'com' is supported)
    :param env_type: str: The type of environment to create
    :param df: pd.DataFrame: The dataframe to use for the environment
    :param window_size: int: The window size to use for the environment
    :param frame_bound: tuple: The frame bound to use for the environment
    :return: CombinedEnv: The environment created
    '''
    if env_type == 'com':
        return CombinedEnv(df=df, window_size=window_size, frame_bound=frame_bound)
    raise NotImplementedError('unknown type')


def create_training_envs(env_type: str):
    '''Create a list of training environments based on the type of environment
    :param env_type: str: The type of environment to create
    :return: list: The list of training environments created
    '''

    def file_to_path(file):
        return path.join(path.dirname(__file__), f'stocks_data/{file}')

    # use file from this file's directory
    ko_df = load_dataset(file_to_path('KO.csv'))
    amzn_df = load_dataset(file_to_path('AMZN.csv'))
    amd_df = load_dataset(file_to_path('AMD.csv'))
    pypl_df = load_dataset(file_to_path('PYPL.csv'))
    nflx_df = load_dataset(file_to_path('NFLX.csv'))
    window_size = 10

    return [
        # KO training envs
        create_env(env_type, ko_df, window_size, (10, 120)),
        create_env(env_type, ko_df, window_size, (120, 230)),
        create_env(env_type, ko_df, window_size, (350, 470)),
        create_env(env_type, ko_df, window_size, (1000, 1120)),
        create_env(env_type, ko_df, window_size, (1700, 1820)),

        # AMZN training envs
        create_env(env_type, amzn_df, window_size, (10, 120)),
        create_env(env_type, amzn_df, window_size, (120, 230)),
        create_env(env_type, amzn_df, window_size, (350, 470)),
        create_env(env_type, amzn_df, window_size, (1000, 1120)),
        create_env(env_type, amzn_df, window_size, (1700, 1820)),

        # AMD training envs
        create_env(env_type, amd_df, window_size, (10, 120)),
        create_env(env_type, amd_df, window_size, (120, 230)),
        create_env(env_type, amd_df, window_size, (350, 470)),
        create_env(env_type, amd_df, window_size, (1000, 1120)),
        create_env(env_type, amd_df, window_size, (1700, 1820)),

        # PYPL training envs
        create_env(env_type, pypl_df, window_size, (10, 120)),
        create_env(env_type, pypl_df, window_size, (120, 230)),
        create_env(env_type, pypl_df, window_size, (350, 470)),
        create_env(env_type, pypl_df, window_size, (1000, 1120)),
        create_env(env_type, pypl_df, window_size, (1700, 1820)),

        # NFLX training envs
        create_env(env_type, nflx_df, window_size, (10, 120)),
        create_env(env_type, nflx_df, window_size, (120, 230)),
        create_env(env_type, nflx_df, window_size, (350, 470)),
        create_env(env_type, nflx_df, window_size, (1000, 1120)),
        create_env(env_type, nflx_df, window_size, (1700, 1820)),
    ]


def create_testing_env(env_type: str):
    '''Create a testing environment based on the type of environment
    :param env_type: str: The type of environment to create
    :return: CombinedEnv: The testing environment created
    '''

    def file_to_path(file):
        return path.join(path.dirname(__file__), f'stocks_data/{file}')
    
    ko_df = load_dataset(file_to_path('KO.csv'))
    window_size = 10
    return create_env(env_type, ko_df, window_size, (2000, 2300))


# ====================================== Random Picker ======================================

class BufferedRandomPicker:
  def __init__(self, values, buff_len=5):
    '''Create a new BufferedRandomPicker instance
    :param values: list: The list of values to pick from
    :param buff_len: int: The length of the buffer
    '''
    self.__buffer = []
    self.__values = values
    for i in range(buff_len):
      self.__buffer.append(values[i])

  def pick_random(self):
    '''Pick a random value from the list of values
    :return: object: The random value picked
    '''
    while True:
      idx = self.__rand_idx()
      value = self.__values[idx]
      if value not in self.__buffer:
        self.__buffer.pop(0)
        self.__buffer.append(value)
        return value

  def __rand_idx(self):
    idx = np.random.randint(len(self.__values))
    return idx


# ====================================== Exported Variables ======================================

ENV_TYPE = 'com'

# Create training and testing environments
train_envs = create_training_envs(ENV_TYPE)
eval_env = create_testing_env(ENV_TYPE)

# Create a random picker for training environments
picker = BufferedRandomPicker(train_envs)
def get_random_train_env():
    '''Get a random training environment
    :return: TFPyEnvironment: The random training environment
    '''
    env = picker.pick_random()
    return env
