from os import path

import numpy as np

from sb3.combined_env import CombinedEnv
from . import load_dataset

# ====================================== Utils functions ======================================

def create_env(env_type: str, df, window_size, frame_bound, no_action_punishment=0):
    '''Create an environment based on the type of environment (only 'com' is supported)
    :param env_type: str: The type of environment to create
    :param df: pd.DataFrame: The dataframe to use for the environment
    :param window_size: int: The window size to use for the environment
    :param frame_bound: tuple: The frame bound to use for the environment
    :return: CombinedEnv: The environment created
    '''
    if env_type == 'com':
        return CombinedEnv(df=df, window_size=window_size, frame_bound=frame_bound, no_action_punishment=no_action_punishment)
    raise NotImplementedError('unknown type')


def create_training_envs(env_type: str, no_action_punishment=0):
    '''Create a list of training environments based on the type of environment
    :param env_type: str: The type of environment to create
    :return: list: The list of training environments created
    '''

    def file_to_path(file):
        return path.join(path.dirname(__file__), f'stocks_data/{file}')

    # use file from this file's directory
    files = [
       'CRM.csv',
       'AMZN.csv',
       'AMD.csv',
       'PYPL.csv',
       'NFLX.csv',
       'NVDA.csv',
       'ORCL.csv',
       'BABA.csv',
       'CSCO.csv',
       'INTC.csv',
       'QCOM.csv',
       'UBER.csv',
       ]
    crm_df, amzn_df, amd_df, pypl_df, nflx_df, nvda_df, orcl_df, baba_df, csco_df, intc_df, qcom_df, uber_df = [load_dataset(file_to_path(file)) for file in files]

    window_size = 10

    # Environment Length: 2 months (~45 trading days)
    # Number of environments: 60
    return [
        # CRM training envs
        create_env(env_type, crm_df, window_size, (200, 245), no_action_punishment=no_action_punishment),
        create_env(env_type, crm_df, window_size, (300, 345), no_action_punishment=no_action_punishment),
        create_env(env_type, crm_df, window_size, (500, 545), no_action_punishment=no_action_punishment),
        create_env(env_type, crm_df, window_size, (545, 590), no_action_punishment=no_action_punishment),
        create_env(env_type, crm_df, window_size, (600, 645), no_action_punishment=no_action_punishment),

        # AMZN training envs
        create_env(env_type, amzn_df, window_size, (1000, 1045), no_action_punishment=no_action_punishment),
        create_env(env_type, amzn_df, window_size, (1200, 1245), no_action_punishment=no_action_punishment),
        create_env(env_type, amzn_df, window_size, (1300, 1345), no_action_punishment=no_action_punishment),
        create_env(env_type, amzn_df, window_size, (1400, 1445), no_action_punishment=no_action_punishment),
        create_env(env_type, amzn_df, window_size, (1500, 1545), no_action_punishment=no_action_punishment),

        # AMD training envs
        create_env(env_type, amd_df, window_size, (400, 445), no_action_punishment=no_action_punishment),
        create_env(env_type, amd_df, window_size, (500, 545), no_action_punishment=no_action_punishment),
        create_env(env_type, amd_df, window_size, (600, 645), no_action_punishment=no_action_punishment),
        create_env(env_type, amd_df, window_size, (700, 745), no_action_punishment=no_action_punishment),
        create_env(env_type, amd_df, window_size, (800, 845), no_action_punishment=no_action_punishment),

        # PYPL training envs
        create_env(env_type, pypl_df, window_size, (400, 445), no_action_punishment=no_action_punishment),
        create_env(env_type, pypl_df, window_size, (500, 545), no_action_punishment=no_action_punishment),
        create_env(env_type, pypl_df, window_size, (600, 645), no_action_punishment=no_action_punishment),
        create_env(env_type, pypl_df, window_size, (700, 745), no_action_punishment=no_action_punishment),
        create_env(env_type, pypl_df, window_size, (1000, 1100), no_action_punishment=no_action_punishment),

        # NFLX training envs
        create_env(env_type, nflx_df, window_size, (800, 845), no_action_punishment=no_action_punishment),
        create_env(env_type, nflx_df, window_size, (900, 945), no_action_punishment=no_action_punishment),
        create_env(env_type, nflx_df, window_size, (1000, 1045), no_action_punishment=no_action_punishment),
        create_env(env_type, nflx_df, window_size, (1100, 1145), no_action_punishment=no_action_punishment),
        create_env(env_type, nflx_df, window_size, (1200, 1245), no_action_punishment=no_action_punishment),

        # NVDA training envs
        create_env(env_type, nvda_df, window_size, (800, 845), no_action_punishment=no_action_punishment),
        create_env(env_type, nvda_df, window_size, (900, 945), no_action_punishment=no_action_punishment),
        create_env(env_type, nvda_df, window_size, (1000, 1045), no_action_punishment=no_action_punishment),
        create_env(env_type, nvda_df, window_size, (1100, 1145), no_action_punishment=no_action_punishment),
        create_env(env_type, nvda_df, window_size, (1200, 1245), no_action_punishment=no_action_punishment),

        # ORCL training envs
        create_env(env_type, orcl_df, window_size, (400, 445), no_action_punishment=no_action_punishment),
        create_env(env_type, orcl_df, window_size, (500, 545), no_action_punishment=no_action_punishment),
        create_env(env_type, orcl_df, window_size, (600, 645), no_action_punishment=no_action_punishment),
        create_env(env_type, orcl_df, window_size, (700, 745), no_action_punishment=no_action_punishment),
        create_env(env_type, orcl_df, window_size, (800, 845), no_action_punishment=no_action_punishment),

        # BABA training envs
        create_env(env_type, baba_df, window_size, (800, 845), no_action_punishment=no_action_punishment),
        create_env(env_type, baba_df, window_size, (900, 945), no_action_punishment=no_action_punishment),
        create_env(env_type, baba_df, window_size, (1000, 1045), no_action_punishment=no_action_punishment),
        create_env(env_type, baba_df, window_size, (1100, 1145), no_action_punishment=no_action_punishment),
        create_env(env_type, baba_df, window_size, (1200, 1245), no_action_punishment=no_action_punishment),

        # CSCO training envs
        create_env(env_type, csco_df, window_size, (400, 445), no_action_punishment=no_action_punishment),
        create_env(env_type, csco_df, window_size, (500, 545), no_action_punishment=no_action_punishment),
        create_env(env_type, csco_df, window_size, (600, 645), no_action_punishment=no_action_punishment),
        create_env(env_type, csco_df, window_size, (700, 745), no_action_punishment=no_action_punishment),
        create_env(env_type, csco_df, window_size, (800, 845), no_action_punishment=no_action_punishment),

        # INTC training envs
        create_env(env_type, intc_df, window_size, (800, 845), no_action_punishment=no_action_punishment),
        create_env(env_type, intc_df, window_size, (900, 945), no_action_punishment=no_action_punishment),
        create_env(env_type, intc_df, window_size, (1000, 1045), no_action_punishment=no_action_punishment),
        create_env(env_type, intc_df, window_size, (1100, 1145), no_action_punishment=no_action_punishment),
        create_env(env_type, intc_df, window_size, (1200, 1245), no_action_punishment=no_action_punishment),

        # QCOM training envs
        create_env(env_type, qcom_df, window_size, (400, 445), no_action_punishment=no_action_punishment),
        create_env(env_type, qcom_df, window_size, (500, 545), no_action_punishment=no_action_punishment),
        create_env(env_type, qcom_df, window_size, (600, 645), no_action_punishment=no_action_punishment),
        create_env(env_type, qcom_df, window_size, (700, 745), no_action_punishment=no_action_punishment),
        create_env(env_type, qcom_df, window_size, (800, 845), no_action_punishment=no_action_punishment),

        # UBER training envs
        create_env(env_type, uber_df, window_size, (800, 845), no_action_punishment=no_action_punishment),
        create_env(env_type, uber_df, window_size, (900, 945), no_action_punishment=no_action_punishment),
        create_env(env_type, uber_df, window_size, (1000, 1045), no_action_punishment=no_action_punishment),
        create_env(env_type, uber_df, window_size, (1100, 1145), no_action_punishment=no_action_punishment),
        create_env(env_type, uber_df, window_size, (1200, 1245), no_action_punishment=no_action_punishment),
    ]


def create_testing_env(env_type: str, no_action_punishment=0):
    '''Create a testing environment based on the type of environment
    :param env_type: str: The type of environment to create
    :return: CombinedEnv: The testing environment created
    '''

    def file_to_path(file):
        return path.join(path.dirname(__file__), f'stocks_data/{file}')
    
    ko_df = load_dataset(file_to_path('PYPL.csv'))
    window_size = 10
    return create_env(env_type, ko_df, window_size, (850, 950), no_action_punishment=no_action_punishment)


def create_custom_env(ticker: str, frame_bound: tuple, no_action_punishment=0):
    '''Create a custom environment based on the ticker and frame bound
    :param ticker: str: The ticker of the stock to use
    :param frame_bound: tuple[int, int]: The frame bound to use for the environment
    :return: CombinedEnv: The custom environment created
    '''
    def file_to_path(file):
        return path.join(path.dirname(__file__), f'stocks_data/{file}')
    
    df = load_dataset(file_to_path(f'{ticker}.csv'))
    window_size = 10
    return create_env(ENV_TYPE, df, window_size, frame_bound, no_action_punishment=no_action_punishment)


# ====================================== Random Picker ======================================

class BufferedRandomPicker:
  def __init__(self, values, buff_len=20):
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

def testing_env(no_action_punishment=0):
    '''Create a testing environment
    :return: CombinedEnv: The testing environment created
    '''
    return create_testing_env(ENV_TYPE, no_action_punishment=no_action_punishment)

def random_train_env_getter(no_action_punishment=0):
    train_envs = create_training_envs(ENV_TYPE, no_action_punishment=no_action_punishment)
    picker = BufferedRandomPicker(train_envs)
    def get_random_train_env():
        '''Get a random training environment
        :return: TFPyEnvironment: The random training environment
        '''
        env = picker.pick_random()
        return env
    return get_random_train_env
