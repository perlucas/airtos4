from __future__ import absolute_import, division, print_function

print('Running TF tests from custom script!')

import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import base64
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import random_tf_policy, policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import Trajectory
from tf_agents.utils import common
from tf_agents.specs import tensor_spec

from tf_agents.policies import py_tf_eager_policy

from envs.macd_env import MacdEnv
from envs.adx_env import AdxEnv
from envs.rsi_env import RsiEnv
from envs.moving_average_env import MovingAverageEnv

from utils import load_dataset

import sys
from datetime import datetime


## Helper functions

def create_env(env_type: str, df, window_size, frame_bound):
    if env_type == 'macd':
        return MacdEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    if env_type == 'rsi':
        return RsiEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    if env_type == 'adx':
        return AdxEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    if env_type == 'mas':
        return MovingAverageEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    raise NotImplementedError('unknown type')


def create_training_envs(env_type: str):
    ko_df = load_dataset('./resources/KO.csv')
    amzn_df = load_dataset('./resources/AMZN.csv')
    amd_df = load_dataset('./resources/AMD.csv')
    pypl_df = load_dataset('./resources/PYPL.csv')
    nflx_df = load_dataset('./resources/NFLX.csv')
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
    ko_df = load_dataset('./resources/KO.csv')
    window_size = 10
    # Expected income: 790 USD
    return create_env(env_type, ko_df, window_size, (2000, 2300))


################### DEFINE VARIATIONS #########################
VARIATIONS = [
    ['test_decay.1.1', (64,64), 1e-8, 3],
    ['test_decay.1.2', (128,64), 1e-8, 3],
    ['test_decay.1.3', (128,128), 1e-8, 3],
    ['test_decay.2.1', (128, 64, 64), 1e-8, 3],
    ['test_decay.2.2', (128, 128, 64), 1e-8, 3],
    ['test_decay.2.3', (128, 128, 128), 1e-8, 3],
]


# ====================================== Execution params definition ======================================
PARAM_NUM_ITERATIONS = 1300
PARAM_ENV_TYPE = 'mas'
PARAM_RUN_ID = 'test_' + str(datetime.now().timestamp())

#PARAM_INITIAL_COLLECT_STEPS = 1000
PARAM_COLLECT_EPISODES_PER_ITERATION = 3
#PARAM_COLLECT_STEPS_PER_ITERATION = 500
PARAM_BATCH_SIZE = 32
PARAM_REPLAY_BUFFER_CAPACITY = 1000

PARAM_LOG_INTERVAL = 50
PARAM_EVAL_EPISODES = 10
PARAM_EVAL_INTERVAL = 50
PARAM_LOG_LEARNING_RATE = 50
PARAM_CHANGE_LEARNING_RATE = 100


# ========================== Assert right tf version being used =======================================
from packaging import version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."


# ====================================== Create environments ======================================

train_py_envs = create_training_envs(PARAM_ENV_TYPE)
train_env_sample = tf_py_environment.TFPyEnvironment(train_py_envs[0])

eval_py_env = create_testing_env(PARAM_ENV_TYPE)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# ====================================== Training ======================================
for VARIATION in VARIATIONS:

    PARAM_SET, PARAM_AGENT_LAYERS, PARAM_LEARNING_RATE, PARAM_N_STEP_UPDATE = VARIATION
    
    # Create logging utilities
    LOG_DIR = "logs/scalars/" + f"{PARAM_SET}_lr{PARAM_LEARNING_RATE}_n{PARAM_N_STEP_UPDATE}"
    file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
    file_writer.set_as_default()

    # ====================================== Create Reinforce Agent and optimizer ======================================
    fc_layer_params = PARAM_AGENT_LAYERS

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env_sample.observation_spec(),
        train_env_sample.action_spec(),
        fc_layer_params=fc_layer_params)
    
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=PARAM_LEARNING_RATE,
        decay_steps=PARAM_CHANGE_LEARNING_RATE,
        decay_rate=10,
        staircase=True
    )
    
    # Create an optimizer with the learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
    
    train_step_counter = tf.Variable(initial_value=0, dtype='int64')
    
    agent = reinforce_agent.ReinforceAgent(
        train_env_sample.time_step_spec(),
        train_env_sample.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)
    agent.initialize()


    # ====================================== Helper for Avg Return ======================================
    
    def compute_avg_return(environment, policy, num_episodes=10):
    
        total_return = 0.0
        for _ in range(num_episodes):
    
            time_step = environment.reset()
            episode_return = 0.0
    
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return
    
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    
    
    # ====================================== Collect Data ======================================
    #random_policy = random_tf_policy.RandomTFPolicy(train_env_sample.time_step_spec(),
    #                                            train_env_sample.action_spec())
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env_sample.batch_size,
        max_length=PARAM_REPLAY_BUFFER_CAPACITY)
    
    
    def collect_episode(environment, policy, num_episodes):
        driver = dynamic_episode_driver.DynamicEpisodeDriver(
            environment,
            py_tf_eager_policy.PyTFEagerPolicy(policy, use_tf_function=True),
            [replay_buffer.add_batch],
            num_episodes=num_episodes)
        #initial_time_step = environment.reset()
        driver.run()


    # ====================================== Training Loop ======================================
    
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)
    
    # Reset the train step.
    agent.train_step_counter.assign(0)
    
    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, PARAM_EVAL_EPISODES)
    returns = [avg_return]
    
    
    def get_random_train_env():
        idx = np.random.randint(len(train_py_envs))
        #return train_py_envs[idx]
        return tf_py_environment.TFPyEnvironment(train_py_envs[idx])
    
    
    train_env = get_random_train_env()
    iterations_per_env = 10
    
    
    for _ in range(PARAM_NUM_ITERATIONS):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_episode(
            train_env, agent.collect_policy, PARAM_COLLECT_EPISODES_PER_ITERATION)
    
        # Sample a batch of data from the buffer and update the agent's network.
        dataset=replay_buffer.as_dataset(sample_batch_size=1)
        #print(dataset)

        
        #iterator = iter(dataset)
        #trajectories, _ = next(iterator)

        #print(trajectories)
        #print(trajectories.step_type)

        iterator = iter(dataset)
        experience, unused_info = next(iterator)
        batched_exp = tf.nest.map_structure(
            lambda t: tf.expand_dims(t, axis=0),
            experience
        )
        train_loss = agent.train(batched_exp)

        
        
        #train_loss = agent.train(experience=trajectories)
        replay_buffer.clear()
    
        step = agent.train_step_counter.numpy()
    
        # Log information
        if step % PARAM_LOG_INTERVAL == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))
            #tf.summary.scalar('loss2', data=train_loss.loss, step=int64(step))
    
    
        # Evaluate agent
        if step % PARAM_EVAL_INTERVAL == 0:
            avg_return = compute_avg_return(
                eval_env, agent.policy, PARAM_EVAL_EPISODES)
            print('step = {0}: Average Return = {1:.2f}'.format(
                step, avg_return))
            #returns.append(avg_return)
            tf.summary.scalar('average return', data=avg_return, step=step)
    
    
        # Change training env
        if step % iterations_per_env == 0:
            train_env = get_random_train_env()

    print(f"Finished execution: {PARAM_SET} {PARAM_LEARNING_RATE}")
