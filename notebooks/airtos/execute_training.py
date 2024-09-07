import os

from envs.combined_env import CombinedEnv
from utils import load_dataset

import tensorflow as tf
from packaging import version

from tf_agents.environments import tf_py_environment
import numpy as np

import json

from datetime import datetime
import numpy as np
import keras_tuner as kt
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import categorical_q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import matplotlib.pyplot as plt


# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'


# ====================================== Create environments ======================================

def create_env(env_type: str, df, window_size, frame_bound):

    if env_type == 'com':
        return CombinedEnv(df=df, window_size=window_size, frame_bound=frame_bound)

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
    return create_env(env_type, ko_df, window_size, (2000, 2300))



# Check TensorFlow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."



# ====================================== Create environments ======================================
PARAM_ENV_TYPE = 'com'

train_py_envs = create_training_envs(PARAM_ENV_TYPE)
train_env_sample = tf_py_environment.TFPyEnvironment(train_py_envs[0])

eval_py_env = create_testing_env(PARAM_ENV_TYPE)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# ====================================== Random Picker ======================================

class BufferedRandomPicker:
  def __init__(self, values, buff_len=5):
    self.__buffer = []
    self.__values = values
    for i in range(buff_len):
      self.__buffer.append(values[i])

  def pick_random(self):
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

# Pick random environment
picker = BufferedRandomPicker(train_py_envs)
def get_random_train_env():
  env = picker.pick_random()
  return tf_py_environment.TFPyEnvironment(env)


# ====================================== Load hyperparameters ======================================
hyperparameter_set_ids = [
  '2024-08-26_120829@trial_22',
  '2024-08-26_120829@trial_13',
  '2024-08-26_120829@trial_20',
  '2024-08-26_120829@trial_16',
  '2024-08-26_120829@trial_38',
  '2024-08-26_120829@trial_23',
  '2024-08-26_120829@trial_24',
  '2024-08-26_120829@trial_49',
  '2024-08-26_120829@trial_08',
  '2024-08-26_120829@trial_12',
  '2024-08-26_120829@trial_14',
  '2024-08-26_120829@trial_45'
]

sets_filename = './output_analysis.json'

with open(sets_filename) as f:
  hp_sets = json.load(f)

hyperparameter_sets = []
for hps_id in hyperparameter_set_ids:
  a_set = next(filter(lambda s:s['runId'] == hps_id, hp_sets))
  hyperparameter_sets.append(a_set)

hp_sets = None


# ====================================== Hyperparameter Model ======================================

PARAM_NUM_ATOMS = 51
PARAM_MIN_Q_VALUE = -1000
PARAM_MAX_Q_VALUE = 1000

PARAM_REPLAY_BUFFER_CAPACITY = 100000
PARAM_INITIAL_COLLECT_STEPS = 1000
PARAM_BATCH_SIZE = 64
PARAM_N_STEP_UPDATE = 25

PARAM_NUM_ITERATIONS = 6000
PARAM_COLLECT_STEPS_PER_ITERATION = 500

PARAM_LOG_INTERVAL = 10
PARAM_EVAL_EPISODES = 2
PARAM_EVAL_INTERVAL = 25

EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')

# ====================================== Helper for Avg Return ======================================
def compute_avg_return(environment, policy, num_episodes=2):
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

# Random policy
random_policy = random_tf_policy.RandomTFPolicy(train_env_sample.time_step_spec(), train_env_sample.action_spec())

# ====================================== Helper for Simulation ======================================
scenario_raw_env = create_env(
    PARAM_ENV_TYPE,
    load_dataset('./resources/AMD.csv'),
    10,
    (2000, 2300)
)
scenario_env = tf_py_environment.TFPyEnvironment(scenario_raw_env)

def test_trading_scenario(policy, filename):
  time_step = scenario_env.reset()
  while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = scenario_env.step(action_step.action)
  plt.figure(figsize=(15, 8))
  scenario_raw_env.save_render(filename)


class Airtos2HyperModel(kt.HyperModel):
    def __init__(self, *args, **kwargs):
        super(Airtos2HyperModel, self).__init__(*args, **kwargs)
        self.__use_predef = False

    def use_predef(self):
        self.__use_predef = True

    def build(self, hp):
        # Compute the number of layers for the DQN agent
        print(hp)
        print(hp.values)
        layers_list = []
        num_layers = hp.get('n_layers') if self.__use_predef else hp.Int("n_layers", min_value=3, max_value=12)
        #num_layers = hp.get('n_layers') # hp.Int("n_layers", min_value=3, max_value=12)
        for i in range(num_layers):
            units = hp.get(f'units_{i}') if self.__use_predef else hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
            layers_list.append(units)
            # layers_list.append(hp.Int(f'units_{i}', min_value=32, max_value=512, step=32))
        AGENT_LAYERS = tuple(layers_list)

        categorical_q_net = categorical_q_network.CategoricalQNetwork(
            train_env_sample.observation_spec(),
            train_env_sample.action_spec(),
            num_atoms=PARAM_NUM_ATOMS,
            fc_layer_params=AGENT_LAYERS)

        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hp.Choice('learning_rate', [5e-7, 5e-6, 5e-5, 5e-4, 5e-3]))
        l_rate = hp.get('learning_rate') if self.__use_predef else hp.Choice('learning_rate', [5e-7, 5e-6, 5e-5, 5e-4, 5e-3])
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate)

        train_step_counter = tf.Variable(initial_value=0, dtype='int64')

        agent = categorical_dqn_agent.CategoricalDqnAgent(
            train_env_sample.time_step_spec(),
            train_env_sample.action_spec(),
            categorical_q_network=categorical_q_net,
            optimizer=optimizer,
            min_q_value=PARAM_MIN_Q_VALUE,
            max_q_value=PARAM_MAX_Q_VALUE,
            n_step_update=PARAM_N_STEP_UPDATE,
            # td_errors_loss_fn=common.element_wise_squared_loss, use default error
            gamma=0,
            train_step_counter=train_step_counter)
        agent.initialize()

        return agent


    def run(self, hp, model, trial, *args, **kwargs):
        agent = model
        # ====================================== Collect Initial Data ======================================
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env_sample.batch_size,
            max_length=PARAM_REPLAY_BUFFER_CAPACITY)

        def collect_step(environment, policy):
            time_step = environment.current_time_step()
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

        # Collect initial data, using random policy
        for _ in range(PARAM_INITIAL_COLLECT_STEPS):
            collect_step(train_env_sample, random_policy)

        # Dataset generates trajectories with shape [BxTx...] where
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=PARAM_BATCH_SIZE,
            num_steps=PARAM_N_STEP_UPDATE + 1).prefetch(3)

        iterator = iter(dataset)

        # ====================================== Training Loop ======================================

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        agent.train = common.function(agent.train)

        # Reset the train step.
        agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(eval_env, agent.policy, PARAM_EVAL_EPISODES)

        train_env = get_random_train_env()
        iterations_per_env = 5

        # Create logging utilities
        LOG_DIR = f"/content/{EXECUTION_ID}/trial_" + trial.trial_id
        CHECKPOINT_DIR = f"/content/{EXECUTION_ID}/check_" + trial.trial_id

        checkpoint = tf.train.Checkpoint(agent=agent)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=1)

        file_writer = tf.summary.create_file_writer(LOG_DIR + "/metrics")
        file_writer.set_as_default()

        for _ in range(PARAM_NUM_ITERATIONS):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(PARAM_COLLECT_STEPS_PER_ITERATION):
                collect_step(train_env, agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience)

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
                tf.summary.scalar('average return', data=avg_return, step=step)


            # Change training env
            if step % iterations_per_env == 0:
                train_env = get_random_train_env()

        print("Finished execution!")
        checkpoint_manager.save()
        print("Running simulation...")
        test_trading_scenario(agent.policy, f'/content/{EXECUTION_ID}/test_{trial.trial_id}.jpg')

        avg_return = compute_avg_return(eval_env, agent.policy, PARAM_EVAL_EPISODES)

        return {
            'avg_return': avg_return
        }


# ====================================== Hyperparameter Tuning ======================================
class Airtos2Tunner(kt.RandomSearch):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.run(hp, model, trial=trial, *args, **kwargs)


class Airtos2PredefTuner(kt.Tuner):
    def __init__(self, hyperparameter_sets, *args, **kwargs):
        super(Airtos2PredefTuner, self).__init__(*args, **kwargs)
        self.hyperparameter_sets = hyperparameter_sets
        self.current_set_index = 0

    def run_trial(self, trial, *args, **kwargs):
        # Get the current hyperparameter set
        hp_values = self.hyperparameter_sets[self.current_set_index]
        self.current_set_index += 1

        # Update trial's hyperparameters
        print(hp_values)
        print(hp_values.items())
        for key, value in hp_values.items():
            if key == 'units':
              units = hp_values['units']
              for i in range(len(units)):
                trial.hyperparameters.values[f'units_{i}'] = units[i]
            else:
              trial.hyperparameters.values[key] = value

        self.hypermodel.use_predef()
        model = self.hypermodel.build(trial.hyperparameters)
        return self.hypermodel.run(trial.hyperparameters, model, trial=trial, *args, **kwargs)


    def get_best_hyperparameters(self, num_trials=1):
        # Return all hyperparameter sets since we control trials explicitly
        return [self.hyperparameter_sets[0]]

    def get_best_models(self, num_models=1):
        # Implement as needed
        pass

# ====================================== Run Tuning ======================================
len(hyperparameter_sets)

tuner = Airtos2PredefTuner(
    hyperparameter_sets=hyperparameter_sets,
    oracle=kt.oracles.RandomSearchOracle(
        objective=kt.Objective(name='avg_return', direction='max'),
        max_trials=len(hyperparameter_sets),
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ),
    hypermodel=Airtos2HyperModel(name='airtos4'),
    directory='/content',
    project_name=f'airtos4_{EXECUTION_ID}',
    tuner_id='airtos4_tuner1',
    overwrite=False,
    executions_per_trial=1,
)

print(f'Will run {len(hyperparameter_sets)} configs')

tuner.search_space_summary(extended=True)

tuner.search()

best_hp = tuner.get_best_hyperparameters()[0]
print('Finished!')