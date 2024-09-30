import os
from packaging import version

import tensorflow as tf
import keras_tuner as kt

from dqn.dqn_agent import DuelingDQNAgent
from utils.tuner_throttler import TunnerThrottler





# Keep using keras-2 (tf-keras) rather than keras-3 (keras)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Check TensorFlow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."


# Training and configuration parameters
PARAM_REPLAY_BUFFER_CAPACITY = 100000
# PARAM_BATCH_SIZE = 64
PARAM_NUM_ITERATIONS = 500
PARAM_COLLECT_STEPS_PER_ITERATION = 200

EXECUTION_ID = '2024-09-28__latest'
PARAM_MAX_TRIALS = 100
PARAM_TRIALS_PER_EXECUTION = 3

# Load the tuner throttler to check if there's a current execution running
tuner_throttler = TunnerThrottler(
    status_file_path=os.path.join(
        os.path.dirname(__file__),
        f"train_dudqn/{EXECUTION_ID}/status.json"),
    trials_per_execution=PARAM_TRIALS_PER_EXECUTION,
    execution_id=EXECUTION_ID
)
tuner_throttler.load()

# Abort if there's a current execution running
if tuner_throttler.is_running():
    print("An execution is already running. Aborting...")
    exit(1)


class AirtosHyperModel(kt.HyperModel):

    def build(self, hp):
        # Compute the number of layers for shared/advantage network
        sa_layers_list = []
        num_layers = hp.Choice("num_layers_sa", [6, 9, 12, 15])
        layer_units = hp.Int("layer_units_sa", min_value=50, max_value=400, step=50)
        for _ in range(num_layers):
            sa_layers_list.append(layer_units)
        SA_AGENT_LAYERS = tuple(sa_layers_list)

        # Compute the number of layers for value network (not used for now)
        # v_layers_list = []
        # num_layers = hp.Choice("num_layers_v", [6, 9, 12, 15])
        # layer_units = hp.Int("layer_units_v", min_value=50, max_value=400, step=50)
        # for _ in range(num_layers):
            # v_layers_list.append(layer_units)
        # V_AGENT_LAYERS = tuple(v_layers_list)

        # Compute optimizer learning rate
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hp.Choice('learning_rate', [7e-6, 3e-5, 7e-5, 3e-4, 7e-4]))

        agent = DuelingDQNAgent(
            shared_advantage_layers=SA_AGENT_LAYERS,
            # value_layers=V_AGENT_LAYERS,
            value_layers=SA_AGENT_LAYERS,
            optimizer=optimizer,
            replay_buffer_capacity=PARAM_REPLAY_BUFFER_CAPACITY
        )

        return agent


    def run(self, hp, model, trial, *args, **kwargs):
        agent = model

        LOG_DIR = os.path.join(
            os.path.dirname(__file__),
            f"train_dudqn/{EXECUTION_ID}/trial_{trial.trial_id}"
        )

        final_stats = agent.train_agent(
            initial_collect_steps=1000,
            batch_size=hp.Choice('batch_size', [32, 64, 128]),
            log_dir=LOG_DIR,
            num_iterations=PARAM_NUM_ITERATIONS,
            collect_steps_per_iteration=PARAM_COLLECT_STEPS_PER_ITERATION
        )

        return final_stats

class AirtosTunner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.run(hp, model, trial=trial, *args, **kwargs)

# total trials to run = already run trials + trials per execution => this makes the tuner to only run the remaining trials
# once it loads the tuner's status from the disk, it'll see how many trials have already been run
# and then it'll calculate the remaining trials to run. So only PARAM_TRIALS_PER_EXECUTION trials will be run
trials_to_run = min(
    tuner_throttler.executed_trials() + PARAM_TRIALS_PER_EXECUTION,
    PARAM_MAX_TRIALS
)


# Abort if there are no more trials to run
if trials_to_run >= PARAM_MAX_TRIALS:
    print("No more trials to run. Aborting...")
    exit(0)

# Set the tuner as running
tuner_throttler.set_running()
tuner_throttler.save()

# Create the tuner
tuner = AirtosTunner(
    hypermodel=AirtosHyperModel(name='airtos4'),
    objective=kt.Objective(name='custom_return', direction='max'),
    max_trials=trials_to_run,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    directory=os.path.join(os.path.dirname(__file__), f'train_dudqn/{EXECUTION_ID}'),
    project_name=f'airtos4_{EXECUTION_ID}',
    tuner_id='airtos4_tuner1',
    overwrite=False,
    executions_per_trial=1,
    allow_new_entries=True,
    tune_new_entries=True
)

# Reload the tuner if there are already executions
if tuner_throttler.executions_count() > 0:
    tuner.reload()

tuner.search_space_summary(extended=True)

tuner.search()

# Unlock next executions
tuner_throttler.set_finished()
tuner_throttler.save()

# Exit if more executions are needed
if tuner_throttler.executed_trials() < PARAM_MAX_TRIALS:
    exit(0)

# Reach here if all trials have been executed
# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=40)
best_values = []
for hp in best_hps:
    print(hp.values)
    best_values.append(hp.values)

# Save best values to file
best_values_file = os.path.join(os.path.dirname(__file__), f"train_dudqn/{EXECUTION_ID}/best_values.txt")
with open(best_values_file, 'w') as f:
    f.write(str(best_values))

print('Finished!')