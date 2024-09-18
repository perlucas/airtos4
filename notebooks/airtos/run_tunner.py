import os
from packaging import version
from datetime import datetime

import tensorflow as tf
import keras_tuner as kt

from utils.envs import get_random_train_env, train_env_sample, eval_env
from utils.training import get_random_policy
from c51.c51_agent import C51Agent






# Keep using keras-2 (tf-keras) rather than keras-3 (keras)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# Check TensorFlow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."


# Training and configuration parameters
PARAM_NUM_ATOMS = 51
PARAM_MIN_Q_VALUE = -1000
PARAM_MAX_Q_VALUE = 1000
PARAM_N_STEP_UPDATE = 2
PARAM_REPLAY_BUFFER_CAPACITY = 100000
PARAM_BATCH_SIZE = 64
PARAM_NUM_ITERATIONS = 40
PARAM_COLLECT_STEPS_PER_ITERATION = 200


EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')


class AirtosHyperModel(kt.HyperModel):

    def build(self, hp):
        # Compute the number of layers for the DQN agent
        layers_list = []
        num_layers = hp.Choice("num_layers", [6, 9, 12, 15])
        layer_units = hp.Int("layer_units", min_value=50, max_value=400, step=50)
        for _ in range(num_layers):
            layers_list.append(layer_units)
        AGENT_LAYERS = tuple(layers_list)

        # Compute optimizer learning rate
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=hp.Choice('learning_rate', [7e-5, 3e-4, 7e-4, 3e-3, 7e-3]))

        agent = C51Agent(
            layers=AGENT_LAYERS,
            num_atoms=PARAM_NUM_ATOMS,
            optimizer=optimizer,
            min_q_value=PARAM_MIN_Q_VALUE,
            max_q_value=PARAM_MAX_Q_VALUE,
            n_step_update=PARAM_N_STEP_UPDATE,
            replay_buffer_capacity=PARAM_REPLAY_BUFFER_CAPACITY
        )

        return agent


    def run(self, hp, model, trial, *args, **kwargs):
        agent = model

        LOG_DIR = os.path.join(
            os.path.dirname(__file__),
            f"{EXECUTION_ID}/trial_{trial.trial_id}"
        )

        avg_return = agent.train_agent(
            initial_collect_steps=1000,
            batch_size=hp.Choice('batch_size', [32, 64, 128]),
            log_dir=LOG_DIR,
            num_iterations=PARAM_NUM_ITERATIONS,
            collect_steps_per_iteration=PARAM_COLLECT_STEPS_PER_ITERATION
        )

        return {
            'avg_return': avg_return
        }

class AirtosTunner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.run(hp, model, trial=trial, *args, **kwargs)
    

tuner = AirtosTunner(
    hypermodel=AirtosHyperModel(name='airtos4'),
    objective=kt.Objective(name='avg_return', direction='max'),
    max_trials=2,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    directory=os.path.join(os.path.dirname(__file__),EXECUTION_ID),
    project_name=f'airtos4_{EXECUTION_ID}',
    tuner_id='airtos4_tuner1',
    overwrite=False,
    executions_per_trial=1,
    allow_new_entries=False,
    tune_new_entries=True
)

tuner.search_space_summary(extended=True)

tuner.search()

best_hps = tuner.get_best_hyperparameters()
print(best_hps)
for hp in best_hps:
    print(hp.values)
print('Finished!')