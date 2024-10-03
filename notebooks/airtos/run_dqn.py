# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

import os
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import keras_tuner as kt

from utils.envs.sb3 import train_envs, eval_env, get_random_train_env


# =============================== General parameters ===============================================
EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')

LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    EXECUTION_ID
)

PARAM_NUM_ITERATIONS = 500
PARAM_COLLECT_STEPS_PER_ITERATION = 250
PARAM_INITIAL_COLLECT_STEPS = 1000
PARAM_REPLAY_BUFFER_CAPACITY = 100000
PARAM_LOG_INTERVAL_EPISODES = 10
PARAM_EVAL_INTERVAL_EPISODES = 25
PARAM_SWITCH_ENV_INTERVAL = 5

# =============================== Callbacks ===============================================
class EvaluateAndLogCallback(BaseCallback):
    '''Custom callback to evaluate the policy and log the average return mean'''

    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.episode = 0
        self.last_n_avg_returns = []
        self.last_n_avg_returns_len = 4

    def _on_training_end(self) -> None:
        if self.episode % PARAM_EVAL_INTERVAL_EPISODES != 0:
            return True

        # Evaluate policy and log avg_return
        obs, info = self.eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = self.model.predict(obs, deterministic=True)
            obs, _reward, done, _truncated, info = self.eval_env.step(action)
            total_reward += _reward
        print(f'Step {self.n_calls} - Total reward: {total_reward}')

        # Log the average return in the buffer
        self.last_n_avg_returns.append(total_reward)
        if len(self.last_n_avg_returns) > self.last_n_avg_returns_len:
            self.last_n_avg_returns.pop(0)
        return True
    
    def get_avg_return(self):
        return sum(self.last_n_avg_returns) / len(self.last_n_avg_returns)
    
    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

# =============================== Keras Tuner ===============================================
class AirtosHyperModel(kt.HyperModel):

    def build(self, hp):
        # Compute the number of layers for the DQN agent
        layers_list = []
        num_layers = hp.Choice("num_layers", [6, 9, 12, 15])
        layer_units = hp.Int("layer_units", min_value=50, max_value=400, step=50)
        for _ in range(num_layers):
            layers_list.append(layer_units)
        policy_kwargs = dict(net_arch=layers_list)

        # Compute optimizer learning rate
        learning_rate = hp.Choice('learning_rate', [7e-6, 3e-5, 7e-5, 3e-4, 7e-4])

        # Create model
        env = train_envs[0] # Start with the first environment
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            buffer_size=PARAM_REPLAY_BUFFER_CAPACITY,
            learning_starts=PARAM_INITIAL_COLLECT_STEPS,
            gamma=0.99,
            batch_size=hp.Choice('batch_size', [32, 64, 128]),
            tensorboard_log=LOG_DIR)
        return model


    def run(self, hp, model, trial, *args, **kwargs):
        agent = model
        
        # Will use this custom callback to compute metric for Tuner
        custom_evaluate_callback = EvaluateAndLogCallback(eval_env)
        custom_evaluate_callback.episode = 0

        # SB3 callback to evaluate the policy and log in TB
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=2,
            eval_freq=PARAM_EVAL_INTERVAL_EPISODES * PARAM_COLLECT_STEPS_PER_ITERATION)

        for episode in range(PARAM_NUM_ITERATIONS):
            # Train episode
            agent.learn(
                total_timesteps=PARAM_COLLECT_STEPS_PER_ITERATION,
                reset_num_timesteps=False,
                callback=[custom_evaluate_callback, eval_callback],
                tb_log_name=f'trial_{trial.trial_id}')
            
            if episode % PARAM_LOG_INTERVAL_EPISODES == 0:
                print(f'Episode {episode} done!')

            # Switch training environment
            if episode % PARAM_SWITCH_ENV_INTERVAL == 0:
                env = get_random_train_env()
                agent.set_env(env)
            
            custom_evaluate_callback.episode += 1

        return { 'avg_return': custom_evaluate_callback.get_avg_return() }

class AirtosTunner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.run(hp, model, trial=trial, *args, **kwargs)
    

tuner = AirtosTunner(
    hypermodel=AirtosHyperModel(name='airtos4'),
    objective=kt.Objective(name='avg_return', direction='max'),
    max_trials=100,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    directory=os.path.join(os.path.dirname(__file__), EXECUTION_ID),
    project_name=f'airtos4_{EXECUTION_ID}',
    tuner_id='airtos4_tuner1',
    overwrite=False,
    executions_per_trial=1,
    allow_new_entries=True,
    tune_new_entries=True
)


tuner.search_space_summary(extended=True)

tuner.search()

# Get best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=40)
best_values = []
for hp in best_hps:
    print(hp.values)
    best_values.append(hp.values)

# Save best values to file
best_values_file = os.path.join(os.path.dirname(__file__), f"{EXECUTION_ID}/best_values.txt")
with open(best_values_file, 'w') as f:
    f.write(str(best_values))

print('Finished!')