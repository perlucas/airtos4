# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

import os
from datetime import datetime
import threading

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import keras_tuner as kt

from utils.envs.sb3 import testing_env, random_train_env_getter


eval_env = testing_env(no_action_punishment=0)
get_random_train_env = random_train_env_getter(no_action_punishment=0)

# =============================== General parameters ===============================================
EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')

LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    EXECUTION_ID
)

PARAM_NUM_ITERATIONS = 3000
PARAM_COLLECT_STEPS_PER_ITERATION = 250
PARAM_LOG_INTERVAL_EPISODES = 10
PARAM_EVAL_INTERVAL_EPISODES = 25
PARAM_SWITCH_ENV_INTERVAL = 5 * PARAM_COLLECT_STEPS_PER_ITERATION

# =============================== Switch Environment Wrapper ==============================
class SwitchEnvWrapper(gymnasium.Wrapper):
    
    def __init__(self, env, switch_interval):
        super().__init__(env)
        self.switch_interval = switch_interval
        self.should_switch = False
        self.n_steps = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.n_steps += 1

        if self.n_steps % self.switch_interval == 0:
            self.should_switch = True

        if done and self.should_switch:
            self.should_switch = False
            self.env = get_random_train_env()
            # print(f'Switched environment at step {self.n_steps}')
        
        return obs, reward, done, truncated, info

# =============================== Callbacks ===============================================
class EvaluateAndLogCallback(BaseCallback):
    '''Custom callback to evaluate the policy and log the average return mean'''

    def __init__(self, eval_env, log_interval, eval_interval, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.last_n_avg_returns = []
        self.last_n_avg_returns_len = 4

    def _on_step(self) -> bool:

        if self.n_calls % self.log_interval == 0:
            print(f'Step {self.n_calls}')
        
        if self.n_calls % self.eval_interval != 0:
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
        print(f'Number of active threads: {threading.active_count()}')

        # Log the average return in the buffer
        self.last_n_avg_returns.append(total_reward)
        if len(self.last_n_avg_returns) > self.last_n_avg_returns_len:
            self.last_n_avg_returns.pop(0)
        return True
    
    def _on_training_end(self) -> None:
        return True

    def get_avg_return(self):
        return sum(self.last_n_avg_returns) / len(self.last_n_avg_returns)
    
    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass

# =============================== Keras Tuner ===============================================
class AirtosHyperModel(kt.HyperModel):

    def build(self, hp):
        # Compute the number of layers for the DQN agent
        layers_list = []
        num_layers = hp.Int("num_layers", min_value=4, max_value=24, step=4)
        layer_units = hp.Choice("layer_units", [50, 100, 200, 400, 500])
        for _ in range(num_layers):
            layers_list.append(layer_units)
        policy_kwargs = dict(net_arch=layers_list)

        # Compute optimizer learning rate
        learning_rate = hp.Float('learning_rate', min_value=1e-7, max_value=1e-5, step=1.237e-6)

        # Create model
        env = SwitchEnvWrapper(env=get_random_train_env(), switch_interval=PARAM_SWITCH_ENV_INTERVAL)
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            batch_size=128,
            tensorboard_log=LOG_DIR)
        return model


    def run(self, hp, model, trial, *args, **kwargs):
        
        # Will use this custom callback to compute metric for Tuner
        custom_evaluate_callback = EvaluateAndLogCallback(
            eval_env,
            eval_interval=PARAM_EVAL_INTERVAL_EPISODES * PARAM_COLLECT_STEPS_PER_ITERATION,
            log_interval=PARAM_LOG_INTERVAL_EPISODES * PARAM_COLLECT_STEPS_PER_ITERATION)

        # SB3 callback to evaluate the policy and log in TB
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=2,
            eval_freq=PARAM_EVAL_INTERVAL_EPISODES * PARAM_COLLECT_STEPS_PER_ITERATION)

        model.learn(
            total_timesteps=PARAM_COLLECT_STEPS_PER_ITERATION * PARAM_NUM_ITERATIONS,
            reset_num_timesteps=False,
            callback=[custom_evaluate_callback, eval_callback],
            tb_log_name=f'trial_{trial.trial_id}')

        return { 'avg_return': custom_evaluate_callback.get_avg_return() }

class AirtosTunner(kt.BayesianOptimization):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        model = self.hypermodel.build(hp)
        return self.hypermodel.run(hp, model, trial=trial, *args, **kwargs)
    

tuner = AirtosTunner(
    hypermodel=AirtosHyperModel(name='airtos4'),
    objective=kt.Objective(name='avg_return', direction='max'),
    max_trials=140,
    max_retries_per_trial=0,
    max_consecutive_failed_trials=3,
    directory=os.path.join(os.path.dirname(__file__), EXECUTION_ID),
    project_name=f'airtos4_{EXECUTION_ID}',
    tuner_id='airtos4_tuner1',
    overwrite=False,
    executions_per_trial=2,
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