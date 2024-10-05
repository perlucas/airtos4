# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

import os
from datetime import datetime
import threading

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import keras_tuner as kt

from utils.envs.sb3 import eval_env, get_random_train_env


# =============================== General parameters ===============================================
EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')

LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    '2024-10-05'
    # EXECUTION_ID
)

PARAM_NUM_ITERATIONS = 1000
PARAM_COLLECT_STEPS_PER_ITERATION = 250
PARAM_INITIAL_COLLECT_STEPS = 1000
PARAM_REPLAY_BUFFER_CAPACITY = 100000
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
        num_layers = hp.Choice("num_layers", [6, 9, 12, 15])
        layer_units = hp.Int("layer_units", min_value=50, max_value=400, step=50)
        for _ in range(num_layers):
            layers_list.append(layer_units)
        policy_kwargs = dict(net_arch=layers_list)

        # Compute optimizer learning rate
        learning_rate = hp.Choice('learning_rate', [7e-6, 3e-5, 7e-5, 3e-4, 7e-4])

        # Create model
        env = SwitchEnvWrapper(env=get_random_train_env(), switch_interval=PARAM_SWITCH_ENV_INTERVAL)
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            buffer_size=PARAM_REPLAY_BUFFER_CAPACITY,
            learning_starts=PARAM_INITIAL_COLLECT_STEPS,
            gamma=0.99,
            batch_size=hp.Choice('batch_size', [32, 64, 128]),
            train_freq=(1, 'episode'),
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
    

# tuner = AirtosTunner(
#     hypermodel=AirtosHyperModel(name='airtos4'),
#     objective=kt.Objective(name='avg_return', direction='max'),
#     max_trials=100,
#     max_retries_per_trial=0,
#     max_consecutive_failed_trials=3,
#     directory=os.path.join(os.path.dirname(__file__), EXECUTION_ID),
#     project_name=f'airtos4_{EXECUTION_ID}',
#     tuner_id='airtos4_tuner1',
#     overwrite=False,
#     executions_per_trial=1,
#     allow_new_entries=True,
#     tune_new_entries=True
# )


# tuner.search_space_summary(extended=True)

# tuner.search()

# # Get best hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials=40)
# best_values = []
# for hp in best_hps:
#     print(hp.values)
#     best_values.append(hp.values)

# # Save best values to file
# best_values_file = os.path.join(os.path.dirname(__file__), f"{EXECUTION_ID}/best_values.txt")
# with open(best_values_file, 'w') as f:
#     f.write(str(best_values))



def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# Create model
policy_kwargs = dict(net_arch=[50, 100, 100])
env = SwitchEnvWrapper(env=get_random_train_env(), switch_interval=PARAM_SWITCH_ENV_INTERVAL)
# model = DQN(
#     'MlpPolicy',
#     env,
#     learning_rate=linear_schedule(0.05),
#     policy_kwargs=policy_kwargs,
#     buffer_size=PARAM_REPLAY_BUFFER_CAPACITY,
#     learning_starts=PARAM_INITIAL_COLLECT_STEPS,
#     gamma=0.99,
#     exploration_fraction=0.3,
#     batch_size=64,
#     train_freq=(3, 'episode'),
#     tensorboard_log=LOG_DIR)
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=linear_schedule(0.001),
    policy_kwargs=policy_kwargs,
    gamma=0.99,
    batch_size=64,
    tensorboard_log=LOG_DIR)


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
    total_timesteps=PARAM_COLLECT_STEPS_PER_ITERATION * 5000,
    reset_num_timesteps=False,
    callback=[eval_callback],
    tb_log_name=f'test_20_ppo')




print('Finished!')