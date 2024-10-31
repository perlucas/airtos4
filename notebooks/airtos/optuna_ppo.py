# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
import optuna

from utils.envs.sb3 import testing_env, random_train_env_getter


eval_env = testing_env(no_action_punishment=0)
get_random_train_env = random_train_env_getter(no_action_punishment=0)

# =============================== General parameters ===============================================
EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')

LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    EXECUTION_ID
)

PARAM_NUM_ITERATIONS = 6000
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


# =============================== Init and Run Tuner ===============================================
RUNS_PER_TRIAL = 5

def objective(trial):
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e-2)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.01)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    
    num_layers = trial.suggest_categorical("num_layers", [3, 4, 5, 8, 12])
    layer_units = trial.suggest_categorical("layer_units", [15, 25, 50, 75, 100])
    layers_list = [layer_units] * num_layers
    policy_kwargs = dict(net_arch=layers_list)

    env = SwitchEnvWrapper(env=get_random_train_env(), switch_interval=PARAM_SWITCH_ENV_INTERVAL)
    def get_model():
        return PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            gamma=0.99,
            batch_size=128,
            seed=42,
            ent_coef=ent_coef,
            clip_range=clip_range,
            tensorboard_log=LOG_DIR)
    
    def train_model(model):
        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=8300, verbose=1)
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=2,
            callback_on_new_best=callback_on_best,
            callback_after_eval=stop_train_callback,
            eval_freq=PARAM_EVAL_INTERVAL_EPISODES * PARAM_COLLECT_STEPS_PER_ITERATION)

        model.learn(
            total_timesteps=PARAM_COLLECT_STEPS_PER_ITERATION * PARAM_NUM_ITERATIONS,
            reset_num_timesteps=True,
            callback=[eval_callback],
            tb_log_name=f'trial_{trial.number}')
    
    total_eval_results = []
    for _ in range(RUNS_PER_TRIAL):
        model = get_model()
        train_model(model)
        eval_results = evaluate_policy(model, eval_env, n_eval_episodes=2)
        total_eval_results.append(eval_results[0])
        model = None
    
    return sum(total_eval_results) / len(total_eval_results)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3000, n_jobs=1)

print('Finished!')