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

from utils.envs.sb3 import testing_env, random_train_env_getter, create_custom_env

# =============================== General parameters ===============================================
eval_env = testing_env(no_action_punishment=0)
# eval_env = create_custom_env('PYPL', (1200, 1245), no_action_punishment=0)
get_random_train_env = random_train_env_getter(no_action_punishment=0)

EXECUTION_ID = datetime.now().strftime('%Y-%m-%d_%H%M%S')

LOG_DIR = os.path.join(
    os.path.dirname(__file__),
    EXECUTION_ID
)

PARAM_NUM_ITERATIONS = 8000
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
        
        return obs, reward, done, truncated, info

# Hyperparameters
layers_list = [4] * 50
learning_rate = 3e-6

policy_kwargs = dict(net_arch=layers_list)

# def lr_schedule(progress):
#     if progress <= 0.25:
#         return 5e-5
#     if progress <= 0.5:
#         return 1e-5
#     if progress <= 0.75:
#         return 3e-6
#     return 5e-7
# learning_rate=lr_schedule

# Create model
# env = SwitchEnvWrapper(env=get_random_train_env(), switch_interval=PARAM_SWITCH_ENV_INTERVAL)
env = testing_env(no_action_punishment=2)

def get_model():
    return PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        gamma=0.99,
        batch_size=128,
        # seed=42,
        # ent_coef=0.00615,
        # clip_range=0.228,
        normalize_advantage=True,
        tensorboard_log=LOG_DIR)


best_mean = 0
for i in range(3):
    model = get_model()
    
    # SB3 callback to evaluate the policy and log in TB
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=550, verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10, verbose=1)
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
        tb_log_name=f'training_{EXECUTION_ID}')
    
    eval_results = evaluate_policy(model, eval_env, n_eval_episodes=2)
    mean = eval_results[0]

    if mean > best_mean:
        best_mean = mean
        model.save(os.path.join(LOG_DIR, 'model'))
        print(f'Saved model with mean reward {mean}')