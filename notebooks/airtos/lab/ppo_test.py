# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.envs.sb3 import create_training_envs, create_custom_env, ENV_TYPE, create_testing_env

def relative_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

# training_envs = create_training_envs(ENV_TYPE, no_action_punishment=0)
training_envs = [create_testing_env(ENV_TYPE, no_action_punishment=0)]
model = PPO.load(relative_path("5-N-ppo/trial_4_best_model.zip"))

i = 0
for env in training_envs:
    obs, info = env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            break
    i += 1

    env.save_render(relative_path('envs-img/env_{}'.format(i)))
