# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.envs.sb3 import testing_env, create_custom_env, ENV_TYPE

eval_env = create_custom_env('CSCO', (1000, 1120), no_action_punishment=0)

model = RecurrentPPO.load("best_rec_ppo")

total_reward = 0
obs, info = eval_env.reset()
while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = eval_env.step(action)
    total_reward += reward

    if done or truncated:
        break

eval_env.save_render("rec_ppo_test_CSCO")
print(f"Total reward: {total_reward}")
print(f"Info: {info}")