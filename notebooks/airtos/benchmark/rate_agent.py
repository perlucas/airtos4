# Preparation to use gymnasium for SB3
# This should be before any SB# import
import sys
import gymnasium
sys.modules["gym"] = gymnasium
# End of preparation

import sys
import os

from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import RecurrentPPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.envs.sb3 import create_custom_env

def extract_model_files(filename: str):
    if filename.endswith(".zip"):
        return [filename]
    return [os.path.join(filename, f) for f in os.listdir(filename) if f.endswith(".zip")]

def load_model(filename: str):
    if "a2c" in filename:
        return A2C.load(filename)
    elif "rec_ppo" in filename:
        return RecurrentPPO.load(filename)
    elif "dqn" in filename:
        return DQN.load(filename)
    else:
        return PPO.load(filename)
    
def compute_score(total_profit):
    if total_profit > 10:
        return 20
    elif total_profit > 0:
        return 10
    else:
        return -10

def evaluate_model(model, env, ticker, render_dir=None):
    total_reward = 0
    obs, info = env.reset()
    while True:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # print(obs)
        total_reward += reward

        if done or truncated:
            break
    
    if render_dir:
        env.save_render(f"{render_dir}/{ticker}")

    return {
        "total_profit": info['profit'],
        "total_reward": total_reward,
        "score": compute_score(info['profit']),
        "profitable": info['profit'] > 0,
    }

def evaluate_model_lstm(model, env, ticker, render_dir=None):
    lstm_states = None
    episode_start = True
    total_reward = 0
    obs, info = env.reset()
    while True:
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_start = done
        total_reward += reward

        if done or truncated:
            break
    
    if render_dir:
        env.save_render(f"{render_dir}/{ticker}")

    return {
        "total_profit": info['profit'],
        "total_reward": total_reward,
        "score": compute_score(info['profit']),
        "profitable": info['profit'] > 0,
    }

# Evaluation tickers
KNOWN_ENVS = [
    # 5 known tickers (using unknown frame bounds)
    ("CRM", (1100, 1145)),
    ("AMD", (1100, 1145)),
    ("PYPL", (1100, 1145)),
    ("NVDA", (1050, 1095)),
    ("QCOM", (1050, 1095))
]

UNKNOWN_ENVS = [
    # 5 unknown tickers
    ("TSLA", (1200, 1245)),
    ("DELL", (1200, 1245)),
    ("IBM", (1200, 1245)),
    ("ACN", (1200, 1245)),
    ("DDOG", (1200, 1245)),
    ("BTC", (300, 345)),
]

OTHER_SECTOR_ENVS = [
    # 5 unknown tickers belonging to other sectors
    ("KO", (2500, 2545)),
    ("XOM", (1200, 1245)),
    ("WMT", (1200, 1245)),
    ("NKE", (1200, 1245)),
    ("BKNG", (1200, 1245))
]

def evaluate_group(env_specs, model, output_dir = None, use_lstm = False):
    num_profitables = 0
    score = 0
    total_profit = 0
    budget = 0
    results = []
    for specs in env_specs:
        ticker, frame_bounds = specs
        env = create_custom_env(ticker, frame_bounds, no_action_punishment=0)
        result = evaluate_model(model, env, ticker, output_dir) if not use_lstm else evaluate_model_lstm(model, env, ticker, output_dir)
        if result['profitable']:
            num_profitables += 1
        score += result['score']
        total_profit += result['total_profit']
        results.append({
            **result,
            "ticker": ticker,
        })

    perc_profitables = num_profitables / len(env_specs)
    return {
        "results": results,
        "num_profitables": num_profitables,
        "perc_profitables": perc_profitables,
        "score": score,
        "total_profit": total_profit,
        "budget": budget,
    }

# Evaluate the model
GROUPS_TO_EVALUATE = [
    {
        "name": "Known tickers",
        "envs": KNOWN_ENVS
    },
    {
        "name": "Unknown tickers",
        "envs": UNKNOWN_ENVS
    },
    {
        "name": "Other sectors",
        "envs": OTHER_SECTOR_ENVS
    }
]

def evaluate_all(model, use_lstm=False):
    """Mimics the main evaluation loop, but returns the results instead of printing them"""
    total_profitables = 0
    perc_avg = 0
    total_len = 0

    for group in GROUPS_TO_EVALUATE:
        group_results = evaluate_group(group['envs'], model, output_dir=None, use_lstm=use_lstm)
        total_profitables += group_results['num_profitables']
        perc_avg += group_results['perc_profitables']
        total_len += len(group['envs'])

    return {
        "perc_profitables": total_profitables / total_len,
        "avg_income": perc_avg / total_len,
    }

# Main evaluation loop: load the model and evaluate it on the different groups, printing the results
if __name__ == "__main__":
    # Check if the number of arguments is correct
    if len(sys.argv) != 3:
        print("Usage: python rate_agent.py AGENT_FILENAME [OUTPUT_DIRECTORY]")
        sys.exit(1)

    AGENT_FILENAME = os.path.join(os.path.dirname(__file__), sys.argv[1])
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), sys.argv[2]) if sys.argv[2] else None

    model_filenames = extract_model_files(AGENT_FILENAME)

    for model_filename in model_filenames:
        model = load_model(model_filename)
        print(f"Evaluating model {model_filename}")

        total_profitables = 0
        total_score = 0

        for group in GROUPS_TO_EVALUATE:
            group_results = evaluate_group(group['envs'], model, OUTPUT_DIR)
            this_profitables = group_results['num_profitables']
            this_score = group_results['score']
            profit = group_results['total_profit']
            budget = group_results['budget']

            total_profitables += this_profitables
            total_score += this_score

            for ticker_result in group_results['results']:
                print(f">>>> [{ticker_result['ticker']}] Profit: {ticker_result['total_profit']}, Score: {ticker_result['score']}")

            print(f"[{group['name']}] Profitables %: {(this_profitables/len(group['envs']))*100}, Score: {this_score}, Score Avg: {this_score/len(group['envs'])}, Total profit: {profit}, Budget: {budget}")

        LEN_TOTAL = sum([len(group['envs']) for group in GROUPS_TO_EVALUATE])
        print(f"Total profitables %: {(total_profitables/LEN_TOTAL)*100}, Total score: {total_score}, Total score avg: {total_score/LEN_TOTAL}")
        print("\n\n")