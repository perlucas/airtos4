import threading
from queue import Queue

import numpy as np

from utils.envs import get_random_train_env, eval_env
from utils.training import compute_avg_return_a3c
from .a3c_agent import A3CAgent

class TrainingWorker(threading.Thread):
    def __init__(self,
                 agent: A3CAgent,
                 jobs_queue: Queue,
                 stats_buffer: list[dict],
                 lock: threading.Lock):
        threading.Thread.__init__(self)
        self.env = None
        self.agent = agent
        self.jobs_queue = jobs_queue
        self.stats_buffer = stats_buffer
        self.lock = lock

    def run(self):
        # Keep working while there are jobs in the queue
        while not self.jobs_queue.empty():
            current_job = self.jobs_queue.get()
            episode_counter = current_job.get('episode_counter')
            run_episodes = current_job.get('run_episodes')
            self.env = get_random_train_env()

            max_episodes = episode_counter + run_episodes
            # print(f"max_episodes: {max_episodes}, episode_counter: {episode_counter}, run_episodes: {run_episodes}")
            while episode_counter < max_episodes:
                time_step = self.env.reset()
                observations = time_step.observation
                print(f"Worker {self.name} Episode {episode_counter} Starting...")
        
                done = False
                total_reward = 0

                while not done:
                    # Get the action probabilities for action type and number of shares
                    action_type_probs, num_shares_probs = self.agent.policy(observations)

                    # Sample an action tuple (action type, number of shares)
                    try:
                        action_type, num_shares = self.agent.sample_actions(action_type_probs, num_shares_probs)
                    except Exception as e:
                        print(f"Exception: {e}")
                        print(f"action_type_probs: {action_type_probs}")
                        print(f"num_shares_probs: {num_shares_probs}")
                        break

                    # Create action tuple
                    actions = np.array([action_type, num_shares]).reshape(1, 2)
                    
                    time_step = self.env.step(actions)
                    next_observations = time_step.observation
                    reward = time_step.reward
                    done = 1 if time_step.is_last() else 0
                    
                    # Train step
                    with self.lock:
                        self.agent.train_step(observations, actions, reward, next_observations, done)

                    observations = next_observations
                    total_reward += reward

                print(f"[{self.name}] Episode {episode_counter} Total Reward: {total_reward}")

                if episode_counter % 25 == 0:
                    avg_return = compute_avg_return_a3c(eval_env, self.agent)
                    print(f"[{self.name}] Episode {episode_counter} Average Return: {avg_return}")
                    self.stats_buffer.append({ 'episode': episode_counter, 'avg_return': avg_return})
                
                episode_counter += 1