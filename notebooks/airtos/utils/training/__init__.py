from tf_agents.policies import random_tf_policy
import numpy as np

from utils.envs import train_env_sample


# ====================================== Helper for Avg Return ======================================
def compute_avg_return(environment, policy, num_episodes=2):
    '''Compute the average return of the policy in the environment
    :param environment: TFPyEnvironment: The environment to evaluate the policy in
    :param policy: Policy: The policy to evaluate
    :param num_episodes: int: The number of episodes to evaluate the policy
    :return: float: The average return of the policy in the environment
    '''
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def compute_avg_return_a3c(environment, agent, num_episodes=2):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():

            action_type_probs, num_shares_probs = agent.policy(time_step.observation)
            action_type, num_shares = agent.sample_actions(action_type_probs, num_shares_probs)

            # Create action tuple
            actions = np.array([action_type, num_shares]).reshape(1, 2)
            time_step = environment.step(actions)
            episode_return += time_step.reward

        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


# ====================================== Get Random Policy ======================================
def get_random_policy():
    '''Get a random policy
    :return: RandomTFPolicy: The random policy
    '''
    return random_tf_policy.RandomTFPolicy(train_env_sample.time_step_spec(), train_env_sample.action_spec())