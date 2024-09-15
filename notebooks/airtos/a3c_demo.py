# Imports 1)
from packaging import version
import tensorflow as tf

# Imports 2)
import os

# Imports 3)
from tensorflow.keras import layers

# Imports 5)
import threading

# Imports 6)
from tf_agents.environments import tf_py_environment
from utils import load_dataset
from envs.combined_env import CombinedEnv
import numpy as np

# ====================================== 1. Check Tensorflow version ======================================
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# ====================================== 2. Enable Keras 3 ======================================
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# ====================================== 3. Define Actor and Critic Networks ======================================

# Actor Network
class Actor(tf.keras.Model):
    def __init__(self, num_action_types, num_shares_options):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        # Two output layers, one for action type, one for number of shares
        self.action_type_logits = layers.Dense(num_action_types)
        self.num_shares_logits = layers.Dense(num_shares_options)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        # Output logits for both action type and number of shares
        action_type_logits = self.action_type_logits(x)
        num_shares_logits = self.num_shares_logits(x)
        return action_type_logits, num_shares_logits


# Critic Network
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.value = layers.Dense(1)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.value(x)

# ====================================== 4. Define A3C Agent ======================================
class A3CAgent:
    def __init__(self, action_spec, observation_spec, lr=1e-4):
        self.actor = Actor(3, 3)
        self.critic = Critic()

        # Optimizers for both actor and critic
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Action and observation specs
        self.action_spec = action_spec
        self.observation_spec = observation_spec

    def policy(self, observation):
        action_type_logits, num_shares_logits = self.actor(observation)
        action_type_probs = tf.nn.softmax(action_type_logits)
        num_shares_probs = tf.nn.softmax(num_shares_logits)
        return action_type_probs, num_shares_probs
    
    def sample_actions(self, action_type_probs, num_shares_probs):
        # Convert action_type_probs from type Tensor (1,3) to numpy array (3,)
        action_type = np.random.choice(3, p=action_type_probs.numpy().reshape(-1))
        num_shares = np.random.choice(3, p=num_shares_probs.numpy().reshape(-1))
        return action_type, num_shares

    def value(self, observation):
        return self.critic(observation)

    def compute_loss(self, rewards, values, log_probs, entropy, gamma=0.99):
        # Calculate discounted rewards
        discounted_rewards = []
        running_add = 0
        for reward in reversed(rewards):
            running_add = reward + gamma * running_add
            discounted_rewards.insert(0, running_add)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)

        # Compute critic loss
        advantages = discounted_rewards - values
        critic_loss = advantages ** 2

        # Compute actor loss
        actor_loss = -log_probs * tf.stop_gradient(advantages)
        entropy_loss = -0.01 * entropy  # Encourage exploration
        total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

        return tf.reduce_mean(total_loss)

    def train_step(self, observations, actions, rewards, next_observations, dones):
        with tf.GradientTape() as tape:
            values = self.value(observations)

            action_type_logits, num_shares_logits = self.actor(observations)
            action_type_probs = tf.nn.softmax(action_type_logits)
            num_shares_probs = tf.nn.softmax(num_shares_logits)

            action_type_log_probs = tf.math.log(action_type_probs)
            num_shares_log_probs = tf.math.log(num_shares_probs)
            entropy = -tf.reduce_sum(action_type_probs * action_type_log_probs, axis=1) - \
                      tf.reduce_sum(num_shares_probs * num_shares_log_probs, axis=1)
    
            # Sampled actions log_probs
            action_type_log_prob = tf.gather_nd(action_type_log_probs, tf.stack([tf.range(len(actions)), actions[:, 0]], axis=1))
            num_shares_log_prob = tf.gather_nd(num_shares_log_probs, tf.stack([tf.range(len(actions)), actions[:, 1]], axis=1))
    
            next_values = self.value(next_observations)

            # Compute targets (rewards + discounted future value)
            targets = rewards + (1 - dones) * 0.99 * next_values
            
            # Calculate advantages (difference between target and value)
            advantages = targets - values

            # Critic loss (Mean Squared Error between predicted values and targets)
            critic_loss = tf.reduce_mean(tf.square(targets - values))

            # Actor loss (policy gradient loss, using advantages)
            actor_loss = -tf.reduce_mean((action_type_log_prob + num_shares_log_prob) * tf.stop_gradient(advantages))

            # Total loss (actor loss + critic loss - entropy bonus for exploration)
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    
        # Backpropagation
        gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients[:len(self.actor.trainable_variables)], self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(gradients[len(self.actor.trainable_variables):], self.critic.trainable_variables))
        
        return total_loss

# ====================================== 5. Define Worker ======================================
MAX_EPISODES = 1000

class Worker(threading.Thread):
    def __init__(self, env, agent, global_episode_counter):
        threading.Thread.__init__(self)
        self.env = env
        self.agent = agent
        self.global_episode_counter = global_episode_counter

    def run(self):
        while self.global_episode_counter < MAX_EPISODES:
            time_step = self.env.reset()
            observations = time_step.observation
            print(f"Worker {self.name} Episode {self.global_episode_counter} Starting...")
    
            done = False
            total_reward = 0

            while not done:
                # Get the action probabilities for action type and number of shares
                action_type_probs, num_shares_probs = self.agent.policy(observations)

                # Sample an action tuple (action type, number of shares)
                action_type, num_shares = self.agent.sample_actions(action_type_probs, num_shares_probs)

                # Create action tuple
                actions = np.array([action_type, num_shares]).reshape(1, 2)
                
                time_step = self.env.step(actions)
                next_observations = time_step.observation
                reward = time_step.reward
                done = 1 if time_step.is_last() else 0
                
                # Train step
                self.agent.train_step(observations, actions, reward, next_observations, done)

                observations = next_observations
                total_reward += reward

            self.global_episode_counter += 1

            print(f"Episode {self.global_episode_counter} Total Reward: {total_reward}")

            if self.global_episode_counter % 10 == 0:
                self.env = get_random_train_env()

            if self.global_episode_counter % 12 == 0:
                avg_return = compute_avg_return(eval_env, self.agent)
                print(f"Episode {self.global_episode_counter} Average Return: {avg_return}")

# ====================================== 6. Define Environment Utils ======================================

## Helper functions
PARAM_ENV_TYPE = 'com'

def create_env(env_type: str, df, window_size, frame_bound):
    if env_type == 'com':
        return CombinedEnv(df=df, window_size=window_size, frame_bound=frame_bound)

    raise NotImplementedError('unknown type')


def create_training_envs(env_type: str):
    ko_df = load_dataset('./resources/KO.csv')
    amzn_df = load_dataset('./resources/AMZN.csv')
    amd_df = load_dataset('./resources/AMD.csv')
    pypl_df = load_dataset('./resources/PYPL.csv')
    nflx_df = load_dataset('./resources/NFLX.csv')
    window_size = 10

    return [
        # KO training envs
        create_env(env_type, ko_df, window_size, (10, 120)),
        create_env(env_type, ko_df, window_size, (120, 230)),
        create_env(env_type, ko_df, window_size, (350, 470)),
        create_env(env_type, ko_df, window_size, (1000, 1120)),
        create_env(env_type, ko_df, window_size, (1700, 1820)),

        # AMZN training envs
        create_env(env_type, amzn_df, window_size, (10, 120)),
        create_env(env_type, amzn_df, window_size, (120, 230)),
        create_env(env_type, amzn_df, window_size, (350, 470)),
        create_env(env_type, amzn_df, window_size, (1000, 1120)),
        create_env(env_type, amzn_df, window_size, (1700, 1820)),

        # AMD training envs
        create_env(env_type, amd_df, window_size, (10, 120)),
        create_env(env_type, amd_df, window_size, (120, 230)),
        create_env(env_type, amd_df, window_size, (350, 470)),
        create_env(env_type, amd_df, window_size, (1000, 1120)),
        create_env(env_type, amd_df, window_size, (1700, 1820)),

        # PYPL training envs
        create_env(env_type, pypl_df, window_size, (10, 120)),
        create_env(env_type, pypl_df, window_size, (120, 230)),
        create_env(env_type, pypl_df, window_size, (350, 470)),
        create_env(env_type, pypl_df, window_size, (1000, 1120)),
        create_env(env_type, pypl_df, window_size, (1700, 1820)),

        # NFLX training envs
        create_env(env_type, nflx_df, window_size, (10, 120)),
        create_env(env_type, nflx_df, window_size, (120, 230)),
        create_env(env_type, nflx_df, window_size, (350, 470)),
        create_env(env_type, nflx_df, window_size, (1000, 1120)),
        create_env(env_type, nflx_df, window_size, (1700, 1820)),
    ]


def create_testing_env(env_type: str):
    ko_df = load_dataset('./resources/KO.csv')
    window_size = 10
    return create_env(env_type, ko_df, window_size, (2000, 2300))



# ====================================== Create environments ======================================

train_py_envs = create_training_envs(PARAM_ENV_TYPE)
train_env_sample = tf_py_environment.TFPyEnvironment(train_py_envs[0])

eval_py_env = create_testing_env(PARAM_ENV_TYPE)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

def get_random_train_env():
    idx = np.random.randint(len(train_py_envs))
    return tf_py_environment.TFPyEnvironment(train_py_envs[idx])

def compute_avg_return(environment, agent, num_episodes=2):
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

# ====================================== 7. Train A3C Agent ======================================
# Start workers
num_workers = 1
global_episode_counter = 0
agent = A3CAgent(train_env_sample.action_spec(), train_env_sample.observation_spec())
workers = [Worker(train_env_sample, agent, global_episode_counter) for _ in range(num_workers)]

for worker in workers:
    worker.start()

for worker in workers:
    worker.join()