import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np


# Actor Network
class Actor(tf.keras.Model):
    def __init__(
            self,
            observation_spec,
            layers=(100,100),
            num_action_types=3,
            num_shares_options=3):
        super(Actor, self).__init__()

        # Fully connected layers
        self.fc = Sequential()
        _first = True
        for units in layers:
            if _first:
                _first = False
                self.fc.add(Dense(units, activation='relu', input_shape=observation_spec.shape, kernel_initializer=tf.keras.initializers.Constant(0.0005)))
            else:
                self.fc.add(Dense(units, activation='relu', kernel_initializer=tf.keras.initializers.Constant(0.0005)))
        
        # Two output layers, one for action type, one for number of shares
        self.action_type_logits = Dense(num_action_types, kernel_initializer=tf.keras.initializers.Constant(0.0005))
        self.num_shares_logits = Dense(num_shares_options, kernel_initializer=tf.keras.initializers.Constant(0.0005))

    def call(self, inputs):
        x = self.fc(inputs)
        # Output logits for both action type and number of shares
        action_type_logits = self.action_type_logits(x)
        num_shares_logits = self.num_shares_logits(x)
        return action_type_logits, num_shares_logits


# Critic Network
class Critic(tf.keras.Model):
    def __init__(self, layers=(100,100)):
        super(Critic, self).__init__()

        # Fully connected layers
        self.fc = Sequential()
        for units in layers:
            self.fc.add(Dense(units, activation='relu', kernel_initializer=tf.keras.initializers.Constant(0.005)))
        self.fc.add(Dense(1, kernel_initializer=tf.keras.initializers.Constant(0.005)))

    def call(self, inputs):
        return self.fc(inputs)
    
# A3C Agent
class A3CAgent:
    def __init__(
            self,
            action_spec,
            observation_spec,
            actor_layers,
            critic_layers,
            lr=1e-4):
        self.actor = Actor(observation_spec, layers=actor_layers)
        self.critic = Critic(layers=critic_layers)

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
        entropy_loss = -0.02 * entropy  # Encourage exploration
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