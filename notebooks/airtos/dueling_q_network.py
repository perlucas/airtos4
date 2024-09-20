import tensorflow as tf
from tf_agents.networks.q_network import QNetwork

class DuelingQNetwork(QNetwork):
    def __init__(self, input_tensor_spec, action_spec, fc_layer_params=(100,), name='DuelingQNetwork'):
        super(DuelingQNetwork, self).__init__(input_tensor_spec, action_spec, fc_layer_params=fc_layer_params, name=name)
        
        # Number of actions
        num_actions = action_spec.maximum - action_spec.minimum + 1
        
        # Create shared layers using fc_layer_params
        self.shared_layers = tf.keras.Sequential()
        for units in fc_layer_params:
            self.shared_layers.add(tf.keras.layers.Dense(units, activation='relu'))
        
        # Create value stream
        self.value_stream = tf.keras.Sequential()
        for units in fc_layer_params:
            self.value_stream.add(tf.keras.layers.Dense(units, activation='relu'))
        self.value_stream.add(tf.keras.layers.Dense(1))  # Outputs a single state value V(s)
        
        # Create advantage stream
        self.advantage_stream = tf.keras.Sequential()
        for units in fc_layer_params:
            self.advantage_stream.add(tf.keras.layers.Dense(units, activation='relu'))
        self.advantage_stream.add(tf.keras.layers.Dense(num_actions))  # Outputs advantage A(s, a) for each action

    def call(self, observation, step_type=None, network_state=(), training=False):
        """Returns Q-values for each action."""
        # Pass the input through the shared layers
        x = self.shared_layers(observation)
        
        # Compute state value and advantages
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage to get Q-values
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=-1, keepdims=True))
        
        return q_values, network_state
