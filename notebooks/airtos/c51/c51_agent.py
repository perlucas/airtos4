import tensorflow as tf
from tf_agents.networks import categorical_q_network
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import numpy as np

from utils.envs import train_env_sample, get_random_train_env, eval_env
from utils.training import get_random_policy, compute_avg_return


# Custom CategoricalQNetwork with specific weight initialization
class CustomCategoricalQNetwork(categorical_q_network.CategoricalQNetwork):
    def __init__(self, observation_spec, action_spec, num_atoms, fc_layer_params, name=None):
        super().__init__(observation_spec, action_spec, num_atoms=num_atoms, fc_layer_params=fc_layer_params, name=name)
        # Overriding the layers to apply custom initializer
        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.Constant(0.005)

class C51Agent:

    def __init__(self,
                 layers,
                 num_atoms,
                 optimizer,
                 min_q_value,
                 max_q_value,
                 n_step_update,
                 replay_buffer_capacity = 100000):
        '''Create a new C51Agent instance
        :param layers: tuple: The list of layers to use in the neural network
        :param num_atoms: int: The number of atoms to use in the C51 distribution
        :param optimizer: Optimizer: The optimizer to use for the neural network
        :param min_q_value: float: The minimum Q-value
        :param max_q_value: float: The maximum Q-value
        :param n_step_update: int: The number of steps to update the target network
        '''
        categorical_q_net = CustomCategoricalQNetwork(
            train_env_sample.observation_spec(),
            train_env_sample.action_spec(),
            num_atoms=num_atoms,
            fc_layer_params=layers)
        
        train_step_counter = tf.Variable(initial_value=0, dtype='int64')

        self.agent = categorical_dqn_agent.CategoricalDqnAgent(
            train_env_sample.time_step_spec(),
            train_env_sample.action_spec(),
            categorical_q_network=categorical_q_net,
            optimizer=optimizer,
            min_q_value=min_q_value,
            max_q_value=max_q_value,
            n_step_update=n_step_update,
            # td_errors_loss_fn=common.element_wise_squared_loss, use default error
            gamma=1.0,
            epsilon_greedy=0.1,  # Exploration factor
            train_step_counter=train_step_counter)
        
        self.agent.initialize()

        # Create replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.agent.collect_data_spec,
            batch_size=train_env_sample.batch_size,
            max_length=replay_buffer_capacity)



    def train_agent(self,
                    initial_collect_steps,
                    batch_size,
                    log_dir,
                    num_iterations = 1000,
                    collect_steps_per_iteration = 200,
                    eval_episodes = 2,
                    iterations_per_env = 5,
                    logging_interval = 10,
                    evaluation_interval = 25):
        '''Train the agent
        :param initial_collect_steps: int: The number of initial steps to collect
        :param batch_size: int: The batch size to use for training
        :param log_dir: str: The directory to log the training data
        :param num_iterations: int: The number of iterations to train the agent
        :param collect_steps_per_iteration: int: The number of steps to collect per iteration
        :param eval_episodes: int: The number of episodes to evaluate the agent
        :param iterations_per_env: int: The number of iterations to change the training environment
        :param logging_interval: int: The interval to log training data
        :param evaluation_interval: int: The interval to evaluate the agent
        :return: float: The final average return of the agent
        '''
        # ====================================== Collect Initial Data ======================================
        # Collect initial data, using random policy
        random_policy = get_random_policy()
        self.replay_buffer.clear()
        for _ in range(initial_collect_steps):
            self._collect_step(train_env_sample, random_policy)

        # Dataset generates trajectories with shape [BxTx...] where
        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)
        # ====================================== Training Loop ======================================

        # (Optional) Optimize by wrapping some of the code in a graph using TF function.
        self.agent.train = common.function(self.agent.train)

        # Reset the train step.
        self.agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = compute_avg_return(eval_env, self.agent.policy, eval_episodes)

        train_env = get_random_train_env()

        # Tensorboard logger
        file_writer = tf.summary.create_file_writer(f'{log_dir}/metrics')
        file_writer.set_as_default()

        # Store last N average returns
        last_avg_returns = []
        last_avg_returns_size = 3

        for _ in range(num_iterations):
            # Collect a few steps using collect_policy and save to the replay buffer.
            for _ in range(collect_steps_per_iteration):
                self._collect_step(train_env, self.agent.collect_policy)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience)

            step = self.agent.train_step_counter.numpy()

            # Log information
            if step % logging_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss.loss))
                #tf.summary.scalar('loss2', data=train_loss.loss, step=int64(step))


            # Evaluate agent
            if step % evaluation_interval == 0:
                avg_return = compute_avg_return(eval_env, self.agent.policy, eval_episodes)
                print('step = {0}: Average Return = {1:.2f}'.format(step, avg_return))
                tf.summary.scalar('average return', data=avg_return, step=step)

                # Store last N average returns
                last_avg_returns.append(avg_return)
                if len(last_avg_returns) > last_avg_returns_size:
                    last_avg_returns.pop(0)


            # Change training env
            if step % iterations_per_env == 0:
                train_env = get_random_train_env()

        print("Finished execution!")

        avg_return = compute_avg_return(eval_env, self.agent.policy, eval_episodes)

        # Free up memory
        del dataset
        del iterator
        self.replay_buffer.clear()

        return {
            'custom_return': np.average(last_avg_returns),
            'final_avg_return': avg_return
        }
        

    def _collect_step(self, environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # Add trajectory to the replay buffer
        self.replay_buffer.add_batch(traj)