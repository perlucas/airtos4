from queue import Queue
import threading

import tensorflow as tf
import numpy as np

from .a3c_agent import A3CAgent
from .training_worker import TrainingWorker

class TrainingSession:
    def __init__(self, agent: A3CAgent):
        self.agent = agent


    def run(self, num_episodes=1000, episodes_per_job=10, num_workers=4, log_dir='logs'):
        jobs_queue = Queue()
        stats_collector = []

        # Create jobs
        for i in range(0, num_episodes + 1, episodes_per_job):
            job = {
                'episode_counter': i,
                'run_episodes': episodes_per_job
            }
            jobs_queue.put(job)

        # Create workers
        lock = threading.Lock()
        workers = []
        for i in range(num_workers):
            worker = TrainingWorker(self.agent, jobs_queue, stats_collector, lock)
            workers.append(worker)
            worker.start()

        # Wait for all workers to finish
        for worker in workers:
            worker.join()

        # Delete all workers to encourage garbage collection
        del workers

        # Log stats and metrics
        file_writer = tf.summary.create_file_writer(f'{log_dir}/metrics')
        file_writer.set_as_default()

        stats_collector.sort(key=lambda x: x.get('episode'))

        last_n_avg_returns = []
        last_n_avg_returns_len = 4
        final_avg_return = 0
        for stats in stats_collector:
            step = stats.get('episode')
            avg_return = stats.get('avg_return')
            final_avg_return = avg_return

            # Compute last n average returns
            last_n_avg_returns.append(avg_return)
            if len(last_n_avg_returns) > last_n_avg_returns_len:
                last_n_avg_returns.pop(0)

            # Log metrics
            tf.summary.scalar('average return', data=avg_return, step=step)

        # Close file writer
        file_writer.close()

        print('All workers finished!')
        return {
            'final_avg_return': final_avg_return,
            'custom_return': np.average(last_n_avg_returns)
        }
