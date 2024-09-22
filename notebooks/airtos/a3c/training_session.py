from queue import Queue
import threading

import tensorflow as tf

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

        print(stats_collector)

        # Log stats and metrics
        file_writer = tf.summary.create_file_writer(f'{log_dir}/metrics')
        file_writer.set_as_default()

        stats_collector.sort(key=lambda x: x.get('episode'))

        cumulated_deltas = 0
        prev_avg_return = 0
        final_avg_return = 0
        for stats in stats_collector:
            step = stats.get('episode')
            avg_return = stats.get('avg_return')
            final_avg_return = avg_return
            cumulated_deltas += (avg_return - prev_avg_return) / abs(prev_avg_return) if prev_avg_return != 0 else 0
            prev_avg_return = avg_return

            # Log metrics
            tf.summary.scalar('average return', data=avg_return, step=step)
            tf.summary.scalar('cumulated deltas', data=cumulated_deltas, step=step)

        # Close file writer
        file_writer.close()

        print('All workers finished!')
        return {
            'final_avg_return': final_avg_return,
            'cumulated_deltas': cumulated_deltas
        }
