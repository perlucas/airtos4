import os
import json
from datetime import datetime

_STATUS_STARTING = 'Starting'
_STATUS_RUNNING = 'Running'
_STATUS_FINISHED = 'Finished'


class TunnerThrottler:
    '''Class to control the execution of the tuner. It checks if there's a current execution running
    and if it's finished. It also keeps track of the number of executions and trials executed.
    '''
    def __init__(self, status_file_path, trials_per_execution, execution_id):
        self._status_file_path = status_file_path
        self._trials_per_execution = trials_per_execution
        self._execution_id = execution_id
        self._status: dict|None = None

    def load(self):
        if os.path.exists(self._status_file_path):
            with open(self._status_file_path, 'r') as f:
                self._status = json.load(f)
        else:
            self._status = {
                'status': _STATUS_STARTING,
                'executions_count': 0,
                'execution_id': self._execution_id,
                'last_updated': None
            }

    def save(self):
        self._status['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self._status_file_path, 'w') as f:
            json.dump(self._status, f)

    def _current_status(self):
        return self._status['status']

    def is_running(self):
        # is not running if status != Running
        if self._current_status() != _STATUS_RUNNING:
            return False
        last_updated_at = datetime.strptime(self._status['last_updated'], '%Y-%m-%d %H:%M:%S') if self._status['last_updated'] else None

        # If the last update was never set, and status = Running, consider it as running
        if last_updated_at is None:
            return True
        
        # If the last update was more than 5 hours ago, consider it as no longer running
        if (datetime.now() - last_updated_at).total_seconds() > 5 * 60 * 60:
            return False
        return True
    
    def set_running(self):
        self._status['status'] = _STATUS_RUNNING

    def set_finished(self):
        self._status['status'] = _STATUS_FINISHED
        self._status['executions_count'] += 1

    def executions_count(self):
        return self._status['executions_count']
    
    def executed_trials(self):
        return self.executions_count() * self._trials_per_execution