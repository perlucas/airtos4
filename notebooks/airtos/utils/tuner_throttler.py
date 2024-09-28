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
        return self._current_status() == _STATUS_RUNNING
    
    def set_running(self):
        self._status['status'] = _STATUS_RUNNING

    def set_finished(self):
        self._status['status'] = _STATUS_FINISHED
        self._status['executions_count'] += 1

    def executions_count(self):
        return self._status['executions_count']
    
    def executed_trials(self):
        return self.executions_count() * self._trials_per_execution