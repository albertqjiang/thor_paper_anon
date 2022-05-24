import time
import metric_logging
from jobs.core import Job

class HelloJob(Job):
    def execute(self):
        print(f'Hello Job at {time.time()}')
        metric_logging.log_scalar('time', 0, time.time())