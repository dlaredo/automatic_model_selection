import ray
from collections import defaultdict

@ray.remote
class LoggingActor(object):
    def __init__(self):
        self.logs = defaultdict(lambda: [])
    
    def log(self, index, message):
        self.logs[index].append(message)
    
    def get_logs(self):
        return dict(self.logs)