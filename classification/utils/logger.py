import time

class TimeLogger():
    def __init__(self):
        self.logs = {}
        self.current_timers = {}
    
    def start(self, name: str):
        """
        Args:
            name (string): Name of timer to start.
        Start timer with provided name. If timer of that name has already been started, it will be reset and the measurement will begin from last start() call.
        """
        if name not in self.logs:
            self.logs[name] = []
        self.current_timers[name].append(time.time())

    def end(self, name: str):
        end_time = time.time()
        if name not in self.logs:
            raise KeyError(f"[TimeLogger] key '{name}' not found! Start timing before ending.")
        self.logs[name].append(end_time)
