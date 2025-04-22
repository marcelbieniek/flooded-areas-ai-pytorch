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
        self.current_timers[name] = time.time()

    def end(self, name: str):
        end_time = time.time()
        if name not in self.logs:
            raise KeyError(f"[TimeLogger] key '{name}' not found! Start timer with this name before ending it.")
        self.logs[name].append(end_time - self.current_timers[name])
        del self.current_timers[name]

    def print_log(self, name: str):
        if name not in self.logs:
            raise KeyError(f"[TimeLogger] key '{name}' not found!")
        print(self.logs[name])
    
    def print_log_avg(self, name: str):
        if name not in self.logs:
            raise KeyError(f"[TimeLogger] key '{name}' not found!")
        print(f"{sum(self.logs[name])/len(self.logs[name]):.4f} s")


class DataLogger():
    def __init__(self):
        self.logs = {}

    def log(self, name: str, item):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(item)
