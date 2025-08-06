import time

class FPSCounter:
    def __init__(self, buffer_size: int = 15):
        self.time_buffer = [time.time()] * buffer_size
        self.buffer_size = buffer_size

    def get_fps(self) -> float:
        self.time_buffer.append(time.time())
        return (self.buffer_size - 1) / (self.time_buffer[-1] - self.time_buffer.pop(0))