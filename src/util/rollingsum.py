from src.util.rotatingbuffer import RotatingBuffer


class RollingSum:
    def __init__(self, size):
        self._sz = size
        self._sum = 0.0
        self._i = 0
        self._resum = 0.0
        self._rollingover = False
        self._buffer = RotatingBuffer(size)

    def insert_next(self, value):
        # Correct for drift due to floating point error by recomputing sum once every roll-over
        if self._rollingover:
            self._sum = self._resum
            self._resum = value
            self._rollingover = False
        else:
            self._resum += value

        # Detect roll-over
        self._i = (self._i + 1) % self._sz
        if self._i == 0:
            self._rollingover = True

        # Update current rolling sum
        last_value = self._buffer.get_prev(self._sz - 1)
        self._buffer.set_next(value)
        self._sum += value - last_value

    def current_sum(self):
        return self._sum
