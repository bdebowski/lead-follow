class RotatingBuffer:
    def __init__(self, size):
        self._mem = [0.0]*size
        self._sz = size
        self._i = 0

    def set_next(self, val):
        self._i = (self._i + 1) % self._sz
        self._mem[self._i] = val

    def get_prev(self, t_minus=0):
        i = (self._i - t_minus) % self._sz
        return self._mem[i]

    def get_oldest(self):
        i = self._i - self._sz + 1
        return self._mem[i]
