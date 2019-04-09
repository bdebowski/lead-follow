from multiprocessing.sharedctypes import RawArray

from src.common.iupdatable import IUpdatable
from src.env.iobservable import IObservable


class Cart(IObservable, IUpdatable):
    def __init__(self, start_loc):
        self._pos = RawArray('d', start_loc)
        self._vel = [0, 0]
        self._acc = RawArray('d', 2)

    def update(self, dt_sec):
        self._vel[0] += dt_sec * self._acc[0]
        self._vel[1] += dt_sec * self._acc[1]
        self._pos[0] += dt_sec * self._vel[0]
        self._pos[1] += dt_sec * self._vel[1]

    def position(self):
        return self._pos

    @property
    def acceleration(self):
        return self._acc

    @acceleration.setter
    def acceleration(self, value):
        self._acc[0] = value[0]
        self._acc[1] = value[1]
