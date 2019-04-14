from multiprocessing.sharedctypes import RawArray, RawValue

from src.common.iupdatable import IUpdatable
from src.env.iobservable import IObservable


class Cart(IObservable, IUpdatable):
    def __init__(self, start_loc):
        self._pos = RawArray('d', start_loc)
        self._vel = RawArray('d', 2)
        self._acc = RawArray('d', 2)
        self._rot = RawValue('d')

    def update(self, dt_sec):
        self._vel[0] += dt_sec * self._acc[0]
        self._vel[1] += dt_sec * self._acc[1]
        self._pos[0] += dt_sec * self._vel[0]
        self._pos[1] += dt_sec * self._vel[1]

    @property
    def pos(self):
        return self._pos

    @property
    def rot(self):
        return self._rot

    def set_acc(self, value):
        self._acc[0] = value[0]
        self._acc[1] = value[1]
