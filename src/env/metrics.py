import math
from multiprocessing.sharedctypes import RawArray

from src.common.iupdatable import IUpdatable


class Metrics(IUpdatable):

    def __init__(self, leadcart, followcart, target_dist, mean_time_sec):
        self._leadcart = leadcart
        self._followcart = followcart
        self._dist_target = target_dist
        self._abs_err_accum = 0.0
        self._mae = RawArray('d', 1)
        self._mean_time_sec = mean_time_sec
        self._decay_per_sec = 1.0 - 1.0 / mean_time_sec
        self._t_tot_sec = 0.0
        self._mean_time_reached = False

    def update(self, dt_sec):
        self._t_tot_sec += dt_sec
        if self._mean_time_sec <= self._t_tot_sec:
            self._mean_time_reached = True

        decay = self._decay_per_sec ** dt_sec if self._mean_time_reached else 1.0
        err_abs = math.fabs(self._followcart.position()[0] - self._leadcart.position()[0] - self._dist_target)
        self._abs_err_accum = self._abs_err_accum * decay + dt_sec * err_abs
        self._mae[0] = self._abs_err_accum / self._mean_time_sec if self._mean_time_reached \
            else self._abs_err_accum / (self._t_tot_sec + 0.001)

    @property
    def mae(self):
        return self._mae
