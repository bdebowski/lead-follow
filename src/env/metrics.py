import math
from multiprocessing.sharedctypes import RawArray

from src.common.iupdatable import IUpdatable
from src.util.rollingsum import RollingSum


class Metrics(IUpdatable):
    _10K = 10000
    _100K = 100000
    _1M = 1000000

    def __init__(self, leadcart, followcart, target_dist):
        self._leadcart = leadcart
        self._followcart = followcart
        self._dist_target = target_dist

        self._abserr_sum_10k = RollingSum(self._10K)
        self._abserr_sum_100k = RollingSum(self._100K)
        self._abserr_sum_1m = RollingSum(self._1M)

        self._mae = RawArray('d', 3)

    def update(self, _):
        err_abs = math.fabs(self._followcart.pos[0] - self._leadcart.pos[0] - self._dist_target)

        self._abserr_sum_10k.insert_next(err_abs)
        self._abserr_sum_100k.insert_next(err_abs)
        self._abserr_sum_1m.insert_next(err_abs)

        self._mae[0] = self._abserr_sum_10k.mean
        self._mae[1] = self._abserr_sum_100k.mean
        self._mae[2] = self._abserr_sum_1m.mean

    @property
    def mae(self):
        return self._mae
