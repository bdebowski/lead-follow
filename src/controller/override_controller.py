import math

from src.common.iupdatable import IUpdatable


class OverrideController(IUpdatable):
    """Controls a cart by overriding its position"""

    def __init__(self, cart, centre, mag_one, mag_two, mag_shift_per_s, per_one_s, per_two_s, per_shift_per_s):
        self._t_tot_sec = 0.0
        self._t_tot_sec_adj = 0.0
        self._cart = cart
        self._cart_centre = centre
        self._mag_one = mag_one
        self._mag_two = mag_two
        self._mag_shift_per_s = mag_shift_per_s
        self._per_one_s = per_one_s
        self._per_two_s = per_two_s
        self._per_shift_per_s = per_shift_per_s

    def update(self, dt_sec):
        self._t_tot_sec += dt_sec

        per_sel = 0.5 * (math.sin(self._t_tot_sec * 2.0 * math.pi / self._per_shift_per_s) + 1.0)
        time_scaling = (1.0 - per_sel) * 1.0 + per_sel * self._per_one_s / self._per_two_s
        self._t_tot_sec_adj += dt_sec * time_scaling

        mag_sel = 0.5 * (math.sin(self._t_tot_sec * 2.0 * math.pi / self._mag_shift_per_s) + 1.0)
        mag = (1.0 - mag_sel) * self._mag_one + mag_sel * self._mag_two

        self._cart._pos[0] = self._cart_centre[0] + math.sin(self._t_tot_sec_adj * 2.0 * math.pi / self._per_one_s) * mag
