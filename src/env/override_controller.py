import math

from src.common.iupdatable import IUpdatable


class OverrideController(IUpdatable):
    """Controls a cart by overriding its position"""

    def __init__(self, cart, centre, magnitude, period_s):
        self._t_tot_sec = 0.0
        self._cart = cart
        self._cart_centre = centre
        self._magnitude = magnitude
        self._per_s = period_s

    def update(self, dt_sec):
        self._t_tot_sec += dt_sec
        self._cart._pos[0] = self._cart_centre[0] + \
                             math.sin(self._t_tot_sec * 2.0 * math.pi / self._per_s) * self._magnitude
