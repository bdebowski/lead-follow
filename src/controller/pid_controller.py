from src.controller.base_controller import BaseController


class PIDControllerFactory:
    @staticmethod
    def create(cart_to_control, cart_to_follow):
        return PIDController(cart_to_control, cart_to_follow)


class PIDController(BaseController):
    def __init__(self, cart_to_control, cart_to_follow, p=60.0, i=10.0, d=3.0):
        super().__init__()

        self._cart_control = cart_to_control
        self._cart_follow = cart_to_follow

        self._p = p
        self._i = i
        self._d = d

        integral_time_sec = 1.5
        self._decay_per_sec = 1.0 - 1.0 / integral_time_sec

        self._err_accum = 0.0
        self._err_prev = 0.0

    def update(self, dt_sec):
        decay = self._decay_per_sec ** dt_sec

        err = self._cart_control.pos[0] - self._cart_follow.pos[0]
        d_err = (err - self._err_prev) / dt_sec
        self._err_accum = decay * self._err_accum + dt_sec * err
        self._err_prev = err

        acc_applied = self._p * -err + self._i * -self._err_accum + self._d * -d_err
        self._cart_control.set_acc((acc_applied, 0.0))
