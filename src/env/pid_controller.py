import time
from multiprocessing import Process


class PIDController:
    def __init__(self, cart_to_control, cart_to_follow, target_distance=0.0):
        self._cart_control = cart_to_control
        self._cart_follow = cart_to_follow
        self._dist_tgt = target_distance

    def run(self):
        p = Process(target=self._run, daemon=True)
        p.start()

    def _run(self):
        p = 60.0
        i = 10.0
        d = 3.0

        integral_time_sec = 1.5
        decay_per_sec = 1.0 - 1.0 / integral_time_sec

        err_accum = 0.0
        err_prev = 0.0
        t_prev = time.time()
        while True:
            time.sleep(0.001)
            t_now = time.time()
            dt_sec = t_now - t_prev
            t_prev = t_now

            decay = decay_per_sec ** dt_sec

            d_obs = self._cart_control.position()[0] - self._cart_follow.position()[0]

            err = d_obs - self._dist_tgt
            d_err = (err - err_prev) / dt_sec
            err_accum = decay * err_accum + dt_sec * err
            err_prev = err

            acc_applied = p * -err + i * -err_accum + d * -d_err
            self._cart_control.acceleration = (acc_applied, 0.0)
