import time
from multiprocessing import Process

import tensorflow as tf
import numpy as np


class ANNController:
    def __init__(self, cart_to_control, cart_to_follow, target_distance):
        self._cart_control = cart_to_control
        self._cart_follow = cart_to_follow
        self._dist_tgt = target_distance

        self._in = np.zeros((1, 3), np.float32)
        self._graph = tf.Variable([[60.0], [10.0], [3.0]])

    @staticmethod
    def run(cart_to_control, cart_to_follow, target_distance=0.0):
        p = Process(target=ANNController._run, daemon=True, args=[cart_to_control, cart_to_follow, target_distance])
        p.start()

    @staticmethod
    def _run(cart_to_control, cart_to_follow, target_distance):
        controller = ANNController(cart_to_control, cart_to_follow, target_distance)

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

            d_obs = controller._cart_control.position()[0] - controller._cart_follow.position()[0]

            err = d_obs - controller._dist_tgt
            d_err = (err - err_prev) / dt_sec
            err_accum = decay * err_accum + dt_sec * err
            err_prev = err

            controller._in[0] = [-err, -err_accum, -d_err]
            acc_applied = tf.matmul(controller._in, controller._graph)
            controller._cart_control.acceleration = (acc_applied, 0.0)