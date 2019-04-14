import time
from multiprocessing import Process

import tensorflow as tf
import numpy as np


class ANNController:
    def __init__(self, cart_to_control, cart_to_follow, target_distance):
        self._cart_control = cart_to_control
        self._cart_follow = cart_to_follow
        self._dist_tgt = target_distance

        '''self._tau = 1.0
        self._sample_size = 100
        self._sample_index = 0
        self._accumulated_discounted_future_err = np.zeros(self._sample_size, np.float32)'''

        # Create the Policy Network
        self._uin = np.zeros((1, 3), np.float32)
        layer = tf.keras.layers.Dense(1, None, input_dim=3)
        self._u = tf.keras.Sequential([layer])
        layer.set_weights([
            np.array([[60.0], [10.0], [3.0]], np.float32),
            np.array([0.0], np.float32)])

        #self._vin = np.zeros((1, 7), np.float32)
        #self._vh1 = tf.

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

            controller._uin[0] = [-err, -err_accum, -d_err]
            acc_applied = controller._u(controller._uin)
            controller._cart_control.acceleration = (acc_applied, 0.0)
