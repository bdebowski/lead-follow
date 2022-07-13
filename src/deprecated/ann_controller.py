import os
import time
import math
from multiprocessing import Process

import tensorflow as tf
import numpy as np

from src.util.rotatingbuffer import RotatingBuffer
from src.util.rollingsum import RollingSum


class ANNController:
    def __init__(self, cart_to_control, cart_to_follow, xspan, yspan, target_distance_cart_lengths):
        self._controlled_cart = cart_to_control
        self._followed_cart = cart_to_follow
        self._controlled_cart_obs_state_gscale = _CartObsState(xspan, yspan)
        self._followed_cart_obs_state_gscale = _CartObsState(xspan, yspan)
        self._relative_cart_obs_state_lscale = _CartObsState(cart_to_control.length, cart_to_control.length)
        self._dist_tgt_lscale = target_distance_cart_lengths
        self._lscale = [cart_to_control.length, cart_to_control.length]
        self._gscale = [xspan, yspan]

        self._hist_sz = 150
        self._xerr_hist_lscale = RotatingBuffer(self._hist_sz)  # normalized by cart length
        self._dt_sec_hist = RotatingBuffer(self._hist_sz)
        self._tau = 0.25
        # These values are in units of environment width/height.
        # top-right corner: xpos = 0.5, ypos = 0.5
        # bottom-left corner: xpos = -0.5, ypos = -0.5
        self._followed_cart_xpos_hist_gscale = RotatingBuffer(self._hist_sz)
        self._followed_cart_xvel_hist_gscale = RotatingBuffer(self._hist_sz)
        self._followed_cart_xacc_hist_gscale = RotatingBuffer(self._hist_sz)
        self._controlled_cart_xpos_hist_gscale = RotatingBuffer(self._hist_sz)
        self._controlled_cart_xvel_hist_gscale = RotatingBuffer(self._hist_sz)
        self._controlled_cart_xacc_hist_gscale = RotatingBuffer(self._hist_sz)
        # These values are in units of cart length.
        # carts are 1 cart length apart on horizontal: xpos = 1.0
        # carts are 1 cart length apart on vertical: ypos = 1.0
        self._relative_cart_xpos_hist_lscale = RotatingBuffer(self._hist_sz)
        self._relative_cart_xvel_hist_lscale = RotatingBuffer(self._hist_sz)
        self._relative_cart_xacc_hist_lscale = RotatingBuffer(self._hist_sz)
        self._xaction_hist_lscale = RotatingBuffer(self._hist_sz)  # acceleration in cart lengths per s^2

        # Create the Policy Network
        self._uin = np.zeros((1, 3), np.float32)
        layer = tf.keras.layers.Dense(1, input_dim=3)
        self._u = tf.keras.Sequential([layer])
        layer.set_weights([
            np.array([[60.0], [10.0], [3.0]], np.float32),
            np.array([0.0], np.float32)])

        # Create the Value Estimation Network
        self._vin = np.zeros(11, np.float32)
        self._vnet = _ValueNetwork(r"D:\Users\Bazyli\Dropbox\PycharmProjects\lead-follow\model\checkpoints")

    def _get_future_value_at_oldest_hist(self):
        """
        Computes the future discounted value of the system from time t0, where t0 = the oldest time
        point in the controller's memory.  The future value is computed for the time span t0 to now.
        This is done looking into the past and calculating all values given perfect vision of how the
        future will play out.
        :return: Future discounted value, which will be negative, since it is a cost of error.
        """
        t_sec = 0.0
        v = 0.0
        for t_minus in range(self._hist_sz - 1, -1, -1):
            dt_sec = self._dt_sec_hist.get_prev(t_minus)
            t_sec += dt_sec
            abs_err = math.fabs(self._xerr_hist_lscale.get_prev(t_minus))
            discount = math.pow(math.e, -t_sec / self._tau)
            v += -abs_err * dt_sec * discount
        return v

    def _update_histories(self, xerr_lscale, dt_sec):
        self._dt_sec_hist.set_next(dt_sec)
        self._xerr_hist_lscale.set_next(xerr_lscale)

        self._followed_cart_obs_state_gscale.update(self._followed_cart.pos, dt_sec)
        self._followed_cart_xpos_hist_gscale.set_next(self._followed_cart_obs_state_gscale.pos[0])
        self._followed_cart_xvel_hist_gscale.set_next(self._followed_cart_obs_state_gscale.vel[0])
        self._followed_cart_xacc_hist_gscale.set_next(self._followed_cart_obs_state_gscale.acc[0])

        self._controlled_cart_obs_state_gscale.update(self._controlled_cart.pos, dt_sec)
        self._controlled_cart_xpos_hist_gscale.set_next(self._controlled_cart_obs_state_gscale.pos[0])
        self._controlled_cart_xvel_hist_gscale.set_next(self._controlled_cart_obs_state_gscale.vel[0])
        self._controlled_cart_xacc_hist_gscale.set_next(self._controlled_cart_obs_state_gscale.acc[0])

        relative_pos_raw = [(self._controlled_cart.pos[i] - self._followed_cart.pos[i]) for i in (0, 1)]
        self._relative_cart_obs_state_lscale.update(relative_pos_raw, dt_sec)
        self._relative_cart_xpos_hist_lscale.set_next(self._relative_cart_obs_state_lscale.pos[0])
        self._relative_cart_xvel_hist_lscale.set_next(self._relative_cart_obs_state_lscale.vel[0])
        self._relative_cart_xacc_hist_lscale.set_next(self._relative_cart_obs_state_lscale.acc[0])

    def _get_vin(self):
        self._vin = [
            self._followed_cart_xpos_hist_gscale.get_prev(self._hist_sz - 1),
            self._followed_cart_xvel_hist_gscale.get_prev(self._hist_sz - 1),
            self._followed_cart_xacc_hist_gscale.get_prev(self._hist_sz - 1),
            self._controlled_cart_xpos_hist_gscale.get_prev(self._hist_sz - 1),
            self._controlled_cart_xvel_hist_gscale.get_prev(self._hist_sz - 1),
            self._controlled_cart_xacc_hist_gscale.get_prev(self._hist_sz - 1),
            self._relative_cart_xpos_hist_lscale.get_prev(self._hist_sz - 1),
            self._relative_cart_xvel_hist_lscale.get_prev(self._hist_sz - 1),
            self._relative_cart_xacc_hist_lscale.get_prev(self._hist_sz - 1),
            self._xaction_hist_lscale.get_prev(self._hist_sz - 1),
            self._xerr_hist_lscale.get_prev(self._hist_sz - 1)]
        return self._vin

    @staticmethod
    def run(cart_to_control, cart_to_follow, xspan, yspan, target_distance_cart_lengths=0.0):
        p = Process(
            target=ANNController._run,
            daemon=True,
            args=[cart_to_control, cart_to_follow, xspan, yspan, target_distance_cart_lengths])
        p.start()

    @staticmethod
    def _run(cart_to_control, cart_to_follow, xspan, yspan, target_distance_cart_lengths):
        controller = ANNController(cart_to_control, cart_to_follow, xspan, yspan, target_distance_cart_lengths)

        integral_time_sec = 1.5
        decay_per_sec = 1.0 - 1.0 / integral_time_sec

        err_accum_lscale = 0.0
        err_prev_lscale = 0.0
        t_prev = time.time()

        while True:
            time.sleep(0.001)
            t_now = time.time()
            dt_sec = t_now - t_prev
            t_prev = t_now

            decay = decay_per_sec ** dt_sec

            d_obs_lscale = (controller._controlled_cart.pos[0] - controller._followed_cart.pos[0]) / controller._lscale[0]

            err_lscale = d_obs_lscale - controller._dist_tgt_lscale
            d_err_lscale = (err_lscale - err_prev_lscale) / dt_sec
            err_accum_lscale = decay * err_accum_lscale + dt_sec * err_lscale
            err_prev_lscale = err_lscale

            controller._update_histories(err_lscale, dt_sec)

            controller._uin[0] = [-err_lscale, -err_accum_lscale, -d_err_lscale]
            acc_applied_lscale = controller._u(controller._uin)
            controller._controlled_cart.set_acc((acc_applied_lscale * controller._lscale[0], 0.0))
            controller._xaction_hist_lscale.set_next(acc_applied_lscale)

            x = controller._get_vin()
            y = controller._get_future_value_at_oldest_hist()
            controller._vnet.next_data_point(x, y)

            #print("V(t) = {:.3f}".format(y))


class _CartObsState:
    def __init__(self, xscale, yscale):
        self.pos = [0.0, 0.0]
        self.vel = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        self._scale = [xscale, yscale]

    def update(self, new_pos_abs, dt_sec):
        new_pos_scaled = [(new_pos_abs[i] / self._scale[i]) for i in (0, 1)]
        new_vel = [((new_pos_scaled[i] - self.pos[i]) / dt_sec) for i in (0, 1)]
        new_acc = [((new_vel[i] - self.vel[i]) / dt_sec) for i in (0, 1)]
        self.pos = new_pos_scaled
        self.vel = new_vel
        self.acc = new_acc


class _ValueNetwork:
    def __init__(self, checkpoint_dir_path):
        self._v = tf.keras.Sequential()
        self._v.add(tf.keras.layers.Dense(11, activation="tanh", input_dim=11))
        self._v.add(tf.keras.layers.Dense(7, activation="tanh"))
        self._v.add(tf.keras.layers.Dense(3, activation="tanh"))
        self._v.add(tf.keras.layers.Dense(1, activation=None))

        self._base_lr = 0.001
        self._lr_decay = 0.9
        self._optimizer = tf.optimizers.SGD(self._base_lr, momentum=0.9)
        self._loss_fn = tf.losses.mean_absolute_error
        self._input_sz = 11
        self._batch_sz = 16

        self._batch_x = np.zeros((self._batch_sz, self._input_sz))
        self._batch_y = np.zeros((self._batch_sz, self._input_sz))
        self._batch_indx = 0

        self._tick = 0
        self._report_period_ticks = 100
        self._checkpoint_period_ticks = 1000
        self._lr_decay_period_ticks = 1000
        self._rolling_loss = RollingSum(self._report_period_ticks)
        self._checkpoint_dir_path = checkpoint_dir_path

    def __call__(self, x):
        return self._v(x)

    def next_data_point(self, x, y):
        self._batch_x[self._batch_indx] = x
        self._batch_y[self._batch_indx] = y
        self._batch_indx = (self._batch_indx + 1) % self._batch_sz
        if self._batch_indx == 0:
            self._backprop()

    def _backprop(self):
        self._tick += 1

        x = tf.constant(self._batch_x, shape=(self._batch_sz, self._input_sz))
        y_true = tf.constant(self._batch_y, shape=(self._batch_sz, self._input_sz))

        with tf.GradientTape() as tape:
            tape.watch(x)
            y_pred = self(x)
            loss = self._loss_fn(y_true, y_pred)

        gradients = tape.gradient(loss, self._v.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self._v.trainable_variables))

        self._rolling_loss.insert_next(float(tf.reduce_mean(loss)))

        if self._tick % self._report_period_ticks == 0:
            self._report()

        if self._tick % self._checkpoint_period_ticks == 0:
            self._save_checkpoint()

        if self._tick % self._lr_decay_period_ticks == 0:
            self._optimizer.lr = self._optimizer.lr * self._lr_decay

    def _report(self):
        print("Avg Loss: {:.3f}".format(
            self._rolling_loss.mean))

    def _save_checkpoint(self):
        checkpoint_name = "v-{}-{:.5f}".format(self._tick, self._rolling_loss.mean)
        save_path = os.path.join(self._checkpoint_dir_path, checkpoint_name)
        tf.saved_model.save(self._v, save_path)
