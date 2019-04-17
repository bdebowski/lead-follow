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
        self._tau = 1.0
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
        self._vin = np.zeros((1, 11), np.float32)
        self._v = tf.keras.Sequential()
        self._v.add(tf.keras.layers.Dense(11, activation="sigmoid", input_dim=11))
        self._v.add(tf.keras.layers.Dense(7, activation="sigmoid"))
        self._v.add(tf.keras.layers.Dense(3, activation="sigmoid"))
        self._v.add(tf.keras.layers.Dense(1, activation=None))

        self._base_lr = 0.0001
        self._optimizer = tf.optimizers.SGD(self._base_lr)

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

    def _set_vin(self):
        self._vin[0] = [
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

        t = 0.0
        tick = 0
        loss_rsum = RollingSum(100)
        while True:
            tick += 1
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

            controller._set_vin()

            x_batch = np.array([controller._vin[0], controller._vin[0]])
            x = tf.constant(x_batch, shape=(2, 11))
            v_actual_past = tf.constant([controller._get_future_value_at_oldest_hist(), controller._get_future_value_at_oldest_hist()])
            t += dt_sec
            if t > 5.0:
                with tf.GradientTape() as tape:
                    tape.watch(x)
                    v_pred_past = controller._v(x)
                    loss = tf.losses.mean_squared_error(v_actual_past, v_pred_past)
                gradients = tape.gradient(loss, controller._v.trainable_variables)
                controller._optimizer.apply_gradients(zip(gradients, controller._v.trainable_variables))
                loss_rsum.insert_next(float(tf.reduce_mean(loss)))

                if tick % 100 == 0:
                    controller._optimizer.lr = controller._base_lr * loss_rsum.mean
                    print("Avg Loss: {:.3f}, V_pred={:.2f}, V_true={:.2f}".format(
                        loss_rsum.mean,
                        float(tf.reduce_mean(v_pred_past)),
                        float(tf.reduce_mean(v_actual_past))))

            if tick % 216000 == 0:
                tf.saved_model.save(
                    controller._v,
                    r"D:\Users\Bazyli\Dropbox\PycharmProjects\lead-follow\model\v-{}".format(tick))

            #print("V(t)={:.3f}".format(float(controller._v(controller._vin))))
            #print("V(t)={:.3f}".format(controller.future_value_at_oldest_hist()))


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
