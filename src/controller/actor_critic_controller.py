from pathlib import Path

import torch
from torch.optim.adamw import AdamW

from src.controller.base_controller import BaseController
from src.common.iupdatable import IUpdatable
from src.env.cart import Cart
from src.util.rollingsum import RollingSum


class PIDBootstrapper:
    def __init__(self, cart_to_control, cart_to_follow, policy_network_parameters, normalization_fn, p=60.0, i=10.0, d=3.0):
        self._cart_control = cart_to_control
        self._cart_follow = cart_to_follow

        self._p = p
        self._i = i
        self._d = d

        integral_time_sec = 1.5
        self._decay_per_sec = 1.0 - 1.0 / integral_time_sec

        self._err_accum = 0.0
        self._err_prev = 0.0

        self._optimizer = AdamW(policy_network_parameters, 0.001)
        self._loss_fn = torch.nn.MSELoss()
        self._loss_rolling_sum = RollingSum(10000)
        self._normalization_fn = normalization_fn

    def update_policy(self, dt_sec: float, y_policy: torch.Tensor):
        decay = self._decay_per_sec ** dt_sec

        err = self._cart_control.pos[0] - self._cart_follow.pos[0]
        d_err = (err - self._err_prev) / dt_sec
        self._err_accum = decay * self._err_accum + dt_sec * err
        self._err_prev = err

        y_pid = self._normalization_fn(self._p * -err + self._i * -self._err_accum + self._d * -d_err)

        self._optimizer.zero_grad()

        loss = self._loss_fn(y_policy, torch.tensor(y_pid))
        loss.backward()
        self._optimizer.step()

        self._loss_rolling_sum.insert_next(loss.item())

        return self._loss_rolling_sum.mean, y_pid


class PolicyNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super().__init__()
        self._layer_hidden = torch.nn.Linear(num_inputs, num_hidden, bias=False)
        self._layer_out = torch.nn.Linear(num_hidden, 1, bias=False)

    def forward(self, x):
        return self._layer_out(torch.tanh_(self._layer_hidden(x)))

    @staticmethod
    def load(path):
        state_dict = torch.load(path)
        policy_network = PolicyNetwork(state_dict["_layer_hidden.weight"].shape[1], state_dict["_layer_hidden.weight"].shape[0])
        policy_network.load_state_dict(state_dict)
        return policy_network

    def save(self, path):
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.state_dict(), path)


class ActorCriticController(BaseController):
    def __init__(
            self,
            cart_to_control: Cart,
            cart_to_follow: Cart,
            x_span: int,
            bootstrap_policy=True,
            bootstrap_loss_threshold=5.0,
            policy_network_save_file_path=None):
        super().__init__()

        self._lead_cart = cart_to_follow
        self._controlling_cart = cart_to_control

        self._lead_cart_observer = CartObserver(self._lead_cart, x_span)
        self._controlling_cart_observer = CartObserver(self._controlling_cart, x_span)
        self.register_iupdatable(self._lead_cart_observer)
        self.register_iupdatable(self._controlling_cart_observer)

        self._lead_follow_cart_difference_observer = DifferenceObserver(self._lead_cart_observer, self._controlling_cart_observer)
        self.register_iupdatable(self._lead_follow_cart_difference_observer)

        self._normalize_output_fn = lambda x: x * 2 / x_span
        self._denormalize_output_fn = lambda x: x * x_span / 2

        self._policy_network_save_file_path = Path(policy_network_save_file_path) if policy_network_save_file_path else ""
        if self._policy_network_save_file_path and self._policy_network_save_file_path.exists():
            self._policy_network = PolicyNetwork.load(self._policy_network_save_file_path)
            print("Policy network loaded from {}".format(self._policy_network_save_file_path))
        else:
            self._policy_network = PolicyNetwork(9, 5)

        if bootstrap_policy:
            self._policy_bootstrapper = PIDBootstrapper(cart_to_control, cart_to_follow, self._policy_network.parameters(), self._normalize_output_fn)
        else:
            self._policy_bootstrapper = None
        self._bootstrap_loss_threshold = bootstrap_loss_threshold

    def update(self, dt_sec):
        x = torch.tensor([
            self._lead_cart_observer.pos,
            self._lead_cart_observer.vel,
            self._lead_cart_observer.acc,
            self._controlling_cart_observer.pos,
            self._controlling_cart_observer.vel,
            self._controlling_cart_observer.acc,
            self._lead_follow_cart_difference_observer.d_pos,
            self._lead_follow_cart_difference_observer.d_vel,
            self._lead_follow_cart_difference_observer.d_acc])
        y_policy = self._policy_network(x)

        if self._policy_bootstrapper:
            mean_loss, y_norm = self._policy_bootstrapper.update_policy(dt_sec, y_policy)
            print(mean_loss)
            if mean_loss < self._bootstrap_loss_threshold:
                self._policy_bootstrapper = None
                print("Policy Network bootstrapped")
                if self._policy_network_save_file_path:
                    self._policy_network.save(self._policy_network_save_file_path)
        else:
            y_norm = y_policy.item()

        self._controlling_cart.set_acc((self._denormalize_output_fn(y_norm), 0.0))


class CartObserver(IUpdatable):
    def __init__(self, cart_to_observe: Cart, x_span: int):
        self._cart = cart_to_observe
        self._x_span = x_span

        # The position is normalized such that far left side of screen = -1.0 and far right of screen = 1.0
        # Velocity and acceleration values are computed in terms of these normalized position values
        self._pos = 0.0
        self._vel = 0.0
        self._acc = 0.0

    @property
    def pos(self):
        return self._pos

    @property
    def vel(self):
        return self._vel

    @property
    def acc(self):
        return self._acc

    def update(self, dt_sec):
        new_pos = self._normalize_pos(self._cart.pos[0])
        new_vel = (new_pos - self._pos) / dt_sec
        self._acc = (new_vel - self._vel) / dt_sec
        self._vel = new_vel
        self._pos = new_pos

    def _normalize_pos(self, pos_abs):
        half_span = self._x_span / 2
        return (pos_abs - half_span) / half_span


class DifferenceObserver(IUpdatable):
    def __init__(self, cart_observer_lead_cart: CartObserver, cart_observer_following_cart: CartObserver):
        self._lead_cart_observer = cart_observer_lead_cart
        self._following_cart_observer = cart_observer_following_cart
        self._d_pos = 0.0
        self._d_vel = 0.0
        self._d_acc = 0.0

    @property
    def d_pos(self):
        return self._d_pos

    @property
    def d_vel(self):
        return self._d_vel

    @property
    def d_acc(self):
        return self._d_acc

    def update(self, dt_sec):
        self._d_pos = self._lead_cart_observer.pos - self._following_cart_observer.pos
        self._d_vel = self._lead_cart_observer.vel - self._following_cart_observer.vel
        self._d_acc = self._lead_cart_observer.acc - self._following_cart_observer.acc
