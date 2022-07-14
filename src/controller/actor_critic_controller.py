from pathlib import Path

import torch
from torch.optim.adamw import AdamW

from src.controller.base_controller import BaseController
from src.common.iupdatable import IUpdatable
from src.env.cart import Cart
from src.util.rollingsum import RollingSum
from src.util.rotatingbuffer import RotatingBuffer

# todo:
#   Works but looks like i) system converges ii) then value network gets dumb because it's only seeing easy examples and it overfits to them iii) then
#   policy network diverges on feedback from dumb value network iv) then value network gets better again after seeing more difficult examples
#   - add randomness (i.e. noise) to policy (at input? at output?) so that value network learns smoother and broader function
#   - change value network input to include policy network output (and env state?) from previous time steps as well (t, t-1, t-2)


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

        self._optimizer = AdamW(policy_network_parameters, lr=0.001)
        self._loss_fn = torch.nn.MSELoss()
        self._loss_rolling_sum = RollingSum(1000)
        self._normalization_fn = normalization_fn

    def update_policy(self, dt_sec: float, y_policy: torch.Tensor):
        decay = self._decay_per_sec ** dt_sec

        err = self._cart_control.pos[0] - self._cart_follow.pos[0]
        d_err = (err - self._err_prev) / dt_sec
        self._err_accum = decay * self._err_accum + dt_sec * err
        self._err_prev = err

        y_pid = self._normalization_fn(self._p * -err + self._i * -self._err_accum + self._d * -d_err)

        self._optimizer.zero_grad()

        loss = self._loss_fn(y_policy, torch.tensor([y_pid]))
        loss.backward()
        self._optimizer.step()

        self._loss_rolling_sum.insert_next(loss.item())

        return self._loss_rolling_sum.mean, y_pid


class PolicyOrValueNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden):
        super().__init__()
        self._layer_hidden = torch.nn.Linear(num_inputs, num_hidden, bias=False)
        self._layer_out = torch.nn.Linear(num_hidden, 1, bias=False)

    def forward(self, x):
        return self._layer_out(torch.tanh_(self._layer_hidden(x)))

    @staticmethod
    def load(path):
        state_dict = torch.load(path)
        policy_or_value_network = PolicyOrValueNetwork(state_dict["_layer_hidden.weight"].shape[1], state_dict["_layer_hidden.weight"].shape[0])
        policy_or_value_network.load_state_dict(state_dict)
        return policy_or_value_network

    def save(self, path):
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        torch.save(self.state_dict(), path)


class CriticTrainer:
    def __init__(self, value_network, err_window_size=25, gamma=0.95, save_path=None, save_threshold_dloss=0.1):
        self._value_network = value_network
        self._optimizer = AdamW(value_network.parameters(), lr=0.001)
        self._loss_fn = torch.nn.MSELoss()
        self._loss_rolling_sum = RollingSum(1000)
        self._mean_loss_best = 9999999.0

        self._err_window_size = err_window_size
        self._gamma = gamma
        self._discounted_error_history = RotatingBuffer(err_window_size)

        self._num_x_history_insertions = 0
        self._x_history = RotatingBuffer(err_window_size)

        self._save_path = save_path
        self._save_threshold_dloss = save_threshold_dloss

    def update_critic(self, x: torch.Tensor, current_error: float):
        """
        We are passing in the current state x and the current error.
        We want the value network to predict the current discounted error G_t given the previous state x_t-n; where n = error window size
        """
        # The last item in the discounted_error_history is the sum of an infinite length series e_t + gamma * e_t-1 + gamma^2 * e_t-2 + ...
        # The second to last item in the discounted_error_history stores what the value of that sum was at the previous time step
        current_discounted_error = self._gamma * self._discounted_error_history.get_prev() + current_error
        self._discounted_error_history.set_next(current_discounted_error)

        if self._num_x_history_insertions < self._err_window_size:
            self._x_history.set_next(x)
            self._num_x_history_insertions += 1
            return self._mean_loss_best

        # We want to get the sum of the discounted error series of finite length running from t=0 to t=1-window_length
        # We can obtain this sum by taking the current infinite length series sum and subtracting the value of the previous sum at t+1-window_length
        # discounted by a factor of gamma^window_length
        prev_discounted_error = self._discounted_error_history.get_prev(t_minus=self._err_window_size)
        discounted_error_over_window = current_discounted_error - self._gamma ** self._err_window_size * prev_discounted_error

        self._optimizer.zero_grad()
        y_critic = self._value_network(self._x_history.get_oldest())
        loss = self._loss_fn(y_critic, torch.tensor([discounted_error_over_window / self._err_window_size]))
        loss.backward()
        self._optimizer.step()

        # We wait until now to insert the current state because all the above math is computed with respect to the previous state(s)
        self._x_history.set_next(x)
        self._num_x_history_insertions += 1

        self._loss_rolling_sum.insert_next(loss.item())
        mean_loss = self._loss_rolling_sum.mean

        if self._save_threshold_dloss < (self._mean_loss_best - mean_loss) / self._mean_loss_best and self._save_path:
            self._mean_loss_best = mean_loss
            self._value_network.save(self._save_path)

        return mean_loss


class PolicyTrainer:
    def __init__(self, policy_network, value_network, save_path=None, save_threshold_dloss=0.1):
        self._policy_network = policy_network
        self._value_network = value_network
        self._optimizer = AdamW(policy_network.parameters(), lr=0.00025)
        self._loss_fn = torch.nn.MSELoss()
        self._loss_rolling_sum = RollingSum(1000)
        self._mean_loss_best = 9999999.0

        self._save_path = save_path
        self._save_threshold_dloss = save_threshold_dloss

    def update_policy(self, x: torch.Tensor, y_policy):
        self._optimizer.zero_grad()
        y_critic = self._value_network(torch.cat([x, y_policy]))
        loss = self._loss_fn(y_critic, torch.tensor([0.0]))
        loss.backward()
        self._optimizer.step()

        self._loss_rolling_sum.insert_next(loss.item())
        mean_loss = self._loss_rolling_sum.mean

        if self._save_threshold_dloss < (self._mean_loss_best - mean_loss) / self._mean_loss_best and self._save_path:
            self._mean_loss_best = mean_loss
            self._policy_network.save(self._save_path)

        return mean_loss


class ActorCriticController(BaseController):
    def __init__(
            self,
            cart_to_control: Cart,
            cart_to_follow: Cart,
            x_span: int,
            bootstrap_policy=True,
            bootstrap_policy_loss_threshold=5.0,
            policy_network_save_file_path=None,
            pretrain_critic=True,
            pretrain_critic_loss_threshold=1.0,
            value_network_save_file_path=None,
            log_frequency_s=1.0):
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
            self._policy_network = PolicyOrValueNetwork.load(self._policy_network_save_file_path)
            print("Policy network loaded from {}".format(self._policy_network_save_file_path))
        else:
            self._policy_network = PolicyOrValueNetwork(9, 5)

        if bootstrap_policy:
            self._policy_bootstrapper = PIDBootstrapper(cart_to_control, cart_to_follow, self._policy_network.parameters(), self._normalize_output_fn)
        else:
            self._policy_bootstrapper = None
        self._bootstrap_loss_threshold = bootstrap_policy_loss_threshold

        self._value_network_save_file_path = Path(value_network_save_file_path) if value_network_save_file_path else ""
        if self._value_network_save_file_path and self._value_network_save_file_path.exists():
            self._value_network = PolicyOrValueNetwork.load(self._value_network_save_file_path)
            print("Value network loaded from {}".format(self._value_network_save_file_path))
        else:
            self._value_network = PolicyOrValueNetwork(10, 5)

        self._critic_trainer = CriticTrainer(self._value_network, save_path=value_network_save_file_path)
        self._pretraining_critic = pretrain_critic
        self._pretrain_critic_loss_threshold = pretrain_critic_loss_threshold

        self._policy_network_trainer = PolicyTrainer(self._policy_network, self._value_network, save_path=policy_network_save_file_path)

        self._log_frequency_s = log_frequency_s
        self._time_s_since_last_log = 0.0

    def update(self, dt_sec):
        log_str = "Policy Bootstrap Loss: {:0.5f}   Value Network Loss: {:0.5f}   Policy Network Loss: {:0.5f}"
        mean_loss_policy_bootstrap, mean_loss_value_network, mean_loss_policy_network = 0.0, 0.0, 0.0
        self._time_s_since_last_log += dt_sec

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
            mean_loss_policy_bootstrap, y_norm = self._policy_bootstrapper.update_policy(dt_sec, y_policy)
            policy_updated = True
            if mean_loss_policy_bootstrap < self._bootstrap_loss_threshold:
                self._policy_bootstrapper = None
                print("Bootstrapping Policy Network finished")
                if self._policy_network_save_file_path:
                    self._policy_network.save(self._policy_network_save_file_path)
        else:
            policy_updated = False
            y_norm = y_policy.item()

        current_error = self._lead_cart_observer.pos - self._controlling_cart_observer.pos

        mean_loss_value_network = self._critic_trainer.update_critic(torch.cat([x, torch.tensor([y_norm])]), current_error)

        if self._pretraining_critic and mean_loss_value_network < self._pretrain_critic_loss_threshold:
            print("Pretraining Value Network finished")
            self._pretraining_critic = False

        if not self._pretraining_critic and not self._policy_bootstrapper and not policy_updated:
            mean_loss_policy_network = self._policy_network_trainer.update_policy(x, y_policy)

        if self._log_frequency_s < self._time_s_since_last_log:
            print(log_str.format(mean_loss_policy_bootstrap, mean_loss_value_network, mean_loss_policy_network))
            self._time_s_since_last_log = 0.0

        # Bound the cart to stay in the visible area by adding acceleration artificially when the bounds are exceeded
        y_norm += (min(self._controlling_cart_observer.pos, -1.0) + 1.0) * -10.0 + (max(self._controlling_cart_observer.pos, 1.0) - 1.0) * -10.0
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
