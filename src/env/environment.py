from src.common.iupdatable import IUpdatable
from src.env.cart import Cart
from src.gfx.rectgfx import RectGfx
from src.env.override_controller import OverrideController
from src.env.pid_controller import PIDController


class Environment(IUpdatable):
    def __init__(self):
        self._updatables = []

        self._lead_centre = (600.0, 400.0)
        self._leadcart = Cart(self._lead_centre)
        self._updatables.append(self._leadcart)

        self._follow_centre = self._lead_centre
        self._followcart = Cart(self._follow_centre)
        self._updatables.append(self._followcart)

        self._override_controller = OverrideController(self._leadcart, self._lead_centre, 150.0, 5.0)
        self._updatables.append(self._override_controller)

        self._pid_controller = PIDController(self._followcart, self._leadcart, 0.0)
        self._pid_controller.run()

    def update(self, dt_sec):
        # update self
        # update each updatable
        for updatable in self._updatables:
            updatable.update(dt_sec)

    def create_gfx_items(self):
        return [
            RectGfx(self._leadcart, 10, 5, (200, 100, 100)),
            RectGfx(self._followcart, 10, 5, (100, 100, 200))]
