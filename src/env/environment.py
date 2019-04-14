from src.common.iupdatable import IUpdatable
from src.env.cart import Cart
from src.gfx.rectgfx import RectGfx
from src.controller.override_controller import OverrideController
from src.controller.ann_controller import ANNController
from src.env.metrics import Metrics
from src.gfx.metricsgfx import MetricsGfx


LEAD_PROGRAM_SIMPLE = (350.0, 350.0, 1.0, 3.0, 3.0, 1.0)
LEAD_PROGRAM_COMPLEX = (350.0, 250.0, 16.0, 3.0, 2.0, 8.0)
LEAD_PROGRAM_RANDOMISH = (400.0, 200.0, 3.428571, 2.5, 1.5, 2.142857)


class Environment(IUpdatable):
    def __init__(self):
        self._updatables = []

        self._lead_centre = (600.0, 400.0)
        self._leadcart = Cart(self._lead_centre)
        self._updatables.append(self._leadcart)

        self._follow_centre = (600.0, 400.0)
        self._followcart = Cart(self._follow_centre)
        self._updatables.append(self._followcart)

        self._override_controller = OverrideController(
            self._leadcart,
            self._lead_centre,
            *LEAD_PROGRAM_RANDOMISH)
        self._updatables.append(self._override_controller)

        self._follow_offset = 0.0
        ANNController.run(self._followcart, self._leadcart, self._follow_offset)

        self._metrics = Metrics(self._leadcart, self._followcart, self._follow_offset)
        self._updatables.append(self._metrics)

    def update(self, dt_sec):
        # update self
        # update each updatable
        for updatable in self._updatables:
            updatable.update(dt_sec)

    def create_gfx_items(self):
        return [
            RectGfx(self._leadcart, 10, 5, (200, 100, 100)),
            RectGfx(self._followcart, 10, 5, (100, 100, 200)),
            MetricsGfx(self._metrics, 10, 790)]
