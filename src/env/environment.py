from multiprocessing import Process
import time

from src.env.cart import Cart
from src.gfx.rectgfx import RectGfx
from src.controller.override_controller import OverrideController
from src.env.metrics import Metrics
from src.gfx.metricsgfx import MetricsGfx


LEAD_PROGRAM_SIMPLE = (350.0, 350.0, 1.0, 3.0, 3.0, 1.0)
LEAD_PROGRAM_COMPLEX = (350.0, 250.0, 16.0, 3.0, 2.0, 8.0)
LEAD_PROGRAM_RANDOMISH = (400.0, 200.0, 3.428571, 2.5, 1.5, 2.142857)


class Environment:
    def __init__(self, width, height, cartlength, cartwidth):
        self._updatables = []

        self._width = width
        self._height = height

        self._lead_centre = (width / 2, height / 2)
        self._leadcart = Cart(self._lead_centre, cartlength, cartwidth)
        self._updatables.append(self._leadcart)

        self._follow_centre = (width / 2, height / 2)
        self._followcart = Cart(self._follow_centre, cartlength, cartwidth)
        self._updatables.append(self._followcart)

        self._override_controller = OverrideController(
            self._leadcart,
            self._lead_centre,
            *LEAD_PROGRAM_SIMPLE)
        self._updatables.append(self._override_controller)

        self._metrics = Metrics(self._leadcart, self._followcart)
        self._updatables.append(self._metrics)

    @property
    def lead_cart(self):
        return self._leadcart

    @property
    def following_cart(self):
        return self._followcart

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def create_gfx_items(self):
        return [
            RectGfx(self._leadcart, self._leadcart.length, self._leadcart.width, (200, 100, 100)),
            RectGfx(self._followcart, self._leadcart.length, self._leadcart.width, (100, 100, 200)),
            MetricsGfx(self._metrics, 10, 790)]

    def run(self):
        p = Process(target=self._run, daemon=True)
        p.start()

    def _run(self):
        t_prev = time.time()
        while True:
            t_now = time.time()
            dt_sec = t_now - t_prev
            t_prev = t_now

            for updatable in self._updatables:
                updatable.update(dt_sec)
