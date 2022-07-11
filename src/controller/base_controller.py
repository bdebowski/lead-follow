from multiprocessing import Process
import time

from src.common.iupdatable import IUpdatable


class BaseController(IUpdatable):
    def __init__(self):
        self._iupdatables = []

    @property
    def iupdatables(self):
        return self._iupdatables

    def register_iupdatable(self, iupdatable):
        self._iupdatables.append(iupdatable)

    @staticmethod
    def run(controller_factory_method, *args):
        p = Process(target=BaseController._run, daemon=True, args=(controller_factory_method,) + args)
        p.start()

    @staticmethod
    def _run(controller_factory_method, *args):
        controller = controller_factory_method(*args)

        t_prev = time.time()
        while True:
            time.sleep(0.001)
            t_now = time.time()
            dt_sec = t_now - t_prev
            t_prev = t_now

            for updatable in controller.iupdatables:
                updatable.update(dt_sec)
            controller.update(dt_sec)

    def update(self, dt_sec):
        pass
