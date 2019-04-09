from multiprocessing import Process
import time


class Backend:
    def __init__(self, iupdatable):
        self._updatable = iupdatable
        self._process = Process(target=self._run, daemon=True)

    def run(self):
        # Technically, we are required to ensure this is only called only once
        self._process.start()

    def end(self):
        self._process.join()

    def _run(self):
        t_prev = time.time()
        while True:
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now

            self._updatable.update(dt)
