import pyglet
from pyglet.gl import *


class MainWin(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._fps_display = pyglet.window.FPSDisplay(self)

        self._drawables = []
        self._drawables.append(self._fps_display)

    def draw(self):
        self.clear()
        for d in self._drawables:
            d.draw()

    def update_gfx(self, _):
        pass
