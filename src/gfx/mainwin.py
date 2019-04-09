from abc import ABCMeta

import pyglet
from pyglet.gl import *

from src.gfx.idrawable import IDrawable
from src.common.iupdatable import IUpdatable


class _MetaClassCombined(type(pyglet.window.Window), ABCMeta):
    pass


class MainWin(pyglet.window.Window, IDrawable, IUpdatable, metaclass=_MetaClassCombined):
    def __init__(self, vectgfxes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._fps_display = pyglet.window.FPSDisplay(self)

        self._drawables = []
        self._drawables.append(self._fps_display)
        self._drawables.extend(vectgfxes)

        self._updatables = []
        self._updatables.extend(vectgfxes)

    def draw(self):
        self.clear()
        for d in self._drawables:
            d.draw()

    def update(self, dt_sec):
        for u in self._updatables:
            u.update(dt_sec)
