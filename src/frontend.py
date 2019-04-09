import pyglet

from src.gfx.mainwin import MainWin


class Frontend:
    def __init__(self):
        self._mainwin = MainWin(1200, 800)

        pyglet.clock.schedule(self._mainwin.update_gfx)

        @self._mainwin.event
        def on_draw():
            self._mainwin.draw()

    def run(self):
        pyglet.app.run()
