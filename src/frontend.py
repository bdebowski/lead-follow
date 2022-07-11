import pyglet

from src.gfx.mainwin import MainWin


def create_and_run(gfxitems, winwidth, winheight):
    mainwin = MainWin(gfxitems, winwidth, winheight)

    pyglet.clock.schedule(mainwin.update)

    @mainwin.event
    def on_draw():
        mainwin.draw()

    pyglet.app.run()
