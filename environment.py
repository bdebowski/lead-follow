from idrawable import IDrawable
from iupdatable import IUpdatable


class Environment(IDrawable, IUpdatable):
    def __init__(self):
        self._drawables = []
        self._updatables = []

    def draw(self):
        # Draw env
        # Draw each drawable
        for drawable in self._drawables:
            drawable.draw()

    def update(self, dt):
        # update self
        # update each updatable
        for updatable in self._updatables:
            updatable.update(dt)
