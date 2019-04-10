from src.common.iupdatable import IUpdatable
from src.gfx.idrawable import IDrawable


class GfxItem(IDrawable, IUpdatable):
    def draw(self):
        pass

    def update(self, dt_sec):
        pass
