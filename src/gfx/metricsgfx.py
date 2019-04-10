import pyglet as pg

from src.gfx.gfxitem import GfxItem


class MetricsGfx(GfxItem):
    MAE_STR = "MAE: {:.3f}"

    def __init__(self, metrics, x_pos, y_pos):
        self._mae_label = pg.text.Label(
            self.MAE_STR.format(0.0),
            font_name="Courier New",
            color=(100, 225, 100, 175),
            font_size=10,
            x=x_pos,
            y=y_pos,
            anchor_x='left',
            anchor_y='top')
        self._metrics = metrics

    def update(self, dt_sec):
        self._mae_label.text = self.MAE_STR.format(self._metrics.mae[0])

    def draw(self):
        self._mae_label.draw()
