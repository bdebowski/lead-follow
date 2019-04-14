import pyglet as pg

from src.gfx.gfxitem import GfxItem


class MetricsGfx(GfxItem):
    MAE_STR = "MAE_10k:  {:.3f}\n" \
              "MAE_100k: {:.3f}\n" \
              "MAE_1m:   {:.3f}"

    def __init__(self, metrics, x_pos, y_pos):
        self._mae_label = pg.text.Label(
            self.MAE_STR.format(0.0, 0.0, 0.0),
            font_name="Courier New",
            color=(100, 225, 100, 175),
            font_size=10,
            x=x_pos,
            y=y_pos,
            anchor_x='left',
            anchor_y='top',
            multiline=True,
            width=200)
        self._metrics = metrics

    def update(self, dt_sec):
        self._mae_label.text = self.MAE_STR.format(
            self._metrics.mae[0],
            self._metrics.mae[1],
            self._metrics.mae[2])

    def draw(self):
        self._mae_label.draw()
