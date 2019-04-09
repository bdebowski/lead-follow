from abc import abstractmethod

import pyglet as pg

from src.gfx.idrawable import IDrawable
from src.common.iupdatable import IUpdatable


class VectGfx(IDrawable, IUpdatable):
    def __init__(self, iobservable, edge_list, colour):
        """
        Vector graphics for an observable object.
        :param iobservable: the source object represented by these graphics.
        :param edge_list: defines edges by listing connected point indices.  ex:
                            [0, 1, 1, 2, 2, 0] defines 3 edges connected by points 0, 1, 2
        :param colour: RGB colours tuple.  ex:
                         (200, 100, 100)
        """
        self._observable = iobservable
        self._batch = pg.graphics.Batch()

        num_pts = len(edge_list) // 2
        self._vertex_list = self._batch.add_indexed(
            num_pts,
            pg.gl.GL_LINES,
            None,
            edge_list,
            ('v2f', [0] * num_pts * 2),
            ('c3B', colour * num_pts))

    def draw(self):
        self._batch.draw()

    def update(self, _):
        self._vertex_list.vertices = self.compute_vertices()

    @abstractmethod
    def compute_vertices(self):
        """
        Computes x y coordinates for the vector graphics, given the observed state of the source observable.
        :return: 1-D array (list) of coordinates.  ex:
                   [x1, y1, x2, y2, x3, y3, x4, y4]
        """
        pass
