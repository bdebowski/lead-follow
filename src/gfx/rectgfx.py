from src.gfx.vectgfx import VectGfx


class RectGfx(VectGfx):
    def __init__(self, iobservable, width, height, colour):
        """
        Rectangular vector graphics for an observable object.
        :param iobservable: the source object which will be drawn as a rectangle.
        :param width: the rectangle's width
        :param height: the rectangle's height
        :param colour: the rectangle's colour as an RGB tuple (r, g, b)
        """
        super().__init__(iobservable, [0, 1, 1, 2, 2, 3, 3, 0], colour)

        self._width = width
        self._height = height

    def compute_vertices(self):
        """
        Computes coordinates for rectangle points with respect to the centre point observed.
        :return: 1-D array (list) of coordinates:
                   [x1, y1, x2, y2, x3, y3, x4, y4]
                 Where (x1, y1) is the bottom left corner, and we go around clockwise from there.
        """
        # centre is (x, y) tuple of centre point coordinates
        centre = self._observable.position()
        c_x, c_y = centre[0], centre[1]

        h_w = self._width / 2.0
        h_h = self._height / 2.0

        x1 = c_x - h_w
        y1 = c_y - h_h
        x2 = x1
        y2 = c_y + h_h
        x3 = c_x + h_w
        y3 = y2
        x4 = x3
        y4 = y1

        return [x1, y1, x2, y2, x3, y3, x4, y4]
