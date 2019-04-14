from unittest import TestCase

from src.util.rotatingbuffer import RotatingBuffer


class TestRotatingBuffer(TestCase):
    def test_buffer_rotates_accurately(self):
        buffer = RotatingBuffer(5)

        for i in range(1, 8):  # 1 through 7
            buffer.set_next(i)

        last_five = [7, 6, 5, 4, 3]
        for t_minus in range(5):  # 0 through 4
            self.assertEqual(last_five[t_minus], buffer.get_prev(t_minus))
