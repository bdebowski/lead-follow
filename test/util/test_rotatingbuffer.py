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

    def test_get_oldest_retrieves_accurately(self):
        buffer1 = RotatingBuffer(1)
        buffer10 = RotatingBuffer(10)

        for i in range(25):
            buffer1.set_next(i)
            buffer10.set_next(i)

        self.assertEqual(24, buffer1.get_oldest())
        self.assertEqual(15, buffer10.get_oldest())
