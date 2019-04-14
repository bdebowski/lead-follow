from unittest import TestCase

from src.util.rollingsum import RollingSum


class TestRollingSum(TestCase):
    def test_rollingsum_rolls_sum_accurately(self):
        rsum = RollingSum(5)
        sums = [1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        for i, v in enumerate(range(1, 13)):  # 1 through 12
            rsum.insert_next(v)
            self.assertEqual(sums[i], rsum.current_sum())
