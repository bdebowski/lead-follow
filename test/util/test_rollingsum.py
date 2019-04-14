from unittest import TestCase

from src.util.rollingsum import RollingSum


class TestRollingSum(TestCase):
    def test_rollingsum_rolls_sum_accurately(self):
        rsum = RollingSum(5)
        sums = [1, 3, 6, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        means = [1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        for i, v in enumerate(range(1, 13)):  # 1 through 12
            rsum.insert_next(v)
            self.assertEqual(sums[i], rsum.sum)
            self.assertEqual(means[i], rsum.mean)
