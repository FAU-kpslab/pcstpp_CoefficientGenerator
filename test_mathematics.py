import unittest

from mathematics import *
from quasiPolynomial import QuasiPolynomial as qp


class TestHelper(unittest.TestCase):

    def test_energy(self):
        self.assertEqual(energy((2, -2)), 0)
        self.assertEqual(energy((2, 2)), 4)
        self.assertEqual(energy((2,)), 2)

    def test_signum(self):
        self.assertEqual(signum((2,), (-2,)), 2)
        self.assertEqual(signum((2,), (0,)), 1)
        self.assertEqual(signum((0,), (2,)), -1)

    def test_exponential(self):
        self.assertEqual(exponential((2, -2), (2,), (-2,)), qp.new([[], [], [], [], [1]]))
        self.assertEqual(exponential((1, -1), (1,), (-1,)), qp.new([[], [], [1]]))

    def test_partitions(self):
        self.assertEqual(partitions((2, -2)), [((2,), (-2,))])
        self.assertEqual(partitions((2,)), [])
        self.assertEqual(partitions((1, 2, 3)), [((1,), (2, 3)), ((1, 2), (3,))])


if __name__ == '__main__':
    unittest.main()