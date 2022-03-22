import unittest

from coefficientFunction import FunctionCollection
from quasiPolynomial import QuasiPolynomial as qp


class TestFunctionCollection(unittest.TestCase):
    def test_contains(self):
        collection = FunctionCollection()
        collection[(-2,)] = qp.new([[1]])
        collection[(0,)] = qp.new([[1]])
        collection[(2,)] = qp.new([[1]])
        collection[(1, 0)] = qp.zero()
        self.assertTrue((2,) in collection)
        self.assertFalse((1,) in collection)

    def test_keys(self):
        collection = FunctionCollection()
        collection[(-2,)] = qp.new([[1]])
        collection[(0,)] = qp.new([[1]])
        collection[(2,)] = qp.new([[1]])
        collection[(1, 0)] = qp.zero()
        self.assertEqual(collection.keys(), [(-2,), (0,), (2,), (1, 0)])

    def test_pretty_print(self):
        collection = FunctionCollection()
        collection[(-2,)] = qp.new([[1]])
        collection[(2,)] = qp.new([[1]])
        collection[(1, 0)] = qp.zero()
        self.assertEqual(str(collection), "['(-2,): [[Fraction(1, 1)]]', '(2,): [[Fraction(1, 1)]]', '(1, 0): []']")


class TestDifferentialEquation(unittest.TestCase):
    def test_differential_equation(self):
        collection = FunctionCollection()
        collection[(-2,)] = qp.new([[1]])
        collection[(0,)] = qp.new([[1]])
        collection[(2,)] = qp.new([[1]])
        self.assertEqual(collection[(2, -2)].function, qp.new([[1/2], [], [], [], [-1/2]]))
        self.assertEqual(collection[(1, 0)].function, qp.zero())


if __name__ == '__main__':
    unittest.main()
