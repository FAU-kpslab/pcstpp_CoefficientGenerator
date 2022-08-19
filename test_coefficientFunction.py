import unittest

from quasiPolynomial import Polynomial, QuasiPolynomial as qp
from coefficientFunction import *
from sympy import I

# Defining symbols for testing
a = sym.symbols('a', positive=True)


translation = dict([(1, -2), (2, 0), (3, 2), (4, -2), (5, 0), (6, 2), (7, a), (8, -a), (9, I*a), (10, -I*a)])
collection = FunctionCollection(translation)
collection[((1,), ())] = qp.new_integer([[1]])
collection[((2,), ())] = qp.new_integer([[1]])
collection[((3,), ())] = qp.new_integer([[1]])
collection[((), (4,))] = qp.new_integer([[-1]])
collection[((), (5,))] = qp.new_integer([[-1]])
collection[((), (6,))] = qp.new_integer([[-1]])
collection[((7,), ())] = qp.new_integer([[1]])
collection[((8,), ())] = qp.new_integer([[1]])
collection[((9,), ())] = qp.new_integer([[1]])
collection[((10,), ())] = qp.new_integer([[1]])


# Classes and methods are tested in the order of alphabet 
# https://stackoverflow.com/questions/30286268/order-of-tests-in-python-unittest
class TestAFunctionCollection(unittest.TestCase):

    def test_contains(self):
        self.assertTrue(((1,), ()) in collection)
        self.assertFalse(((0,), ()) in collection)

    def test_keys(self):
        self.assertEqual(collection.keys(),
                         [((1,), ()), ((2,), ()), ((3,), ()), ((), (4,)), ((), (5,)),
                          ((), (6,)), ((7,), ()), ((8,), ()), ((9,), ()), ((10,), ())])

    def test_print(self):
        self.assertEqual(str(collection),
                         str(["((1,), ()): [(0, ['1'])]", "((2,), ()): [(0, ['1'])]",
                              "((3,), ()): [(0, ['1'])]", "((), (4,)): [(0, ['-1'])]",
                              "((), (5,)): [(0, ['-1'])]", "((), (6,)): [(0, ['-1'])]",
                              "((7,), ()): [(0, ['1'])]", "((8,), ()): [(0, ['1'])]",
                              "((9,), ()): [(0, ['1'])]", "((10,), ()): [(0, ['1'])]"]))


class TestBDifferentialEquation(unittest.TestCase):
    def test_differential_equation(self):
        self.assertEqual(collection[((3,), ())].function, qp.new_integer([[1]]))
        self.assertEqual(collection[((), (4,))].function, qp.new_integer([[-1]]))
        with self.assertRaises(AttributeError):
            # As the item is not in 'collection' a 'NoneType' object is returned 
            collection[((2,), (4,))].function

    def test_key_sequence(self):
        self.assertEqual(sequence_to_key(collection.keys()[0]), ((1,), ()))
        self.assertEqual(key_to_sequence(((2,), ())), collection.keys()[1])

    def test_calc(self):
        self.assertEqual(calc(((2, 2), tuple()), collection, translation, 2, signum, energy), QuasiPolynomial.zero())
        self.assertEqual(calc(((1, 3), tuple()), collection, translation, 2, signum, energy),
                         QuasiPolynomial({0: Polynomial([Fraction("-1/2")]), 4: Polynomial([Fraction("1/2")])}))
        self.assertEqual(calc(((8, 7), tuple()), collection, translation, a, signum_complex, energy),
                         QuasiPolynomial({0: Polynomial([-1/a]), 2*a: Polynomial([1/a])}))
        self.assertEqual(calc(((9, 10), tuple()), collection, translation, a, signum_complex, energy),
                         QuasiPolynomial({0: Polynomial([-I/a]), 2*a: Polynomial([I/a])}))
        self.assertEqual(calc(((10, 9), tuple()), collection, translation, a, signum_complex, energy),
                         QuasiPolynomial({0: Polynomial([I/a]), 2*a: Polynomial([-I/a])}))


if __name__ == '__main__':
    unittest.main()
