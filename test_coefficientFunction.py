from fractions import Fraction
import unittest

from coefficientFunction import FunctionCollection
from quasiPolynomial import Polynomial, Fraction, QuasiPolynomial as qp
from coefficientFunction import *

translation = dict([('-2.id', -2), ('0.id', 0), ('2.id', 2), ('id.-2', -2), ('id.0', 0), ('id.2', 2)])
collection = FunctionCollection(translation)
collection[(('-2.id',), ())] = qp.new_integer([[1]])
collection[(('0.id',), ())] = qp.new_integer([[1]])
collection[(('2.id',), ())] = qp.new_integer([[1]])
collection[((), ('id.-2',))] = qp.new_integer([[-1]])
collection[((), ('id.0',))] = qp.new_integer([[-1]])
collection[((), ('id.2',))] = qp.new_integer([[-1]])

# Classes and methods are tested in the order of alphabet 
# https://stackoverflow.com/questions/30286268/order-of-tests-in-python-unittest
class Test_A_FunctionCollection(unittest.TestCase):

    def test_contains(self):
        self.assertTrue((('-2.id',), ()) in collection)
        self.assertFalse((('1',), ()) in collection)

    def test_keys(self):
        self.assertEqual(collection.keys(),
                         [(('-2.id',), ()), (('0.id',), ()), (('2.id',), ()), ((), ('id.-2',)), ((), ('id.0',)),
                          ((), ('id.2',))])

    def test_print(self):
        self.assertEqual(str(collection),
                         str(["(('-2.id',), ()): [(0, ['1'])]", "(('0.id',), ()): [(0, ['1'])]",
                              "(('2.id',), ()): [(0, ['1'])]", "((), ('id.-2',)): [(0, ['-1'])]",
                              "((), ('id.0',)): [(0, ['-1'])]", "((), ('id.2',)): [(0, ['-1'])]"]))


class Test_B_DifferentialEquation(unittest.TestCase):
    def test_differential_equation(self):
        self.assertEqual(collection[(('2.id',), ())].function, qp.new_integer([[1]]))
        self.assertEqual(collection[((), ('id.-2',))].function, qp.new_integer([[-1]]))
        with self.assertRaises(AttributeError):
            # As the item is not in 'collection' a 'NoneType' object is returned 
            collection[(('0.id',), ('id.-2',))].function

    def test_key_sequence(self):
        self.assertEqual(sequence_to_key(collection.keys()[0]), (('-2.id',), ()))
        self.assertEqual(key_to_sequence((('0.id',), ())),collection.keys()[1])

    def test_calc(self):
        self.assertEqual(calc((('0.id','0.id'),tuple()),collection,translation,2),QuasiPolynomial.zero())
        self.assertEqual(calc((('-2.id','2.id'),tuple()),collection,translation,2),
                QuasiPolynomial({0: Polynomial([Fraction("-1/2")]), 4: Polynomial([Fraction("1/2")])}))
if __name__ == '__main__':
    unittest.main()
