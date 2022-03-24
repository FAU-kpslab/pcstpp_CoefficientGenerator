import unittest

from coefficientFunction import FunctionCollection
from quasiPolynomial import QuasiPolynomial as qp


class TestFunctionCollection(unittest.TestCase):
    def test_contains(self):
        translation = dict([('-2.id', -2), ('0.id', 0), ('2.id', 2), ('id.-2', -2), ('id.0', 0), ('id.2', 2)])
        collection = FunctionCollection(translation, max_energy=2)
        collection[(('-2.id',), ())] = qp.new([[1]])
        collection[(('0.id',), ())] = qp.new([[1]])
        collection[(('2.id',), ())] = qp.new([[1]])
        collection[((), ('id.-2',))] = qp.new([[-1]])
        collection[((), ('id.0',))] = qp.new([[-1]])
        collection[((), ('id.2',))] = qp.new([[-1]])
        self.assertTrue((('-2.id',), ()) in collection)
        self.assertFalse((('1',), ()) in collection)

    def test_keys(self):
        translation = dict([('-2.id', -2), ('0.id', 0), ('2.id', 2), ('id.-2', -2), ('id.0', 0), ('id.2', 2)])
        collection = FunctionCollection(translation, max_energy=2)
        collection[(('-2.id',), ())] = qp.new([[1]])
        collection[(('0.id',), ())] = qp.new([[1]])
        collection[(('2.id',), ())] = qp.new([[1]])
        collection[((), ('id.-2',))] = qp.new([[-1]])
        collection[((), ('id.0',))] = qp.new([[-1]])
        collection[((), ('id.2',))] = qp.new([[-1]])
        self.assertEqual(collection.keys(),
                         [(('-2.id',), ()), (('0.id',), ()), (('2.id',), ()), ((), ('id.-2',)), ((), ('id.0',)),
                          ((), ('id.2',))])

    def test_print(self):
        translation = dict([('-2.id', -2), ('0.id', 0), ('1.id', 1), ('2.id', 2)])
        collection = FunctionCollection(translation, max_energy=2)
        collection[(('-2.id',), ())] = qp.new([[1]])
        collection[(('2.id',), ())] = qp.new([[1]])
        collection[(('1.id', '0.id'), ())] = qp.zero()
        self.assertEqual(str(collection),
                         str(["(('-2.id',), ()): [[Fraction(1, 1)]]", "(('2.id',), ()): [[Fraction(1, 1)]]",
                              "(('1.id', '0.id'), ()): []"]))


class TestDifferentialEquation(unittest.TestCase):
    def test_differential_equation(self):
        translation = dict([('-2.id', -2), ('0.id', 0), ('2.id', 2), ('id.-2', -2), ('id.0', 0), ('id.2', 2)])
        collection = FunctionCollection(translation, max_energy=2)
        collection[(('-2.id',), ())] = qp.new([[1]])
        collection[(('0.id',), ())] = qp.new([[1]])
        collection[(('2.id',), ())] = qp.new([[1]])
        collection[((), ('id.-2',))] = qp.new([[-1]])
        collection[((), ('id.0',))] = qp.new([[-1]])
        collection[((), ('id.2',))] = qp.new([[-1]])
        self.assertEqual(collection[(('2.id', '-2.id'), ())].function, qp.new([[1/2], [], [], [], [-1/2]]))
        self.assertEqual(collection[(('0.id',), ('id.-2',))].function, qp.zero())


if __name__ == '__main__':
    unittest.main()
