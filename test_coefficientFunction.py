import unittest

from coefficientFunction import FunctionCollection
from quasiPolynomial import QuasiPolynomial as qp

translation = dict([('-2.id', -2), ('0.id', 0), ('2.id', 2), ('id.-2', -2), ('id.0', 0), ('id.2', 2)])
collection = FunctionCollection(translation)
collection[[('-2.id',), ()]] = qp.new([[1]])
collection[[('0.id',), ()]] = qp.new([[1]])
collection[[('2.id',), ()]] = qp.new([[1]])
collection[[(), ('id.-2',)]] = qp.new([[-1]])
collection[[(), ('id.0',)]] = qp.new([[-1]])
collection[[(), ('id.2',)]] = qp.new([[-1]])


class TestFunctionCollection(unittest.TestCase):

    def test_contains(self):
        self.assertTrue((('-2.id',), ()) in collection)
        self.assertFalse((('1',), ()) in collection)

    def test_keys(self):
        self.assertEqual(collection.keys(),
                         [[('-2.id',), ()], [('0.id',), ()], [('2.id',), ()], [(), ('id.-2',)], [(), ('id.0',)],
                          [(), ('id.2',)]])

    def test_print(self):
        self.assertEqual(str(collection),
                         str(["[('-2.id',), ()]: [['1']]", "[('0.id',), ()]: [['1']]",
                              "[('2.id',), ()]: [['1']]", "[(), ('id.-2',)]: [['-1']]",
                              "[(), ('id.0',)]: [['-1']]", "[(), ('id.2',)]: [['-1']]"]))


class TestDifferentialEquation(unittest.TestCase):
    def test_differential_equation(self):
        self.assertEqual(collection[(('2.id',), ())].function, qp.new([[1]]))
        self.assertEqual(collection[((), ('id.-2',))].function, qp.new([[-1]]))
        with self.assertRaises(AttributeError):
            # As the item is not in 'collection' a 'NoneType' object is returned 
            collection[(('0.id',), ('id.-2',))].function


if __name__ == '__main__':
    unittest.main()
