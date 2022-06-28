import unittest

from fractions import Fraction
from mathematics import *
from quasiPolynomial import QuasiPolynomial as qp


class TestHelper(unittest.TestCase):

    def test_energy(self):
        self.assertEqual(energy(((2, -2), ())), 0)
        self.assertEqual(energy(((2, 2), ())), 4)
        self.assertEqual(energy(((2,),)), 2)
        self.assertEqual(energy(((2,), (2,), ())), 4)
        self.assertEqual(energy(((2,), (-2,), (-3,), (-1,))), -4)
        self.assertEqual(energy(((2.,),)), 2.)
        self.assertEqual(energy(((2,), (2.,), ())), 4.)
        self.assertEqual(energy(((0.2,), (2.,), ())), 2.2)
        self.assertEqual(energy(((Fraction(1,2),Fraction(1,2)), (2.,), ())), 3.0)

    def test_signum(self):
        self.assertEqual(signum(((2,),), ((-2,),)), 2)
        self.assertEqual(signum(((2,), ()), ((0,), ())), 1)
        self.assertEqual(signum(((-1,), (), (1,)), ((2,), (),(-1,))), -1)
        self.assertEqual(signum(((2,), (2,),(-3,)), ((2,), (2,),())), 0)
        self.assertEqual(signum(((2, 2), (2,)), ((2,), (2, 2))), 0)
        self.assertEqual(signum(((), (2,)), ((2,), (2, 2))), 0)
        self.assertEqual(signum(((),), ((),)), 0)
        self.assertEqual(signum(((), ()), ((2,), ())), -1)
        self.assertEqual(signum(((), ()), ((), (2,))), -1)
        self.assertEqual(signum(((2.,),), ((-2,),)), 2)
        self.assertEqual(signum(((), (0.2,)), ((0.1,), (2.2, -1))), 0)
        self.assertEqual(signum(((Fraction(1,10),),), ((Fraction(-10,23),),)), 2)
        self.assertEqual(signum(((Fraction(1,2),), (-0.5,)), ((Fraction(1,2),), ())), -1.)

    def test_exponential(self):
        self.assertEqual(exponential(((2, -2), ()), ((2,), ()), ((-2,), ())), qp.new_integer([[], [], [], [], [1]]))
        self.assertEqual(exponential(((1, -1),), ((1,),), ((-1,),)), qp.new_integer([[], [], [1]]))
        self.assertEqual(exponential(((0, 0), ()), ((0,), ()), ((0,), ())), qp.new_integer([[1]]))
        self.assertEqual(exponential(((0, 0), (),(1,-1)), ((0,), (),(1,)), ((0,), (),(-1,))), qp.new_integer([[], [], [1]]))
        self.assertEqual(exponential(((2., -2.), ()), ((2.,), ()), ((-2.,), ())), qp.new_integer([[], [], [], [], [1]]))
        self.assertEqual(exponential(((Fraction(1,2), -Fraction(1,2)), ()), ((Fraction(1,2),), ()), ((-Fraction(1,2),), ())), qp.new({1:[1]}))
        self.assertEqual(exponential(((Fraction(1,4), -Fraction(1,4)), ()), ((Fraction(1,4),), ()), ((-Fraction(1,4),), ())), qp.new({Fraction(1,2):[1]}))

    def test_partitions(self):
        self.assertEqual(partitions(((2,), ())), [])
        self.assertEqual(partitions(((2, -2), ())), [(((2,), ()), ((-2,), ()))])
        self.assertEqual(partitions(((2, -2),)), [(((2,),), ((-2,),))])
        self.assertEqual(partitions(((2, -2),(),(0,))), [(((), (), (0,)), ((2, -2), (), ())), 
                            (((2,), (), ()), ((-2,), (), (0,))),
                            (((2,), (), (0,)), ((-2,), (), ())),
                            (((2, -2), (), ()), ((), (), (0,)))])
        self.assertEqual(partitions(((1, 2, 3), ())), [(((1,), ()), ((2, 3), ())), (((1, 2), ()), ((3,), ()))])
        self.assertEqual(partitions(((1, 2), (4, 5))),
                         [(((), (4,)), ((1, 2), (5,))), (((), (4, 5)), ((1, 2), ())), (((1,), ()), ((2,), (4, 5))),
                          (((1,), (4,)), ((2,), (5,))), (((1,), (4, 5)), ((2,), ())), (((1, 2), ()), ((), (4, 5))),
                          (((1, 2), (4,)), ((), (5,)))])
        self.assertEqual(partitions(((2, -2.), ())), [(((2,), ()), ((-2.,), ()))])
        self.assertEqual(partitions(((2, -Fraction(1,2)), ())), [(((2,), ()), ((-Fraction(1,2),), ()))])

if __name__ == '__main__':
    unittest.main()
