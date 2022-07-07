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
        self.assertEqual(energy(((2,), (2j,), ())), 2+2j)
        self.assertAlmostEqual(energy(((2,-4+1.2j), (2.1-1.2j,), ())), 0.1)

    def test_energy_broad(self):
        self.assertEqual(energy_broad(((2, -2), ()),0), 0)
        self.assertEqual(energy_broad(((2, 2), ()),2), 4)
        self.assertEqual(energy_broad(((2, 2), ()),4), 0)
        self.assertEqual(energy_broad(((2,), (-2,), (-3,), (-1,)),2), -4)
        self.assertEqual(energy_broad(((2,), (-2,), (-3,), (-1,)),5), 0)
        self.assertEqual(energy_broad(((2, 2.2), ()),4), 4.2)
        self.assertEqual(energy_broad(((Fraction(1,3), 2.2), ()),1), Fraction(1,3)+2.2)
        self.assertEqual(energy_broad(((2, 2.2, -1.01), ()),4), 0)
        self.assertEqual(energy_broad(((2, 2.+ Fraction(1,10000000000000)), ()),4), 0)

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

    def test_signum_broad(self):
        self.assertEqual(signum_broad(((2,),), ((-2,),),delta=1), 2)
        self.assertEqual(signum_broad(((2,),), ((-2,),),delta=3), 0)
        self.assertEqual(signum_broad(((-1,), (), (1,)), ((2,), (),(-1,)),delta=1), 0)
        self.assertEqual(signum_broad(((), (2,)), ((2,), (2, 2)),delta=0), 0)
        self.assertEqual(signum_broad(((), (-1,)), ((2,), ()),delta=1), -1)
        self.assertIsInstance(signum_broad(((2,),), ((-2,),),delta=1), int)
        self.assertEqual(signum_broad(((2.2,), (-1.3,)), ((2,), ()),delta=1), -1)
        self.assertEqual(signum_broad(((Fraction(1,3),), ()), ((-0.5,), ()),delta=1), 0)

    def test_signum_complex(self):
        self.assertEqual(signum_complex(((2,),), ((-2,),)), 2)
        self.assertEqual(signum_complex(((2j,),), ((-2j,),)), -2j)
        self.assertEqual(signum_complex(((2j,-1j),), ((-2j,-3j),)), -2j)
        # TODO: Is this the correct behaviour for the Uhrig generator?
        self.assertAlmostEqual(signum_complex(((1+1j,),), ((-2j-2,),)), np.exp(-np.pi/4*1j) - np.exp(-5*np.pi/4*1j))
        self.assertAlmostEqual(signum_complex(((1+3j,-2+2j),), ((-2j-2,1),)), np.exp(-np.angle(-1+5j)*1j) - np.exp(-np.angle(-1-2j)*1j))
        self.assertEqual(signum_complex(((1+1j,),), ((+2j+2,),)), 0)
        self.assertEqual(signum_complex(((1+1j,0),(-1,3)), ((+2j+2,0),(2,2))), 0)
        
    def test_exponential(self):
        self.assertEqual(exponential(((2, -2), ()), ((2,), ()), ((-2,), ()), energy), qp.new_integer([[], [], [], [], [1]]))
        self.assertEqual(exponential(((1, -1),), ((1,),), ((-1,),), energy), qp.new_integer([[], [], [1]]))
        self.assertEqual(exponential(((0, 0), ()), ((0,), ()), ((0,), ()), energy), qp.new_integer([[1]]))
        self.assertEqual(exponential(((0, 0), (),(1,-1)), ((0,), (),(1,)), ((0,), (),(-1,)), energy), qp.new_integer([[], [], [1]]))
        self.assertEqual(exponential(((2., -2.), ()), ((2.,), ()), ((-2.,), ()), energy), qp.new_integer([[], [], [], [], [1]]))
        self.assertEqual(exponential(((Fraction(1,2), -Fraction(1,2)), ()), ((Fraction(1,2),), ()), ((-Fraction(1,2),), ()), energy), qp.new({1:[1]}))
        self.assertEqual(exponential(((Fraction(1,4), -Fraction(1,4)), ()), ((Fraction(1,4),), ()), ((-Fraction(1,4),), ()), energy), qp.new({Fraction(1,2):[1]}))
        self.assertEqual(exponential(((1+1j, -1-1j), ()), ((1+1j,), ()), ((-1-1j,), ()), energy), qp.new({2*abs(1+1j):[1]}))

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

if __name__ == '__main__':
    unittest.main()
