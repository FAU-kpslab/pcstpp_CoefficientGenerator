import unittest

from fractions import Fraction
from mathematics import *
from quasiPolynomial import QuasiPolynomial as qp

# Defining symbols for testing
a, b = sym.symbols('a b', positive=True)


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
        self.assertEqual(energy(((Fraction(1, 2), Fraction(1, 2)), (2.,), ())), 3.0)
        self.assertEqual(energy(((2,), (2j,), ())), 2 + 2j)
        # officially `complex` is not supported, maybe rewrite this test
        # https://stackoverflow.com/questions/12136762/assertalmostequal-in-python-unit-test-for-collections-of-floats#:~:text=The%20assertAlmostEqual%20%28x%2C%20y%29%20method%20in%20Python%27s%20unit,%28%29%20is%20that%20it%20only%20works%20on%20floats.
        self.assertAlmostEqual(energy(((2, -4 + 1.2j), (2.1 - 1.2j,), ())), 0.1)
        self.assertEqual(energy(((2, 2 * a),)), 2 + 2 * a)
        self.assertEqual(energy(((-2 * a, 2 * a),)), 0)
        self.assertEqual(energy(((Fraction(1, 2) * a ** 2, Fraction(1, 2)), (2,), ())),
                         Fraction(1, 2) * a ** 2 + Fraction(5, 2))

    def test_energy_broad(self):
        self.assertEqual(energy_broad(((2, -2), ()), 0), 0)
        self.assertEqual(energy_broad(((2, 2), ()), 2), 4)
        self.assertEqual(energy_broad(((2, 2), ()), 4), 0)
        self.assertEqual(energy_broad(((2,), (-2,), (-3,), (-1,)), 2), -4)
        self.assertEqual(energy_broad(((2,), (-2,), (-3,), (-1,)), 5), 0)
        self.assertEqual(energy_broad(((2, 2.2), ()), 4), 4.2)
        self.assertEqual(energy_broad(((Fraction(1, 3), 2.2), ()), 1), Fraction(1, 3) + 2.2)
        self.assertEqual(energy_broad(((2, 2.2, -1.01), ()), 4), 0)
        self.assertEqual(energy_broad(((2, 2. + Fraction(1, 10000000000000)), ()), 4), 0)
        self.assertEqual(energy_broad_expr(((2, a, -1), ()), 1), 2-1+a)
        self.assertEqual(energy_broad_expr(((2, a), (-2*a,)), 1), sym.Piecewise((2-a,sym.Abs(2-a)>1),(0,True)))
        self.assertEqual(energy_broad_expr(((2, -a),), a), sym.Piecewise((2-a,sym.Abs(2-a)>a),(0,True)))

    def test_signum(self):
        self.assertEqual(signum(((2,),), ((-2,),)), 2)
        self.assertEqual(signum(((2,), ()), ((0,), ())), 1)
        self.assertEqual(signum(((-1,), (), (1,)), ((2,), (), (-1,))), -1)
        self.assertEqual(signum(((2,), (2,), (-3,)), ((2,), (2,), ())), 0)
        self.assertEqual(signum(((2, 2), (2,)), ((2,), (2, 2))), 0)
        self.assertEqual(signum(((), (2,)), ((2,), (2, 2))), 0)
        self.assertEqual(signum(((),), ((),)), 0)
        self.assertEqual(signum(((), ()), ((2,), ())), -1)
        self.assertEqual(signum(((), ()), ((), (2,))), -1)
        self.assertEqual(signum(((2.,),), ((-2,),)), 2)
        self.assertEqual(signum(((), (0.2,)), ((0.1,), (2.2, -1))), 0)
        self.assertEqual(signum(((Fraction(1, 10),),), ((Fraction(-10, 23),),)), 2)
        self.assertEqual(signum(((Fraction(1, 2),), (-0.5,)), ((Fraction(1, 2),), ())), -1.)

    def test_signum_broad(self):
        self.assertEqual(signum_broad(((2,),), ((-2,),), delta=1), 2)
        self.assertEqual(signum_broad(((2,),), ((-2,),), delta=3), 0)
        self.assertEqual(signum_broad(((-1,), (), (1,)), ((2,), (), (-1,)), delta=1), 0)
        self.assertEqual(signum_broad(((), (2,)), ((2,), (2, 2)), delta=0), 0)
        self.assertEqual(signum_broad(((), (-1,)), ((2,), ()), delta=1), -1)
        self.assertIsInstance(signum_broad(((2,),), ((-2,),), delta=1), int)
        self.assertEqual(signum_broad(((2.2,), (-1.3,)), ((2,), ()), delta=1), -1)
        self.assertEqual(signum_broad(((Fraction(1, 3),), ()), ((-0.5,), ()), delta=1), 0)
        self.assertEqual(signum_broad_expr(((2*a,),), ((-2*a,),), delta=5*a), 0)
        self.assertEqual(signum_broad_expr(((2*a,),), ((-2,),), delta=2*a), -sym.Piecewise((-1,a<1),(0,True)))

    def test_signum_complex(self):
        self.assertEqual(signum_complex(((2,),), ((-2,),)), 2)
        self.assertEqual(signum_complex(((2j,),), ((-2j,),)), -2j)
        self.assertEqual(signum_complex(((2j, -1j),), ((-2j, -3j),)), -2j)
        self.assertAlmostEqual(signum_complex(((1 + 1j,),), ((-2j - 2,),)),
                               np.exp(-np.pi / 4 * 1j) - np.exp(-5 * np.pi / 4 * 1j))
        self.assertAlmostEqual(signum_complex(((1 + 3j, -2 + 2j),), ((-2j - 2, 1),)),
                               np.exp(-np.angle(-1 + 5j) * 1j) - np.exp(-np.angle(-1 - 2j) * 1j))
        self.assertEqual(signum_complex(((1 + 1j,),), ((+2j + 2,),)), 0)
        self.assertEqual(signum_complex(((1 + 1j, 0), (-1, 3)), ((+2j + 2, 0), (2, 2))), 0)
        self.assertEqual(signum_expr(((2 * a,),), ((2 * a,),)), 0)
        self.assertEqual(signum_expr(((2 * a,),), ((3 * a,),)), 0)
        self.assertEqual(signum_expr(((2 * a,),), ((0,),)), (2 * a).conjugate() / abs(2 * a))
        self.assertEqual(signum_expr(((2 * a,),), ((-3 * a,),)),
                         (2 * a).conjugate() / abs(2 * a) + (3 * a).conjugate() / abs(3 * a))
        self.assertEqual(signum_expr(((2 * a,),), ((a -1,),)), 1 - sym.Piecewise(((a - 1)/sym.Abs(a - 1), sym.Ne(a, 1)), (0, True)))
        self.assertEqual(signum_expr(((a, 2, -2*a),(-3,a,0)), ((a -2,),)), -1 - sym.Piecewise(((a - 2)/sym.Abs(a - 2), sym.Ne(a, 2)), (0, True)))

    def test_exponential(self):
        self.assertEqual(exponential(((2, -2), ()), ((2,), ()), ((-2,), ()), energy),
                         qp.new_integer([[], [], [], [], [1]]))
        self.assertEqual(exponential(((1, -1),), ((1,),), ((-1,),), energy), qp.new_integer([[], [], [1]]))
        self.assertEqual(exponential(((0, 0), ()), ((0,), ()), ((0,), ()), energy), qp.new_integer([[1]]))
        self.assertEqual(exponential(((0, 0), (), (1, -1)), ((0,), (), (1,)), ((0,), (), (-1,)), energy),
                         qp.new_integer([[], [], [1]]))
        self.assertEqual(exponential(((2., -2.), ()), ((2.,), ()), ((-2.,), ()), energy),
                         qp.new_integer([[], [], [], [], [1]]))
        self.assertEqual(
            exponential(((Fraction(1, 2), -Fraction(1, 2)), ()), ((Fraction(1, 2),), ()), ((-Fraction(1, 2),), ()),
                        energy), qp.new({1: [1]}))
        self.assertEqual(
            exponential(((Fraction(1, 4), -Fraction(1, 4)), ()), ((Fraction(1, 4),), ()), ((-Fraction(1, 4),), ()),
                        energy), qp.new({Fraction(1, 2): [1]}))
        self.assertEqual(exponential(((1 + 1j, -1 - 1j), ()), ((1 + 1j,), ()), ((-1 - 1j,), ()), energy),
                         qp.new({2 * abs(1 + 1j): [1]}))
        self.assertEqual(exponential(((2 * a, -2 * a),), ((2 * a,),), ((-2 * a,),), energy),
                         qp.new({4 * sym.Abs(a): [1]}))
        self.assertEqual(exponential(((2 * a, -2), ()), ((2 * a,), ()), ((-2,), ()), energy),
                         qp.new({2 * sym.Abs(a) + 2 - sym.Abs(2 * a - 2): [1]}))
        self.assertEqual(
            exponential(((Fraction(3, 2) * a, -Fraction(3, 2) * a),), ((3 * a / 2,),), ((-3 * a / 2,),), energy),
            qp.new({6 * sym.Abs(a / 2): [1]}))

    def test_partitions(self):
        self.assertEqual(partitions(((2,), ())), [])
        self.assertEqual(partitions(((2, -2), ())), [(((2,), ()), ((-2,), ()))])
        self.assertEqual(partitions(((2, -2),)), [(((2,),), ((-2,),))])
        self.assertEqual(partitions(((2, -2), (), (0,))), [(((), (), (0,)), ((2, -2), (), ())),
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
