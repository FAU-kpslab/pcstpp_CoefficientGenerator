import unittest
import numpy as np
import sympy as sym

from fractions import Fraction
from quasiPolynomial import Polynomial as P
from quasiPolynomial import QuasiPolynomial as QP

# Defining symbols for testing
a, b = sym.symbols('a b')


class TestPolynomial(unittest.TestCase):

    def test_str(self):
        self.assertEqual(str(P([Fraction(2), Fraction(1, 2), 2, 3.5, 1., 1j, 1.2 + 2.1j, 2 * a + 1])),
                         "['2', '1/2', '2', '3.5', '1.0', '1j', '(1.2+2.1j)', '2*a + 1']")

    def test_simplify(self):
        self.assertEqual(str(P([Fraction(2), Fraction(4), Fraction(8), Fraction(0)]).simplify()),
                         str(P([Fraction(2), Fraction(4), Fraction(8)])))
        self.assertEqual(str(P([Fraction(2), Fraction(4), Fraction(0), Fraction(0)]).simplify()),
                         str(P([Fraction(2), Fraction(4)])))
        self.assertEqual(str(P([Fraction(0), Fraction(0), Fraction(0)]).simplify()), str(P.zero()))
        self.assertEqual(str(P.zero().simplify()), str(P.zero()))
        self.assertEqual(str(P([Fraction(5, 10)]).simplify()), str(P([Fraction(1, 2)])))
        self.assertEqual(str(P([1.0, 1e-25]).simplify()), str(P([1.0])))
        self.assertEqual(str(P([1.0 + 2j, 1e-10j]).simplify()), str(P([1.0 + 2j])))
        self.assertEqual(str(P([2 * a]).simplify()), str(P([2 * a + 1 - 1])))
        self.assertEqual(str(P([2 * a - 2 * a]).simplify()), str(P.zero()))
        self.assertEqual(str(P([2 * a, Fraction(1, 2) * 2]).simplify()), str(P([2 * a, 1])))
        # TODO: Add a function that checks for equality without simplification.

    def test_new(self):
        self.assertEqual(str(P.new(['3/15', Fraction(1, 2), 2, 3.5, 1., a])), "['1/5', '1/2', '2', '3.5', '1.0', 'a']")
        with self.assertRaises(TypeError):
            P.new([0, 2, [1]])
        with self.assertRaises(TypeError):
            P.new([np.float32(2.1), 1])

    def test_copy(self):
        temp = P.new([2, 4, 8])
        temp_copy = temp.copy()
        self.assertNotEqual(id(temp), id(temp.copy()))
        self.assertEqual(temp, temp_copy)

    def test_eq(self):
        self.assertTrue(P.new([2, 4, 8]) == P.new([2, 4, 8]))
        self.assertFalse(P.new([2, 8]) == P.new([2, 4, 8]))
        self.assertFalse(P.new([2, 4, 8]) == P.new([2, 4]))
        self.assertTrue(P.new([0]) == P.zero())
        self.assertTrue(P.new([2, 4, 0, 0]) == P.new([2, 4]))
        self.assertTrue(P.new([4 / 2, 4, 8]) == P.new([2, 4, 8]))
        self.assertTrue(P.new([2., 4, 8]) == P.new([2, 4, 8]))
        self.assertTrue(P.new([1e-50, 2]) == P.new([0, 2]))
        self.assertTrue(P.new([1 / 3]) == P.new(['1/3']))
        self.assertTrue(P.new([0.1 + 0.2]) == P.new([0.3]))
        self.assertTrue(P.new([0.1j + 0.2 - 0.1j]) == P.new([0.2]))
        # Assuming `scalar` of type `Expr` to be exact
        self.assertTrue(P.new([a, 4, 0, 0]) == P.new([a, 4]))

    def test_pretty_print(self):
        self.assertEqual(P.zero().pretty_print(), '0')
        self.assertEqual(P.new([0]).pretty_print(), '0')
        self.assertEqual(P.new([2]).pretty_print(), '2')
        self.assertEqual(P.new([-2]).pretty_print(), '-2')
        self.assertEqual(P.new([2, 4]).pretty_print(), '2+4x')
        self.assertEqual(P.new([Fraction(1, 2), 4]).pretty_print(), '1/2+4x')
        self.assertEqual(P.new(['1/2', 4]).pretty_print(), '1/2+4x')
        self.assertEqual(P.new(['0.5', 4]).pretty_print(), '1/2+4x')
        self.assertEqual(P.new([0.5, 4]).pretty_print(), '0.5+4x')
        self.assertEqual(P.new([0.5j, 4]).pretty_print(), '0.5j+4x')
        self.assertEqual(P.new([-2, -4]).pretty_print(), '-2-4x')
        self.assertEqual(P.new([0, 4]).pretty_print(), '4x')
        self.assertEqual(P.new([2, 4j - 2, 8]).pretty_print(), '2+(-2+4j)x+8x^2')
        self.assertEqual(P.new([0, 4, 8, 0, 0]).pretty_print(), '4x+8x^2')
        self.assertEqual(P.new([2, 4, 0, 100]).pretty_print(), '2+4x+100x^3')
        self.assertEqual(P.new([2, 4*a, 0, 1]).pretty_print(), '2+4*ax+x^3')
        self.assertEqual(P.new([2, 2+4*a**2, 0, 100]).pretty_print(), '2+(4*a^2 + 2)x+100x^3')
        self.assertEqual(P.new([2, a**5, 0, 0]).pretty_print(), '2+a^5x')
        self.assertEqual(P.new([2, 1, 0]).pretty_print(), '2+x')

    def test_scalar_multiplication(self):
        self.assertEqual(P.new([2, 4, 8]), P.new([1, 2, 4]).scalar_multiplication(2))
        self.assertEqual(P.zero(), P.new([1, 2, 4]).scalar_multiplication(0))
        self.assertEqual(P.new([1, 3 / 2, 2]), P.new([2, 3, 4]).scalar_multiplication(Fraction(1, 2)))
        self.assertEqual(P.new([1, 3 / 2, 2]), P.new([2, 3, 4]).scalar_multiplication(1 / 2))
        self.assertEqual(P.new([1, 3 / 2, 2]), P.new([2, 3, 4]).scalar_multiplication(0.5))
        self.assertEqual(P.new([1j, 3j / 2, 2j]), P.new([2, 3, 4]).scalar_multiplication(0.5j))
        self.assertEqual(P.new([4, 6, 2 * a]), P.new([2, 3, a]).scalar_multiplication(2))
        self.assertEqual(P.new([4 * a, 6 * a, 2 * a ** 2]), P.new([2, 3, a]).scalar_multiplication(2 * a))

    def test_negation(self):
        self.assertEqual(-P.new([2, 4, 8]), P.new([-2, -4, -8]))

    def test_addition(self):
        self.assertEqual(P.new([2, 4, 8]) + P.new([1, 5, 70]), P.new([3, 9, 78]))
        self.assertEqual(P.new([5]) + P.new([2, 4]), P.new([7, 4]))
        self.assertEqual(P.new([2, 4]) + P.new([5]), P.new([7, 4]))
        self.assertEqual(P.new([1]) + P.new([1]), P.new([2]))
        self.assertEqual(P.new([1]) + P.new([1, 2j + 1]), P.new([2, 2j + 1]))
        self.assertEqual(P.zero() + P.new([2, 4]), P.new([2, 4]))
        self.assertEqual(P.zero() + P.zero(), P.zero())
        self.assertEqual(P.new([a]) + P.new([2, 4]), P.new([2 + a, 4]))
        self.assertEqual(P.zero() + P.new([a]), P.new([a]))

    def test_multiplication(self):
        self.assertEqual(P.new([1, 2]) * P.new([5, 1]), P.new([5, 11, 2]))
        self.assertEqual(P.new([1, 2, 3, 4]) * P.new([5, 1]), P.new([5, 11, 17, 23, 4]))
        self.assertEqual(P.new([5, 1]) * P.new([1, 2, 3, 4]), P.new([5, 11, 17, 23, 4]))
        self.assertEqual(P.new([1]) * P.new([1, 2, 3, 4]), P.new([1, 2, 3, 4]))
        self.assertEqual(P.new([1, 2, 3, 4]) * P.new([1]), P.new([1, 2, 3, 4]))
        self.assertEqual(- P.new([1j, 2j, 3j, 4j]) * P.new([1j]), P.new([1, 2, 3, 4]))
        self.assertEqual(P.zero() * P.new([1, 2, 3, 4]), P.zero())
        self.assertEqual(P.new([1, 2, 3, 4]) * P.zero(), P.zero())
        self.assertEqual(P.new([1, 2, 3, 4]) * 2, P.new([2, 4, 6, 8]))
        self.assertEqual(2 * P.new([1, 2, 3, 4]), P.new([2, 4, 6, 8]))
        self.assertEqual(P.new([a]) * P.new([1, 2, 3, 4]), P.new([a, 2 * a, 3 * a, 4 * a]))

    def test_integrate(self):
        self.assertEqual(P.new([5, 5, 7]).integrate(), P.new([0, 5, Fraction(5, 2), Fraction(7, 3)]))
        self.assertEqual(P.new([5, 5j + 2.1, 7.]).integrate(), P.new([0, 5, (5j + 2.1) / 2, 7 / 3]))
        self.assertEqual(P.zero().integrate(), P.zero())
        self.assertEqual(P.new([5, 5, a]).integrate(), P.new([0, 5, Fraction(5, 2), a / 3]))

    def test_diff(self):
        self.assertEqual(P.new([5, 5, 7]).diff(), P.new([5, 14]))
        self.assertEqual(P.zero().diff(), P.zero())
        self.assertEqual(P.new([5]).diff(), P.zero())
        self.assertEqual(P.new([5, 5 * a, 7]).diff(), P.new([5 * a, 14]))


class TestQuasiPolynomial(unittest.TestCase):

    def test_sort(self):
        self.assertEqual(str(QP({2: P.new([2, 4, 8]), 1: P.new([1, 5, 25]), 0: P.new([1])}).sort()),
                         str(QP({0: P.new([1]), 1: P.new([1, 5, 25]), 2: P.new([2, 4, 8])})))
        self.assertEqual(str(QP({Fraction(1, 2): P.new([2, 4, 8]), 1.1: P.new([1, 5, 25]), 0: P.new([1])}).sort()),
                         str(QP({0: P.new([1]), Fraction(1, 2): P.new([2, 4, 8]), 1.1: P.new([1, 5, 25])})))

    def test_str(self):
        self.assertEqual(str(QP({0: P.new([2, 4, 8]), 1: P.new([1, 5, 25]), 2: P.new([3, 9])})),
                         "[(0, ['2', '4', '8']), (1, ['1', '5', '25']), (2, ['3', '9'])]")
        self.assertEqual(str(QP({0: P.new([2, 4, 8]), 2: P.new([3, 9]), 1: P.new([1, 5, 25])})),
                         "[(0, ['2', '4', '8']), (1, ['1', '5', '25']), (2, ['3', '9'])]")
        self.assertEqual(str(QP({0: P.new([2, 4, 8]), Fraction(1, 2): P.new([3, 2j + 9]), 1.1: P.new([1, 5, 25])})),
                         "[(0, ['2', '4', '8']), (Fraction(1, 2), ['3', '(9+2j)']), (1.1, ['1', '5', '25'])]")
        self.assertEqual(str(QP({0: P.new([2, 4, 8]), a: P.new([1, 5, 25]), 2: P.new([3, a])})),
                         "[(0, ['2', '4', '8']), (2, ['3', 'a']), (a, ['1', '5', '25'])]")

    def test_simplify(self):
        self.assertEqual(str(QP({0: P.new([2, 4, 8, 0])}).simplify()), str(QP({0: P.new([2, 4, 8])})))
        self.assertEqual(str(QP({0: P.new([2, 4, 0, 0])}).simplify()), str(QP({0: P.new([2, 4])})))
        self.assertEqual(str(QP({0: P.new([0, 0, 0])}).simplify()), str(QP({})))
        self.assertEqual(str(QP({}).simplify()), str(QP({})))
        self.assertEqual(str(QP({0: P.new([2, 4, 8]), Fraction(1, 2): P.new([1, 5, 25]), 2: P.new([0, 0])}).simplify()),
                         str(QP({0: P.new([2, 4, 8]), Fraction(1, 2): P.new([1, 5, 25])})))
        self.assertEqual(str(QP({0: P.new([2, 4, 8]), 1: P.new([0, 0, 0]), 2: P.new([3, 9])}).simplify()),
                         str(QP({0: P.new([2, 4, 8]), 2: P.new([3, 9])})))
        self.assertEqual(str(QP({0: P.zero(), 1: P.zero(), 2: P.zero()}).simplify()), str(QP({})))
        self.assertEqual(str(QP({1. - 1e-15: P.new([2, 4, 8])}) + QP({1. + 1e-15: P.new([3, 9])})),
                         str(QP({1.: P.new([5, 13, 8])})))
        self.assertNotEqual(
            str(QP({1 - Fraction(1, 10 ** 25): P.new([2, 4, 8]), 1 + Fraction(1, 10 ** 25): P.new([3, 9])})),
            str(QP({1: P.new([5, 13, 8])})))

    def test_new_integer(self):
        self.assertEqual(QP({0: P.new([2, 3, 4]), 1: P.new([1])}), QP.new_integer([[2, 3, 4], [1]]))
        self.assertIsInstance(QP.new_integer([[2]]).polynomial_dict[0].coefficients()[0], Fraction)
        self.assertIsInstance(QP.new_integer([[2.]]).polynomial_dict[0].coefficients()[0], float)
        self.assertIsInstance(QP.new_integer([[2.1j]]).polynomial_dict[0].coefficients()[0], complex)
        self.assertIsInstance(QP.new_integer([[a]]).polynomial_dict[0].coefficients()[0], sym.core.expr.Expr)

    def test_new(self):
        self.assertEqual(QP.new({0: [2, 3, 4], 1: [1]}), QP.new_integer([[2, 3, 4], [1]]))
        self.assertEqual(QP.new({0: [2, 3, 4], Fraction(1, 2): [1]}),
                         QP({0: P.new([2, 3, 4]), Fraction(1, 2): P.new([1])}))
        self.assertIsInstance(list(QP.new({Fraction(1, 2): [1]}).polynomial_dict.keys())[0], Fraction)
        self.assertIsInstance(list(QP.new({2: [1]}).polynomial_dict.keys())[0], int)
        self.assertIsInstance(list(QP.new({0.2: [1]}).polynomial_dict.keys())[0], float)
        self.assertIsInstance(list(QP.new({a: [1]}).polynomial_dict.keys())[0], sym.core.expr.Expr)

    def test_copy(self):
        temp = QP.new_integer([[2, 4, 8], [0, 0, 0], [3, 9]])
        temp_copy = temp.copy()
        self.assertNotEqual(id(temp), id(temp_copy))
        self.assertNotEqual(id(temp.polynomial_dict[0]), id(temp_copy.polynomial_dict[0]))
        self.assertEqual(temp, temp_copy)
        temp2 = QP.new_integer([[]])
        self.assertNotEqual(id(temp2), id(temp2.copy()))

    def test_eq(self):
        self.assertTrue(QP.new_integer([[2, 4, 8]]) == QP.new_integer([[2, 4, 8]]))
        self.assertFalse(QP.new_integer([[2, 4]]) == QP.new_integer([[2, 4, 8]]))
        self.assertTrue(QP.new_integer([]) == QP.new_integer([[]]))
        self.assertFalse(QP.new_integer([[2, 4], [3]]) == QP.new_integer([[2, 4]]))
        self.assertTrue(QP.new_integer([[2, 4], [3]]) == QP({1: P.new([3]), 0: P.new([2, 4])}))
        self.assertTrue(QP.new_integer([[2, 4], []]) == QP.new_integer([[2, 4]]))
        self.assertTrue(QP.new_integer([[0, 0], [2]]) == QP.new_integer([[], [2]]))
        self.assertFalse(QP.new_integer([[1]]) == QP.new_integer([[2]]))
        self.assertTrue(QP.new({Fraction(1, 3): [2, 4], Fraction(1, 2): [3]}) == QP.new(
            {Fraction(1, 3): [2, 4], Fraction(1, 2): [3]}))
        self.assertTrue(
            QP.new({Fraction(1, 3): [2, 4], Fraction(2, 1): [3]}) == QP.new({Fraction(1, 3): [2, 4], 2: [3]}))
        self.assertTrue(QP.new_integer([[], [2, 4, 8]]) == QP.new({Fraction(1, 1): [2, 4, 8]}))
        self.assertTrue(QP.new_integer([[], [2, 4, 8]]) == QP.new({1.: [2, 4, 8]}))
        self.assertFalse(QP.new_integer([[], [2, 4, 8]]) == QP.new_integer([[], [], [2, 4, 8]]))
        self.assertFalse(QP.new_integer([[], [2, 4, 8]]) == QP.new({1 + Fraction(1, 10 ** 15): [2, 4, 8]}))
        self.assertTrue(QP.new_integer([[], [2, 4, 8]]) == QP.new({1. + 1e-15: [2, 4, 8]}))
        self.assertTrue(QP.new_integer([[], [2, 4j - 4j + 2, 8]]) == QP.new({1. + 1e-15: [2, 2, 8]}))
        self.assertTrue(QP.new({1: [a, 2], a**2 + 1: [3]}) == QP.new({a**2 + 1: [3], 1: [a, 2]}))

    def test_pretty_print(self):
        self.assertEqual(QP.new_integer([]).pretty_print(), '0')
        self.assertEqual(QP.new_integer([[0]]).pretty_print(), '0')
        self.assertEqual(QP.new_integer([[2, 4, 8]]).pretty_print(), '2+4x+8x^2')
        self.assertEqual(QP.new_integer([[2, 4, 8], [1, 5, 25]]).pretty_print(), '2+4x+8x^2+(1+5x+25x^2)exp(-x)')
        self.assertEqual(QP.new_integer([[2, 4, 8], [1, 5, 25], [3, 9]]).pretty_print(),
                         '2+4x+8x^2+(1+5x+25x^2)exp(-x)+(3+9x)exp(-2x)')
        self.assertEqual(QP.new_integer([[2], [3], [4]]).pretty_print(), '2+3exp(-x)+4exp(-2x)')
        self.assertEqual(QP.new_integer([[0], [3], [4]]).pretty_print(), '3exp(-x)+4exp(-2x)')
        self.assertEqual(QP.new_integer([[2], [0], [4]]).pretty_print(), '2+4exp(-2x)')
        self.assertEqual(QP.new_integer([[2], [3], [0]]).pretty_print(), '2+3exp(-x)')
        self.assertEqual(QP.new_integer([[], [3], [4]]).pretty_print(), '3exp(-x)+4exp(-2x)')
        self.assertEqual(QP.new_integer([[2], [], [4]]).pretty_print(), '2+4exp(-2x)')
        self.assertEqual(QP.new_integer([[2], [3], []]).pretty_print(), '2+3exp(-x)')
        self.assertEqual(QP.new_integer([[2], [1], [4]]).pretty_print(), '2+exp(-x)+4exp(-2x)')
        self.assertEqual(QP.new_integer([[2], [3], [1]]).pretty_print(), '2+3exp(-x)+exp(-2x)')
        self.assertEqual(QP.new_integer([[-2], [-3]]).pretty_print(), '-2-3exp(-x)')
        self.assertEqual(QP.new_integer([[-2], [-3], [-4]]).pretty_print(), '-2-3exp(-x)-4exp(-2x)')
        self.assertEqual(QP.new({0: [2], Fraction(1, 2): [4]}).pretty_print(), '2+4exp(-1/2x)')
        self.assertEqual(QP.new({0: [2.], 1. + 1e-15: [4]}).pretty_print(), '2.0+4exp(-x)')
        self.assertEqual(QP.new({0: [2.], 1.: [4j + 1]}).pretty_print(), '2.0+(1+4j)exp(-x)')
        self.assertEqual(QP.new({0: [2 + 2*a], a: [a**3]}).pretty_print(), '(2*a + 2)+a^3exp(-ax)')
        self.assertEqual(QP.new({0: [1 + a - a], 1 + a: [a**3]}).pretty_print(), '1+a^3exp(-(a + 1)x)')

    def test_scalar_multiplication(self):
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]),
                         QP.new_integer([[1, 2, 4], [1, 5, 25]]).scalar_multiplication(2))
        self.assertEqual(QP.new({0: [4], Fraction(1, 2): [8]}),
                         QP.new({0: [2], Fraction(1, 2): [4]}).scalar_multiplication(2))
        self.assertEqual(QP.new_integer([]), QP.new_integer([[2, 4, 8], [1, 5, 25]]).scalar_multiplication(0))
        self.assertEqual(QP.new_integer([]), QP.new_integer([]).scalar_multiplication(2))
        temp = QP.new_integer([[2, 4, 8], [0, 0, 0], [3, 9]])
        temp_times_1 = 1 * temp
        self.assertNotEqual(id(temp), id(temp_times_1))
        self.assertNotEqual(id(temp.polynomial_dict[0]), id(temp_times_1.polynomial_dict[0]))
        self.assertEqual(QP.new({0: [4*a], Fraction(1, 2): [8*a]}),
                         QP.new({0: [2], Fraction(1, 2): [4]}).scalar_multiplication(2*a))

    def test_negation(self):
        self.assertEqual(-QP.new_integer([[2, 4, 8], [2, 10, 50]]), QP.new_integer([[-2, -4, -8], [-2, -10, -50]]))

    def test_add(self):
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) + QP.new_integer([[2, 4, 8], [2, 10, 50]]),
                         QP.new_integer([[4, 8, 16], [4, 20, 100]]))
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) + QP({1: P.new([2, 10, 50]), 0: P.new([2, 4, 8])}),
                         QP.new_integer([[4, 8, 16], [4, 20, 100]]))
        self.assertEqual(QP.new_integer([[2, 4, 8]]) + QP.new_integer([[2, 4, 8], [2, 10, 50]]),
                         QP.new_integer([[4, 8, 16], [2, 10, 50]]))
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) + QP.new_integer([[2, 4, 8]]),
                         QP.new_integer([[4, 8, 16], [2, 10, 50]]))
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) + QP.zero(), QP.new_integer([[2, 4, 8], [2, 10, 50]]))
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) + QP.new_integer([]),
                         QP.new_integer([[2, 4, 8], [2, 10, 50]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) + QP.new_integer([[-1, -2], [-3, -4]]), QP.new_integer([]))
        self.assertEqual(QP.new({0: [2, 4, 8], Fraction(1, 2): [2, 10, 50]}) + QP.new_integer([[2, 4, 8], [2, 10, 50]]),
                         QP.new({0: [4, 8, 16], Fraction(1, 2): [2, 10, 50], 1: [2, 10, 50]}))
        self.assertIsInstance((QP.zero() + QP.new_integer([[2]])).polynomial_dict[0].coefficients()[0], Fraction)
        self.assertIsInstance((QP.zero() + QP.new_integer([[2.]])).polynomial_dict[0].coefficients()[0], float)
        self.assertIsInstance(list((QP.zero() + QP.new({0.2: [1]})).polynomial_dict.keys())[0], float)
        self.assertEqual(
            QP.new({1. + 1e-15: [2, 4, 8], 1 / 2: [2, 10, 50]}) + QP.new({1: [2, 4, 8], Fraction(1, 2): [2, 10, 50]}),
            QP.new({1.: [4, 8, 16], Fraction(1, 2): [4, 20, 100]}))
        self.assertEqual(QP.new({a: [2, 4, 8]}) + QP.new({1: [2, 4, 8]}), QP.new({a: [2, 4, 8], 1: [2, 4, 8]}))
        self.assertEqual(QP.new({a - a: [2, 4, 8]}) + QP.new({0: [2, 4, 8]}), QP.new({0: [4, 8, 16]}))

    def test_sub(self):
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) - QP.new_integer([[1, 2, 4], [1, 5, 25]]),
                         QP.new_integer([[1, 2, 4], [1, 5, 25]]))
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) - QP.new_integer([[2, 4, 8], [2, 10, 50]]),
                         QP.new_integer([]))
        self.assertEqual(QP.new_integer([[2, 4, 8]]) - QP.new_integer([[2, 4, 8], [2, 10, 50]]),
                         QP.new_integer([[], [-2, -10, -50]]))
        self.assertEqual(QP.new_integer([[2, 4, 8], [2, 10, 50]]) - QP.new_integer([[2, 4, 8]]),
                         QP.new_integer([[], [2, 10, 50]]))

    def test_multiplication(self):
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * QP.new_integer([[5, 6], [7, 8]]),
                         QP.new_integer([[5, 16, 12], [22, 60, 40], [21, 52, 32]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * QP.new_integer([[5, 6]]),
                         QP.new_integer([[5, 16, 12], [15, 38, 24]]))
        self.assertEqual(QP.new_integer([[5, 6]]) * QP.new_integer([[1, 2], [3, 4]]),
                         QP.new_integer([[5, 16, 12], [15, 38, 24]]))
        self.assertEqual(QP.new_integer([[1]]) * QP.new_integer([[1, 2], [3, 4]]), QP.new_integer([[1, 2], [3, 4]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * QP.new_integer([[1]]), QP.new_integer([[1, 2], [3, 4]]))
        self.assertEqual(QP.new_integer([[]]) * QP.new_integer([[1, 2], [3, 4]]), QP.new_integer([[]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * QP.new_integer([[]]), QP.new_integer([[]]))
        self.assertEqual(QP.new_integer([[1, 2]]) * QP.new_integer([[]]), QP.new_integer([[]]))
        self.assertEqual(QP.new_integer([[]]) * QP.new_integer([[]]), QP.new_integer([[]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * P.new([5, 6]), QP.new_integer([[5, 16, 12], [15, 38, 24]]))
        self.assertEqual(P.new([5, 6]) * QP.new_integer([[1, 2], [3, 4]]), QP.new_integer([[5, 16, 12], [15, 38, 24]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * 2, QP.new_integer([[2, 4], [6, 8]]))
        self.assertEqual(2 * QP.new_integer([[1, 2], [3, 4]]), QP.new_integer([[2, 4], [6, 8]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * QP.new({Fraction(1, 2): [5, 6]}),
                         QP.new({Fraction(1, 2): [5, 16, 12], Fraction(3, 2): [15, 38, 24]}))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4]]) * QP.new_integer([[5, 1j]]),
                         QP.new_integer([[5, 10 + 1j, 2j], [15, 20 + 3j, 4j]]))
        self.assertEqual(QP.new({1 + a: [1]}) * QP.new({a: [2]}), QP.new({1 + 2*a: [2]}))
        self.assertEqual(QP.new({1: [a]}) * QP.new({1: [a]}), QP.new({2: [a**2]}))

    def test_integrate(self):
        self.assertEqual(QP.new_integer([[5, 5, 7]]).integrate(),
                         QP.new_integer([[0, 5, Fraction(5, 2), Fraction(7, 3)]]))
        self.assertEqual(QP.new_integer([[1, 2], [3, 4], [5, 6]]).integrate(),
                         QP.new_integer([[11, 1, 1], [-7, -4], [-4, -3]]))
        self.assertEqual(QP.new_integer([[], [], [1, 2, 0, 0, 5]]).integrate(), QP.new_integer(
            [[Fraction(19, 4)], [], [Fraction(-19, 4), Fraction(-17, 2), Fraction(-15, 2), -5, Fraction(-5, 2)]]))
        self.assertEqual(QP.new_integer([[]]).integrate(), QP.new_integer([[]]))
        self.assertEqual(QP.new({Fraction(1, 5): [1, 2], Fraction(1, 3): [3, 4], Fraction(1, 2): [5, 6]}).integrate(),
                         QP.new({0: [134], Fraction(1, 5): [-55, -10], Fraction(1, 3): [-45, -12],
                                 Fraction(1, 2): [-34, -12]}))
        self.assertEqual(QP.new({2.: [1, 2, 0, 0, 5]}).integrate(), QP.new_integer(
            [[Fraction(19, 4)], [], [Fraction(-19, 4), Fraction(-17, 2), Fraction(-15, 2), -5, Fraction(-5, 2)]]))
        self.assertEqual(QP.new({1e-15: [1], 2.: [1, 2, 0, 0, 5]}).integrate(), QP.new_integer(
            [[Fraction(19, 4), 1], [], [Fraction(-19, 4), Fraction(-17, 2), Fraction(-15, 2), -5, Fraction(-5, 2)]]))
        self.assertEqual(QP.new({a: [1, 2, 0, 0, 5]}).integrate(), QP.new({0: [1 / a + 2 / a ** 2 + 120 / a ** 5],
                                                                           a: [-1 / a - 2 / a ** 2 - 120 / a ** 5,
                                                                               -2 / a - 120 / a ** 4, -60 / a ** 3,
                                                                               -20 / a ** 2, -5 / a]}))
        self.assertEqual(QP.new({2: [a, a ** 3]}).integrate(),
                         QP.new({0: [a ** 3 / 4 + a / 2], 2: [-a ** 3 / 4 - a / 2, -a ** 3 / 2]}))

    def test_get_constant(self):
        self.assertEqual(QP.new_integer([[5, 5, 7]]).get_constant(), Fraction(5))
        self.assertEqual(QP.zero().get_constant(), Fraction(0))
        self.assertEqual(QP.new_integer([[], [], [1, 2, 0, 0, 5]]).get_constant(), Fraction(0))
        self.assertEqual(QP.new({Fraction(1, 2): [1]}).get_constant(), Fraction(0))
        self.assertEqual(QP.new_integer([[5j + 2, 5, 7]]).get_constant(), 5j + 2)
        self.assertEqual(QP.new({a - a: [2]}).get_constant(), 2)
        self.assertEqual(QP.new({0: [a**2]}).get_constant(), a**2)


if __name__ == '__main__':
    unittest.main()
