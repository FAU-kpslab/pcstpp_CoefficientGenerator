from unittest import TestCase

import numpy as np
from quasiPolynomial import Polynomial, QuasiPolynomial


class TestPolynomial(TestCase):

    def test_pretty_print(self):
        self.assertEqual(Polynomial([]).pretty_print(), '0')
        self.assertEqual(Polynomial([0]).pretty_print(), '0')
        self.assertEqual(Polynomial([2]).pretty_print(), '2')
        self.assertEqual(Polynomial([-2]).pretty_print(), '-2')
        self.assertEqual(Polynomial([2, 4]).pretty_print(), '2+4x')
        self.assertEqual(Polynomial([-2, -4]).pretty_print(), '-2-4x')
        self.assertEqual(Polynomial([0, 4]).pretty_print(), '4x')
        self.assertEqual(Polynomial([2, 4, 8]).pretty_print(), '2+4x+8x^2')
        self.assertEqual(Polynomial([0, 4, 8]).pretty_print(), '4x+8x^2')
        self.assertEqual(Polynomial([2, 4, 0, 100]).pretty_print(), '2+4x+100x^3')

    def test_simplify(self):
        self.assertEqual(print(Polynomial([2, 4, 8, 0]).simplify()), print(Polynomial([2, 4, 8])))
        self.assertEqual(print(Polynomial([2, 4, 0, 0]).simplify()), print(Polynomial([2, 4])))
        self.assertEqual(print(Polynomial([0, 0, 0]).simplify()), print(Polynomial([])))
        self.assertEqual(print(Polynomial([]).simplify()), print(Polynomial([])))


class TestQuasiPolynomial(TestCase):
    def test_init(self):
        self.assertEqual(print(QuasiPolynomial([])), print(QuasiPolynomial([[]])))

    def test_pretty_print(self):
        self.assertEqual(QuasiPolynomial([[]]).pretty_print(), '0')
        self.assertEqual(QuasiPolynomial([[0]]).pretty_print(), '0')
        self.assertEqual(QuasiPolynomial([[2, 4, 8]]).pretty_print(), '2+4x+8x^2')
        self.assertEqual(
            QuasiPolynomial([[2, 4, 8], [1, 5, 25]]).pretty_print(), '2+4x+8x^2+(1+5x+25x^2)exp(-x)')
        self.assertEqual(QuasiPolynomial([[2, 4, 8], [1, 5, 25], [3, 9]]).pretty_print(),
                         '2+4x+8x^2+(1+5x+25x^2)exp(-x)+(3+9x)exp(-2x)')
        self.assertEqual(QuasiPolynomial([[2], [3], [4]]).pretty_print(), '2+3exp(-x)+4exp(-2x)')
        self.assertEqual(QuasiPolynomial([[0], [3], [4]]).pretty_print(), '3exp(-x)+4exp(-2x)')
        self.assertEqual(QuasiPolynomial([[2], [0], [4]]).pretty_print(), '2+4exp(-2x)')
        self.assertEqual(QuasiPolynomial([[2], [3], [0]]).pretty_print(), '2+3exp(-x)')
        self.assertEqual(QuasiPolynomial([[], [3], [4]]).pretty_print(), '3exp(-x)+4exp(-2x)')
        self.assertEqual(QuasiPolynomial([[2], [], [4]]).pretty_print(), '2+4exp(-2x)')
        self.assertEqual(QuasiPolynomial([[2], [3], []]).pretty_print(), '2+3exp(-x)')
        self.assertEqual(QuasiPolynomial([[2], [1], [4]]).pretty_print(), '2+exp(-x)+4exp(-2x)')
        self.assertEqual(QuasiPolynomial([[2], [3], [1]]).pretty_print(), '2+3exp(-x)+exp(-2x)')
        self.assertEqual(QuasiPolynomial([[-2], [-3]]).pretty_print(), '-2-3exp(-x)')
        self.assertEqual(QuasiPolynomial([[-2], [-3], [-4]]).pretty_print(), '-2-3exp(-x)-4exp(-2x)')

    def test_simplify(self):
        self.assertEqual(print(QuasiPolynomial([[2, 4, 8, 0]]).simplify()), print(QuasiPolynomial([[2, 4, 8]])))
        self.assertEqual(print(QuasiPolynomial([[2, 4, 0, 0]]).simplify()), print(QuasiPolynomial([[2, 4]])))
        self.assertEqual(print(QuasiPolynomial([[0, 0, 0]]).simplify()), print(QuasiPolynomial([])))
        self.assertEqual(print(QuasiPolynomial([]).simplify()), print(QuasiPolynomial([])))
        self.assertEqual(print(QuasiPolynomial([[2, 4, 8], [1, 5, 25], [0, 0]]).simplify()),
                         print(QuasiPolynomial([[2, 4, 8], [1, 5, 25]])))
        self.assertEqual(print(QuasiPolynomial([[2, 4, 8], [0, 0, 0], [3, 9]]).simplify()),
                         print(QuasiPolynomial([[2, 4, 8], [], [3, 9]])))