from unittest import TestCase

import numpy as np
from quasiPolynomial import Polynomial, QuasiPolynomial


class TestMonomial(TestCase):
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

class TestQuasiPolynomial(TestCase):
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
