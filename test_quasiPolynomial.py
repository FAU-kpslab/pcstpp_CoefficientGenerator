from unittest import TestCase

import numpy as np
from quasiPolynomial import Polynomial, QuasiPolynomial


class TestMonomial(TestCase):
    def test_pretty_print(self):
        self.assertEqual(Polynomial(np.array([])).pretty_print(), '0')
        self.assertEqual(Polynomial(np.array([0])).pretty_print(), '0')
        self.assertEqual(Polynomial(np.array([2])).pretty_print(), '2')
        self.assertEqual(Polynomial(np.array([-2])).pretty_print(), '-2')
        self.assertEqual(Polynomial(np.array([2, 4])).pretty_print(), '2+4x')
        self.assertEqual(Polynomial(np.array([-2, -4])).pretty_print(), '-2-4x')
        self.assertEqual(Polynomial(np.array([0, 4])).pretty_print(), '4x')
        self.assertEqual(Polynomial(np.array([2, 4, 8])).pretty_print(), '2+4x+8x^2')
        self.assertEqual(Polynomial(np.array([0, 4, 8])).pretty_print(), '4x+8x^2')
        self.assertEqual(Polynomial(np.array([2, 4, 0, 100])).pretty_print(), '2+4x+100x^3')

class TestQuasiPolynomial(TestCase):
    def test_pretty_print(self):
        self.assertEqual(QuasiPolynomial(np.array([])).pretty_print(), '0')
        self.assertEqual(QuasiPolynomial(np.array([Polynomial(np.array([]))])).pretty_print(), '0')
        self.assertEqual(QuasiPolynomial(np.array([Polynomial(np.array([0]))])).pretty_print(), '0')
        self.assertEqual(QuasiPolynomial(np.array([Polynomial(np.array([2, 4, 8]))])).pretty_print(), '2+4x+8x^2')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2, 4, 8])), Polynomial(np.array([1, 5, 25]))])).pretty_print(),
            '2+4x+8x^2+(1+5x+25x^2)exp(-x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2, 4, 8])), Polynomial(np.array([1, 5, 25])),
                                      Polynomial(np.array([3, 9]))])).pretty_print(),
            '2+4x+8x^2+(1+5x+25x^2)exp(-x)+(3+9x)exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([3])),
                                      Polynomial(np.array([4]))])).pretty_print(),
            '2+3exp(-x)+4exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([0])), Polynomial(np.array([3])),
                                      Polynomial(np.array([4]))])).pretty_print(),
            '3exp(-x)+4exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([0])),
                                      Polynomial(np.array([4]))])).pretty_print(),
            '2+4exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([3])),
                                      Polynomial(np.array([0]))])).pretty_print(),
            '2+3exp(-x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([])), Polynomial(np.array([3])),
                                      Polynomial(np.array([4]))])).pretty_print(),
            '3exp(-x)+4exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([])),
                                      Polynomial(np.array([4]))])).pretty_print(),
            '2+4exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([3])),
                                      Polynomial(np.array([]))])).pretty_print(),
            '2+3exp(-x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([1])),
                                      Polynomial(np.array([4]))])).pretty_print(),
            '2+exp(-x)+4exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([2])), Polynomial(np.array([3])),
                                      Polynomial(np.array([1]))])).pretty_print(),
            '2+3exp(-x)+exp(-2x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([-2])), Polynomial(np.array([-3]))])).pretty_print(),
            '-2-3exp(-x)')
        self.assertEqual(
            QuasiPolynomial(np.array([Polynomial(np.array([-2])), Polynomial(np.array([-3])),
                                      Polynomial(np.array([-4]))])).pretty_print(),
            '-2-3exp(-x)-4exp(-2x)')
