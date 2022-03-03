from unittest import TestCase

import numpy as np
from quasiPolynomial import Monomial, QuasiPolynomial


class TestMonomial(TestCase):
    def test_pretty_print(self):
        self.assertEqual(Monomial(np.array([])).pretty_print(), '0')
        self.assertEqual(Monomial(np.array([0])).pretty_print(), '0')
        self.assertEqual(Monomial(np.array([2])).pretty_print(), '2')
        self.assertEqual(Monomial(np.array([2, 4])).pretty_print(), '2+4x')
        self.assertEqual(Monomial(np.array([2, 4, 8])).pretty_print(), '2+4x+8x^2')
        # TODO: Add test for passing a quasi-polynomial that is also a monomial.
        # TODO: Add test for passing a quasi-polynomial that is not a monomial.
