import numpy as np


class Monomial:
    """Data type of monomial."""
    def __init__(self, coefficient_array: np.ndarray) -> None:
        """A quasi-polynomial consists of an array of coefficients."""
        self.coefficients = coefficient_array

    def __str__(self) -> str:
        return str(self.coefficients)

    def pretty_print(self) -> str:
        """The monomials can be printed in the mathematical form."""
        output = []
        if len(self.coefficients) == 0:
            # Check whether the monomial is empty.
            return '0'
        elif len(self.coefficients) == 1:
            # Check whether the monomial contains only the constant term.
            return str(self.coefficients[0])
        else:
            if self.coefficients[0] != 0:
                # Check whether the constant term is zero to leave that away.
                output.append(str(self.coefficients[0]))
            if self.coefficients[1] != 0:
                output.append(str(self.coefficients[1]) + 'x')
            for exponent, coefficient in list(enumerate(self.coefficients))[2:]:
                if coefficient != 0:
                    # Check for the remaining coefficients whether they are zero to leave those away.
                    output.append(str(coefficient) + 'x^' + str(exponent))
            return '+'.join(output).replace('+-', '-')

    # TODO: Define simplification.

    # TODO: Define multiplication with a scalar.

    # TODO: Define negation of a monomial.

    # TODO: Define addition of two monomials.

    # TODO: Define multiplication of two monomials.


class QuasiPolynomial:
    """Data type of quasi-polynomial."""
    def __init__(self, monomial_array: np.ndarray) -> None:
        """A quasi-polynomial consists of an array of monomials."""
        self.monomials = monomial_array

    def __str__(self) -> str:
        return str(self.monomials)
        # TODO: What is "Convert method to property?"

    def pretty_print(self) -> str:
        """The quasi-polynomials can be printed in their mathematical form."""
        output = []
        if len(self.monomials) == 0:
            # Check whether the quasi-polynomial is empty.
            return '0'
        if len(self.monomials) == 1:
            # Check whether the quasi-polynomial contains only the first monomial.
            return self.monomials[0].pretty_print()
        else:
            if self.monomials[0].pretty_print() != '0':
                # Check whether the first monomial is zero to leave those away.
                output.append(self.monomials[0].pretty_print())
            if self.monomials[1].pretty_print() != '0':
                # Check whether the second monomial is zero to leave those away.
                if len(self.monomials[1].coefficients) == 1:
                    # Check whether the monomial contains only the constant term to leave away the brackets.
                    if self.monomials[1].coefficients == 1:
                        # Check whether the monomial contains only 1 to leave that away.
                        output.append('exp(-x)')
                    else:
                        output.append(self.monomials[1].pretty_print() + 'exp(-x)')
                else:
                    output.append('(' + self.monomials[1].pretty_print() + ')exp(-x)')
            for exponent, monomial in list(enumerate(self.monomials))[2:]:
                if monomial.pretty_print() != '0':
                    # Check for the remaining monomials whether they are zero to leave those away.
                    if len(monomial.coefficients) == 1:
                        # Check for the remaining monomials whether they contain only the constant term to leave away
                        # the brackets.
                        if monomial.coefficients == 1:
                            # Check for the remaining monomials whether they contain only 1 to leave that away.
                            output.append('exp(-' + str(exponent) + 'x)')
                        else:
                            output.append(monomial.pretty_print() + 'exp(-' + str(exponent) + 'x)')
                    else:
                        output.append('(' + monomial.pretty_print() + ')exp(-' + str(exponent) + 'x)')
            return '+'.join(output).replace('+-', '-')

    # TODO: Define simplification.

    # TODO: Define multiplication with a scalar.

    # TODO: Define negation of a quasi-polynomial.

    # TODO: Define addition of two quasi-polynomials.

    # TODO: Define multiplication of two quasi-polynomials.


def test_main():
    qp = QuasiPolynomial(np.array([Monomial(np.array([2])), Monomial(np.array([])),
                                      Monomial(np.array([4]))]))
    print(qp.pretty_print())
    print(qp.monomials[1].coefficients)

