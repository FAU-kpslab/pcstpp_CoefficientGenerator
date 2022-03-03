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
            return '0'
        elif len(self.coefficients) == 1:
            return str(self.coefficients[0])
        else:
            output.append(str(self.coefficients[0]))
            output.append(str(self.coefficients[1]) + 'x')
            for coefficient, exponent in list(enumerate(self.coefficients))[2:]:
                output.append(str(exponent) + 'x^' + str(coefficient))
            return '+'.join(output)

    # TODO: Define multiplication with a scalar.

    # TODO: Define negation of a monomial.

    # TODO: Define addition of two monomials.

    # TODO: Define multiplication of two monomials.


class QuasiPolynomial:
    """Data type of quasi-polynomial."""
    def __init__(self, underlying_array: np.ndarray) -> None:
        """A quasi-polynomial consists of an array of monomials."""
        self.quasi_polynomial = underlying_array

    def __str__(self) -> str:
        return str(self.quasi_polynomial)
        # TODO: What is "Convert method to property?"

    def pretty_print(self) -> str:
        """The quasi-polynomials can be printed in their mathematical form."""
        return str(self.quasi_polynomial[0][0])
        # TODO: Define this function.

    # TODO: Define multiplication with a scalar.

    # TODO: Define negation of a quasi-polynomial.

    # TODO: Define addition of two quasi-polynomials.

    # TODO: Define multiplication of two quasi-polynomials.


def test_main():
    qp = QuasiPolynomial(np.array([[2]]))
    print(qp)
    print(qp.pretty_print())

