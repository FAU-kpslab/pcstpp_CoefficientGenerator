from fractions import Fraction
from typing import List, Union

import numpy as np


class Polynomial:
    """
    Polynomial(coefficient_list)

    A class used to represent a polynomial.

        Parameters
        ----------
        coefficient_list : List[Fraction]
            The list of coefficients.
            The coefficient of x^n is coefficient_array[n].

        Attributes
        ----------
        coefficients : np.ndarray[Fraction]
            The numpy array of coefficients.
            The coefficient of x^n is coefficients[n].

        Methods
        -------
        zero : Polynomial
            Creates an empty polynomial.
        simplify : Polynomial
            Simplifies a polynomial by *removing* zeros.
        new : Polynomial
            Creates a quasi-polynomial using a list of coefficients.
        copy : Polynomial
            Copies a polynomial.
        pretty_print : str
            Transform a polynomial in the mathematical form suitable to be read by humans.
        __eq__ : bool
            Checks whether two polynomials are mathematically equal.
        scalar_multiplication : Polynomial
            Multiplies a polynomial with a scalar.
        __neg__ : Polynomial
            Multiplies a polynomial with -1.
        __add__ : Polynomial
            Adds two polynomials.
        __mul__ : Polynomial
            Multiplies two polynomials or a polynomial with a scalar.
        __rmul__ : Polynomial
            Multiplies a scalar with a polynomial.
    """

    def __init__(self, coefficient_list: Union[List[Fraction]]) -> None:
        """
            Parameters
            ----------
            coefficient_list : List[int]
                The array of coefficients.
                The coefficient of x^n is coefficient_list[n].
        """

        self.coefficients = np.asarray(coefficient_list).astype(Fraction, copy=False)

    @staticmethod
    def zero() -> 'Polynomial':
        """
        Polynomial.zero()

        Creates an empty polynomial.

            Returns
            -------
            Polynomial
        """
        return Polynomial([])

    def simplify(self) -> 'Polynomial':
        """
        p.simplify()

        Simplifies a polynomial by *removing* zeros.

            Returns
            -------
            Polynomial
        """

        # Check whether the polynomial is empty.
        if self.coefficients.size != 0:
            # Check whether the last coefficient is zero to remove it.
            while self.coefficients[-1] == 0:
                self.coefficients = self.coefficients[:-1].copy()
                # Check whether the polynomial is empty.
                if self.coefficients.size == 0:
                    break
        return self

    @staticmethod
    def new(coefficient_list: Union[List[Fraction], List[int], List[float]]) -> 'Polynomial':
        """
        new(List[Number])

        Creates a quasi-polynomial using a list of coefficients.

            Parameters
            ----------
            coefficient_list

            Returns
            -------
            Polynomial
        """

        return Polynomial([Fraction(coeff) for coeff in coefficient_list]).simplify()

    def copy(self) -> 'Polynomial':
        """
        copy(p)

        Copies a polynomial.

            Returns
            -------
            Polynomial
        """

        return Polynomial(self.coefficients.copy())

    def __str__(self) -> str:
        return str(self.coefficients)

    def pretty_print(self) -> str:
        """
        p.pretty_print()

        Transform a polynomial in the mathematical form suitable to be read by humans.

            Returns
            -------
            str
        """

        # Check whether the polynomial is empty.
        if self.coefficients.size == 0:
            return '0'
        # Check whether the polynomial contains only the constant term.
        elif self.coefficients.size == 1:
            return str(self.coefficients[0])
        else:
            output = []
            # Check whether the constant term is zero to leave that away.
            if self.coefficients[0] != 0:
                output.append(str(self.coefficients[0]))
            if self.coefficients[1] != 0:
                output.append(str(self.coefficients[1]) + 'x')
            for exponent, coefficient in list(enumerate(self.coefficients))[2:]:
                # Check for the remaining coefficients whether they are zero to leave those away.
                if coefficient != 0:
                    output.append(str(coefficient) + 'x^' + str(exponent))
            return '+'.join(output).replace('+-', '-')

    def __eq__(self, other: 'Polynomial') -> bool:
        """
        p1 == p2

        Checks whether two polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().coefficients, other.simplify().coefficients)

    def scalar_multiplication(self, scalar: Union[Fraction, int, float]) -> 'Polynomial':
        """
        p.scalar_multiplication(int)

        Multiplies a polynomial with a scalar.

            Parameters
            ----------
            scalar

            Returns
            -------
            Polynomial
        """

        return Polynomial(scalar * self.coefficients).simplify()

    def __neg__(self) -> 'Polynomial':
        """
        p.negation()

        Multiplies a polynomial with -1.

            Returns
            -------
            Polynomial
        """

        return self.scalar_multiplication(-1)

    def __add__(self, other: 'Polynomial') -> 'Polynomial':
        """
        p1 + p2

        Adds two polynomials.

            Returns
            -------
            Polynomial
        """

        left_size = self.coefficients.size
        right_size = other.coefficients.size
        if left_size > right_size:
            # Add zeros to the shorter polynomial to make them equally long.
            output = np.concatenate((other.coefficients, np.zeros(left_size - right_size, dtype=Fraction)))
            return Polynomial(list(output + self.coefficients)).simplify()
        else:
            # Add zeros to the shorter polynomial to make them equally long.
            output = np.concatenate((self.coefficients, np.zeros(right_size - left_size, dtype=Fraction)))
            return Polynomial(list(output + other.coefficients)).simplify()

    def __mul__(self, other: Union['Polynomial', Fraction, int, float]) -> 'Polynomial':
        """
        p1 * p2 | p * scalar

        Multiplies two polynomials or a polynomial with a scalar.

            Returns
            -------
            Polynomial
        """

        # Check whether the second object is a polynomial.
        if isinstance(other, Polynomial):
            # Calculate the matrix containing all combinations of coefficients of both polynomials.
            # Flip it such that all coefficients corresponding to the same x^n are part of the same diagonals.
            combinations = np.flipud(np.outer(self.coefficients, other.coefficients))
            # Sum over the diagonals to obtain the real coefficients.
            output = [sum(combinations.diagonal(exponent), start=0) for exponent in
                      np.arange(- self.coefficients.size + 1, other.coefficients.size)]
            return Polynomial(output).simplify()
        # Check whether the second object is a scalar and call scalar_multiplication.
        elif isinstance(other, (Fraction, int, float)):
            return self.scalar_multiplication(other)
        # If the second polynomial is not a polynomial (but e.g. a quasi-polynomial) return NotImplemented to trigger
        # the function __rmul__ of the other class.
        else:
            return NotImplemented

    def __rmul__(self, other: Union[Fraction, int, float]) -> 'Polynomial':
        """
        scalar * p

        Multiplies a scalar with a polynomial.

            Returns
            -------
            Polynomial
        """

        return self * other


class QuasiPolynomial:
    """
    QuasiPolynomial(coefficient_list)

    A class used to represent a quasi-polynomial.

        Parameters
        ----------
        polynomial_list : List[Polynomial]
                The list of polynomials.
                The coefficient polynomial of exp(-nx) is polynomial_list[n].

        Attributes
        ----------
        polynomials : np.ndarray[Polynomial]
            The array of polynomials.
            The coefficient polynomial of exp(-nx) is polynomials[n].

        Methods
        -------
        simplify : QuasiPolynomial
            Simplifies a quasi-polynomial by *removing* zero polynomials.
        new : QuasiPolynomial
            Creates a quasi-polynomial using a nested list of coefficients.
        copy : QuasiPolynomial
            Copies a quasi-polynomial.
        pretty_print : str
            Transform a quasi-polynomial in the mathematical form suitable to be read by humans.
        __eq__ : bool
            Checks whether two quasi-polynomials are mathematically equal.
        scalar_multiplication
            Multiplies a quasi-polynomial with a scalar.
        __neg__ : QuasiPolynomial
            Multiplies a quasi-polynomial with -1.
        __add__ : QuasiPolynomial
            Adds two quasi-polynomials.
        __sub__ : QuasiPolynomial
            Subtracts a quasi-polynomial from another.
        __mul__ : QuasiPolynomial
            Multiplies two quasi-polynomials, a quasi-polynomial with a polynomial or a quasi-polynomial with a scalar.
        __rmul__ : QuasiPolynomial
            Multiplies a polynomial with a quasi-polynomial or a scalar with a quasi-polynomial.
    """

    def __init__(self, polynomial_list: List[Polynomial]) -> None:
        """
            Parameters
            ----------
            polynomial_list : List[Polynomial]
                The list of polynomials.
                The coefficient polynomial of exp(-nx) is polynomial_list[n].
        """

        self.polynomials = np.asarray(polynomial_list, dtype=Polynomial)

    def simplify(self) -> 'QuasiPolynomial':
        """
        qp.simplify()

        Simplifies a quasi-polynomial by *removing* zero polynomials.

            Returns
            -------
            QuasiPolynomial
        """

        # Simplify the remaining polynomials.
        for polynomial in self.polynomials:
            polynomial.simplify()
            # Check whether the quasi-polynomial is empty.
        if self.polynomials.size == 0:
            return QuasiPolynomial([])
        else:
            # Check whether the last polynomial is empty to remove it.
            while self.polynomials[-1].coefficients.size == 0:
                self.polynomials = self.polynomials[:-1].copy()
                # Recheck whether the quasi-polynomial is empty.
                if self.polynomials.size == 0:
                    return QuasiPolynomial([])
        return self

    @staticmethod
    def new(coefficient_list: Union[List[List[Fraction]], List[List[int]], List[List[float]]]) -> 'QuasiPolynomial':
        """
        new(List[List[scalar]])

        Creates a quasi-polynomial using a nested list of coefficients.

            Parameters
            ----------
            coefficient_list

            Returns
            -------
            QuasiPolynomial
        """

        polynomial_list = [Polynomial.new(coeffs) for coeffs in coefficient_list]
        return QuasiPolynomial(polynomial_list).simplify()

    def copy(self) -> 'QuasiPolynomial':
        """
        copy(p)

        Copies a quasi-polynomial.

            Returns
            -------
            quasi-Polynomial
        """

        return QuasiPolynomial([p.copy().coefficients for p in self.polynomials])

    def __str__(self) -> str:
        return str([p.coefficients.tolist() for p in self.polynomials])

    def pretty_print(self) -> str:
        """
        qp.pretty_print()

        Transform a quasi-polynomial in the mathematical form suitable to be read by humans.

            Returns
            -------
            str
        """

        # Check whether the quasi-polynomial is empty.
        if self.polynomials.size == 0:
            return '0'
        # Check whether the quasi-polynomial contains only the first polynomial.
        if self.polynomials.size == 1:
            return self.polynomials[0].pretty_print()
        else:
            output = []
            # Check whether the first polynomial is zero to leave those away.
            if self.polynomials[0].pretty_print() != '0':
                output.append(self.polynomials[0].pretty_print())
            # Check whether the second polynomial is zero to leave those away.
            if self.polynomials[1].pretty_print() != '0':
                # Check whether the polynomial contains only the constant term to leave away the brackets.
                if self.polynomials[1].coefficients.size == 1:
                    # Check whether the polynomial contains only 1 to leave that away.
                    if self.polynomials[1].coefficients == 1:
                        output.append('exp(-x)')
                    else:
                        output.append(self.polynomials[1].pretty_print() + 'exp(-x)')
                else:
                    output.append('(' + self.polynomials[1].pretty_print() + ')exp(-x)')
            for exponent, polynomial in list(enumerate(self.polynomials))[2:]:
                # Check for the remaining polynomials whether they are zero to leave those away.
                if polynomial.pretty_print() != '0':
                    # Check for the remaining polynomials whether they contain only the constant term to leave away the
                    # brackets.
                    if polynomial.coefficients.size == 1:
                        # Check for the remaining polynomials whether they contain only 1 to leave that away.
                        if polynomial.coefficients == 1:
                            output.append('exp(-' + str(exponent) + 'x)')
                        else:
                            output.append(polynomial.pretty_print() + 'exp(-' + str(exponent) + 'x)')
                    else:
                        output.append('(' + polynomial.pretty_print() + ')exp(-' + str(exponent) + 'x)')
            return '+'.join(output).replace('+-', '-')

    def __eq__(self, other: 'QuasiPolynomial') -> bool:
        """
        qp1 == qp2

        Checks whether two quasi-polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().polynomials, other.simplify().polynomials)

    def scalar_multiplication(self, scalar: Union[Fraction, int, float]) -> 'QuasiPolynomial':
        """
        qp.scalar_multiplication(int)

        Multiplies a quasi-polynomial with a scalar.

            Parameters
            ----------
            scalar

            Returns
            -------
            QuasiPolynomial
        """

        return QuasiPolynomial(scalar * self.polynomials).simplify()

    def __neg__(self) -> 'QuasiPolynomial':
        """
        -qp

        Multiplies a quasi-polynomial with -1.

            Returns
            -------
            QuasiPolynomial
        """

        return self.scalar_multiplication(-1)

    def __add__(self, other: 'QuasiPolynomial') -> 'QuasiPolynomial':
        """
        qp1 + qp2

        Adds two quasi-polynomials.

            Returns
            -------
            QuasiPolynomial
        """

        left_size = self.polynomials.size
        right_size = other.polynomials.size
        if left_size > right_size:
            # Add empty polynomials to the shorter quasi-polynomial to make them equally long.
            new_other = np.concatenate(
                (other.polynomials,
                 np.array([Polynomial.zero()] * (left_size - right_size))))
            output = new_other + self.polynomials
        else:
            # Add empty polynomials to the shorter quasi-polynomial to make them equally long.
            new_self = np.concatenate(
                (self.polynomials,
                 np.array([Polynomial.zero()] * (right_size - left_size))))
            output = new_self + other.polynomials
        return QuasiPolynomial(output).simplify()

    def __sub__(self, other: 'QuasiPolynomial') -> 'QuasiPolynomial':
        """
        qp1 - qp2

        Subtracts a quasi-polynomial from another.

            Returns
            -------
            QuasiPolynomial
        """

        return self + (-other)

    def __mul__(self, other: Union['QuasiPolynomial', Polynomial, Fraction, int, float]) -> 'QuasiPolynomial':
        """
        qp1 * qp2 | qp * p | qp * scalar

        Multiplies two quasi-polynomials, a quasi-polynomial with a polynomial or a quasi-polynomial with a scalar.

            Returns
            -------
            QuasiPolynomial
        """

        # Check whether the second object is a quasi-polynomial.
        if isinstance(other, QuasiPolynomial):
            # Calculate the matrix containing all combinations of coefficient polynomials of both quasi-polynomials.
            # Flip it such that all coefficient polynomials corresponding to the same exp(-nx) are part of the same
            # diagonals.
            combinations = np.flipud(np.outer(self.polynomials, other.polynomials))
            # Sum over the diagonals to obtain the real coefficient polynomials.
            output = [sum(combinations.diagonal(exponent), start=Polynomial.zero()) for exponent in
                      np.arange(- self.polynomials.size + 1, other.polynomials.size)]
            return QuasiPolynomial(output).simplify()
        # Check whether the second object is a polynomial and lift it to a quasi-polynomial.
        elif isinstance(other, Polynomial):
            return self * QuasiPolynomial([other])
        # Check whether the second object is a scalar and call scalar_multiplication.
        if isinstance(other, (Fraction, int, float)):
            return self.scalar_multiplication(other)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[Polynomial, Fraction, int, float]) -> 'QuasiPolynomial':
        """
        p * qp | scalar * qp

        Multiplies a polynomial with a quasi-polynomial or a scalar with a quasi-polynomial.

            Returns
            -------
            QuasiPolynomial
        """

        return self * other


def test_main():
    print(Polynomial([Fraction(1, 2), 4]).pretty_print())
