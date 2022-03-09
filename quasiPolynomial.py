import numpy as np
from typing import List


class Polynomial:
    """
    Polynomial(coefficient_list)

    A class used to represent a polynomial.

        Parameters
        ----------
        coefficient_list : List[int]
            The list of coefficients.
            The coefficient of x^n is coefficient_array[n].

        Attributes
        ----------
        coefficients : np.ndarray[int]
            The numpy array of coefficients.
            The coefficient of x^n is coefficients[n].

        Methods
        -------
        to_list : List[int]
            Transforms a polynomial into the integer list used to construct it.
        copy : Polynomial
            Copies a polynomial.
        pretty_print : str
            Transform a polynomial in the mathematical form suitable to be read by humans.
        simplify : Polynomial
            Simplifies a polynomial by *removing* zeros.
        __eq__ : bool
            Checks whether two polynomials are mathematically equal.
        scalar_multiplication : Polynomial
            Multiplies a polynomial with a scalar.
        __neg__ : Polynomial
            Multiplies a polynomial with -1.
        __add__ : Polynomial
            Adds two polynomials.
    """

    def __init__(self, coefficient_list: List[int]) -> None:
        """
            Parameters
            ----------
            coefficient_list : List[int]
                The array of coefficients.
                The coefficient of x^n is coefficient_list[n].
        """

        self.coefficients = np.asarray(coefficient_list).astype(int, copy=False)

    def to_list(self) -> List[int]:
        """
        p.to_list()

        Transforms a polynomial into the integer list used to construct it.

            Returns
            -------
            List[int]
        """

        return list(self.coefficients)

    def copy(self):
        """
        copy(p)

        Copies a polynomial.

            Returns
            -------
            Polynomial
        """

        return Polynomial(self.coefficients.copy())

    def __str__(self) -> str:
        return str(self.to_list())

    def pretty_print(self) -> str:
        """
        p.pretty_print()

        Transform a polynomial in the mathematical form suitable to be read by humans.

            Returns
            -------
            str
        """

        if len(self.coefficients) == 0:
            # Check whether the polynomial is empty.
            return '0'
        elif len(self.coefficients) == 1:
            # Check whether the polynomial contains only the constant term.
            return str(self.coefficients[0])
        else:
            output = []
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

    def simplify(self):
        """
        p.simplify()

        Simplifies a polynomial by *removing* zeros.

            Returns
            -------
            Polynomial
        """

        if self.coefficients.size != 0:
            # Check whether the polynomial is empty.
            while self.coefficients[-1] == 0:
                # Check whether the last coefficient is zero to remove it.
                self.coefficients.resize(self.coefficients.size - 1)
                if self.coefficients.size == 0:
                    # Check whether the polynomial is empty.
                    break
        return self

    def __eq__(self, other) -> bool:
        """
        p1 == p2

        Checks whether two polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().coefficients, other.simplify().coefficients)

    def scalar_multiplication(self, scalar: int):
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

    def __neg__(self):
        """
        p.negation()

        Multiplies a polynomial with -1.

            Returns
            -------
            Polynomial
        """

        return self.scalar_multiplication(-1)

    def __add__(self, other):
        """
        p1 + p2

        Adds two polynomials.

            Returns
            -------
            Polynomial
        """

        if len(self.coefficients) > len(other.coefficients):
            short = other.coefficients.copy()
            short.resize(len(self.coefficients))
            return Polynomial(list(short + self.coefficients)).simplify()
        else:
            short = self.coefficients.copy()
            short.resize(len(other.coefficients))
            return Polynomial(list(short + other.coefficients)).simplify()

    def __mul__(self, other):
        """
        p1 * p2

        Multiplies two polynomials.

            Returns
            -------
            Polynomial
        """

        length = self.coefficients.size + other.coefficients.size - 1
        coeffs = []
        for total in np.arange(length):
            mini = max(0, total - other.coefficients.size + 1)
            maxi = min(total, self.coefficients.size - 1)
            coeffs.append(
                sum([self.coefficients[exp1] * other.coefficients[- exp1 + total] for exp1 in
                     np.arange(mini, maxi + 1)]))
        return Polynomial(coeffs)
    # TODO: Do I need additional simplification beforehand?


class QuasiPolynomial:
    """
    QuasiPolynomial(coefficient_list)

    A class used to represent a quasi-polynomial.

        Parameters
        ----------
        coefficient_list : List[List[int]]
            The list of coefficients.
            The coefficient of x^m exp(-nx) is coefficient_list[n][m].

        Attributes
        ----------
        polynomials : np.ndarray[Polynomial]
            The array of polynomials.
            The polynomial in front of exp(-nx) is polynomials[n].

        Methods
        -------
        copy : QuasiPolynomial
            Copies a quasi-polynomial.
        pretty_print : str
            Transform a quasi-polynomial in the mathematical form suitable to be read by humans.
        simplify : QuasiPolynomial
            Simplifies a quasi-polynomial by *removing* zero polynomials.
        __eq__ : bool
            Checks whether two quasi-polynomials are mathematically equal.
        scalar_multiplication
            Multiplies a quasi-polynomial with a scalar.
        __neg__ : QuasiPolynomial
            Multiplies a quasi-polynomial with -1.
        __add__ : QuasiPolynomial
            Adds two quasi-polynomials.
        __sub__ : QuasiPolynomials
            Subtracts a quasi-polynomial from another.
    """

    def __init__(self, coefficient_list: List[List[int]]) -> None:
        """
            Parameters
            ----------
            coefficient_list : List[List[int]]
                The list of coefficients.
                The coefficient of x^m exp(-nx) is coefficient_list[n][m].
        """

        if len(coefficient_list) == 0:
            self.polynomials = np.asarray([])
        else:
            self.polynomials = np.asarray([Polynomial(p) for p in coefficient_list])

    def copy(self):
        """
        copy(p)

        Copies a quasi-polynomial.

            Returns
            -------
            quasi-Polynomial
        """

        return QuasiPolynomial([p.copy().coefficients.tolist() for p in self.polynomials])

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

        if self.polynomials.size == 0:
            # Check whether the quasi-polynomial is empty.
            return '0'
        if self.polynomials.size == 1:
            # Check whether the quasi-polynomial contains only the first polynomial.
            return self.polynomials[0].pretty_print()
        else:
            output = []
            if self.polynomials[0].pretty_print() != '0':
                # Check whether the first polynomial is zero to leave those away.
                output.append(self.polynomials[0].pretty_print())
            if self.polynomials[1].pretty_print() != '0':
                # Check whether the second polynomial is zero to leave those away.
                if len(self.polynomials[1].coefficients) == 1:
                    # Check whether the polynomial contains only the constant term to leave away the brackets.
                    if self.polynomials[1].coefficients == 1:
                        # Check whether the polynomial contains only 1 to leave that away.
                        output.append('exp(-x)')
                    else:
                        output.append(self.polynomials[1].pretty_print() + 'exp(-x)')
                else:
                    output.append('(' + self.polynomials[1].pretty_print() + ')exp(-x)')
            for exponent, polynomial in list(enumerate(self.polynomials))[2:]:
                if polynomial.pretty_print() != '0':
                    # Check for the remaining polynomials whether they are zero to leave those away.
                    if len(polynomial.coefficients) == 1:
                        # Check for the remaining polynomials whether they contain only the constant term to leave away
                        # the brackets.
                        if polynomial.coefficients == 1:
                            # Check for the remaining polynomials whether they contain only 1 to leave that away.
                            output.append('exp(-' + str(exponent) + 'x)')
                        else:
                            output.append(polynomial.pretty_print() + 'exp(-' + str(exponent) + 'x)')
                    else:
                        output.append('(' + polynomial.pretty_print() + ')exp(-' + str(exponent) + 'x)')
            return '+'.join(output).replace('+-', '-')

    def simplify(self):
        """
        qp.simplify()

        Simplifies a quasi-polynomial by *removing* zero polynomials.

            Returns
            -------
            QuasiPolynomial
        """

        if self.polynomials.size == 0:
            # Check whether the quasi-polynomial is the empty polynomial and replace it by the empty quasi-polynomial.
            return QuasiPolynomial([])
        else:
            while self.polynomials[-1] == Polynomial([]):
                # Check whether the last polynomial is empty to remove it.
                self.polynomials.resize(self.polynomials.size - 1)
                if self.polynomials.size == 0:
                    # Recheck whether the quasi-polynomial is empty.
                    return QuasiPolynomial([])
        for polynomial in self.polynomials:
            polynomial.simplify()
        return self

    def __eq__(self, other) -> bool:
        """
        qp1 == qp2

        Checks whether two quasi-polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().polynomials, other.simplify().polynomials)

    def scalar_multiplication(self, scalar: int):
        """
        qp.scalar_multiplication(int)

        Multiplies a quasi-polynomial with a scalar.

            Parameters
            ----------
            scalar

            Returns
            -------
            Polynomial
        """

        return QuasiPolynomial([p.scalar_multiplication(scalar).to_list() for p in self.polynomials])

    def __neg__(self):
        """
        qp.negation(int)

        Multiplies a quasi-polynomial with -1.

            Returns
            -------
            Polynomial
        """

        return self.scalar_multiplication(-1)

    def __add__(self, other):
        """
        qp1 + qp2

        Adds two quasi-polynomials.

            Returns
            -------
            QuasiPolynomial
        """

        if self.polynomials.size > other.polynomials.size:
            output = self.copy()
            for idx in np.arange(other.polynomials.size):
                output.polynomials[idx] = output.polynomials[idx] + other.polynomials[idx]
        else:
            output = other.copy()
            for idx in np.arange(self.polynomials.size):
                output.polynomials[idx] = output.polynomials[idx] + self.polynomials[idx]
        return output.simplify()

    def __sub__(self, other):
        """
        qp1 - qp2

        Subtracts a quasi-polynomial from another.

            Returns
            -------
            QuasiPolynomial
        """

        # TODO: Is this faster when defined without using add?
        return self + (-other)

    # TODO: Define multiplication of a polynomial with a quasi-polynomial.

    # TODO: Define multiplication of two quasi-polynomials.


def test_main():
    print(QuasiPolynomial([]))
