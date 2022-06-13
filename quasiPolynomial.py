from fractions import Fraction
from typing import List, Union, Tuple

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
        __private_coefficients : np.ndarray[Fraction]
            The numpy array of coefficients.
            The coefficient of x^n is __private_coefficients[n].

        Methods
        -------
        coefficients : np.ndarray[Fraction]
            Gets the coefficient array.
        zero : Polynomial
            Creates an empty polynomial.
        simplify : Polynomial
            Simplifies a polynomial by *removing* zeros.
        new : Polynomial
            Creates a quasi-polynomial using a list of __private_coefficients.
        copy : Polynomial
            Copies a polynomial.
        __str__ : str
            Prints the coefficient array.
        __eq__ : bool
            Checks whether two polynomials are mathematically equal.
        pretty_print : str
            Transform a polynomial in the mathematical form suitable to be read by humans.
        scalar_multiplication : Polynomial
            Multiplies a polynomial with a scalar.
        __neg__ : Polynomial
            Multiplies a polynomial with -1.
        __add__ : Polynomial
            Adds two polynomials.
        __sub__ : Polynomials
            Subtracts a polynomial from another.
        __mul__ : Polynomial
            Multiplies two polynomials or a polynomial with a scalar.
        __rmul__ : Polynomial
            Multiplies a scalar with a polynomial.
        integrate : Polynomial
            Integrate a polynomial with starting condition 0.
        diff : Polynomial
            Perform the derivative of a polynomial.
        get_constant : Fraction
            Returns the constant coefficient.
    """

    def __init__(self, coefficient_list: Union[List[Fraction]]) -> None:
        """
            Parameters
            ----------
            coefficient_list : List[int]
                The list of coefficients.
                The coefficient of x^n is coefficient_list[n].
        """

        self.__private_coefficients = np.asarray(coefficient_list).astype(Fraction, copy=False)

    def coefficients(self) -> np.ndarray:
        """
        p.coefficients()

        Gets the coefficient array.

            Returns
            -------
            np.ndarray[Fraction]
        """

        return self.__private_coefficients.copy()
        # This could be too slow.

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
        if self.__private_coefficients.size != 0:
            # Check whether the last coefficient is zero to remove it.
            while self.__private_coefficients[-1] == 0:
                self.__private_coefficients = self.__private_coefficients[:-1].copy()
                # Check whether the polynomial is empty.
                if self.__private_coefficients.size == 0:
                    return Polynomial.zero()
        return self

    @staticmethod
    def new(coefficient_list: Union[List[Fraction], List[int], List[float], List[str]]) -> 'Polynomial':
        """
        new(List[scalar])

        Creates a quasi-polynomial using a list of __private_coefficients.

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
        p.copy()

        Copies a polynomial.

            Returns
            -------
            Polynomial
        """

        return Polynomial(self.__private_coefficients.copy())

    def __str__(self) -> str:
        """
        print(p)

        Prints the coefficient array.

            Returns
            -------
            str
        """

        return str([str(coeff) for coeff in self.__private_coefficients])

    def __eq__(self, other: 'Polynomial') -> bool:
        """
        p1 == p2

        Checks whether two polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().__private_coefficients, other.simplify().__private_coefficients)

    def pretty_print(self) -> str:
        """
        p.pretty_print()

        Transform a polynomial in the mathematical form suitable to be read by humans.

            Returns
            -------
            str
        """

        # Check whether the polynomial is empty.
        if self == Polynomial.zero():
            return '0'
        # Check whether the polynomial contains only the constant term.
        elif self.__private_coefficients.size == 1:
            return str(self.__private_coefficients[0])
        else:
            output = []
            # Check whether the constant term is zero to leave that away.
            if self.__private_coefficients[0] != 0:
                output.append(str(self.__private_coefficients[0]))
            if self.__private_coefficients[1] != 0:
                # Check whether the coefficient is 1 or -1 to leave that away.
                if self.__private_coefficients[1] == 1:
                    output.append('x')
                elif self.__private_coefficients[1] == -1:
                    output.append('-x')
                else:
                    output.append(str(self.__private_coefficients[1]) + 'x')
            for exponent, coefficient in list(enumerate(self.__private_coefficients))[2:]:
                # Check for the remaining coefficients whether they are zero to leave those away.
                if coefficient != 0:
                    # Check for the remaining coefficients whether they are 1 or -1 to leave that away.
                    if coefficient == 1:
                        output.append('x^' + str(exponent))
                    elif coefficient == -1:
                        output.append('-x^' + str(exponent))
                    else:
                        output.append(str(coefficient) + 'x^' + str(exponent))
            return '+'.join(output).replace('+-', '-')

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

        return Polynomial(scalar * self.__private_coefficients).simplify()

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

        left_size = self.__private_coefficients.size
        right_size = other.__private_coefficients.size
        if left_size > right_size:
            # Add zeros to the shorter polynomial to make them equally long.
            output = np.concatenate((other.__private_coefficients, np.zeros(left_size - right_size, dtype=int)))
            return Polynomial(list(output + self.__private_coefficients)).simplify()
        else:
            # Add zeros to the shorter polynomial to make them equally long.
            output = np.concatenate((self.__private_coefficients, np.zeros(right_size - left_size, dtype=int)))
            return Polynomial(list(output + other.__private_coefficients)).simplify()

    def __sub__(self, other: 'Polynomial') -> 'Polynomial':
        """
        p1 - p2

        Subtracts a polynomial from another.

            Returns
            -------
            QuasiPolynomial
        """

        return self + (-other)

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
            # Calculate the matrix containing all combinations of __private_coefficients of both polynomials.
            # Flip it such that all __private_coefficients corresponding to the same x^n are part of the same diagonals.
            combinations = np.flipud(np.outer(self.__private_coefficients, other.__private_coefficients))
            # Sum over the diagonals to obtain the real __private_coefficients.
            output = [sum(combinations.diagonal(exponent), start=Fraction(0)) for exponent in
                      np.arange(- self.__private_coefficients.size + 1, other.__private_coefficients.size)]
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

    def integrate(self) -> 'Polynomial':
        """
        integrate(p)

        Integrate a polynomial with starting condition 0.

            Returns
            -------
            Polynomial
        """

        prefactors = np.array([Fraction(1, n + 1) for n in np.arange(self.__private_coefficients.size)])
        output = (prefactors * self.__private_coefficients).tolist()
        return Polynomial([Fraction(0)] + output)

    def diff(self) -> 'Polynomial':
        """
        diff(p)

        Perform the derivative of a polynomial.

            Returns
            -------
            Polynomial
        """

        prefactors = np.arange(1, self.__private_coefficients.size)
        polynomial = self.__private_coefficients[1:].copy()
        return Polynomial((prefactors * polynomial).tolist())

    def get_constant(self) -> Fraction:
        """
        p.get_constant()

        Returns the constant coefficient.

            Returns
            -------
            Fraction
        """

        if self == Polynomial.zero():
            return Fraction(0)
        else:
            return self.coefficients()[0]


class QuasiPolynomial:
    """
    QuasiPolynomial(coefficient_list)

    A class used to represent a quasi-polynomial.

        Parameters
        ----------
        polynomial_list : List[Tuple[int, Polynomial]]
            The list of polynomials.
            The coefficient polynomial of exp(- alpha x) is polynomial_list[alpha].

        Attributes
        ----------
        polynomials : np.ndarray[Polynomial]
            The numpy array of polynomials.
            The coefficient polynomial of exp(- alpha x) is stored as (alpha, polynomial).

        Methods
        -------
        __str__ : str
            Prints the coefficient array.
        zero : QuasiPolynomial
            Creates an empty quasi-polynomial.
        simplify : QuasiPolynomial
            Simplifies a quasi-polynomial by *removing* zero polynomials and adding polynomials with same exponent.
        sort : QuasiPolynomial
            Sorts a quasi-polynomial by exponential alpha.
        new : QuasiPolynomial
            Creates a quasi-polynomial using a nested list of __private_coefficients.
        copy : QuasiPolynomial
            Copies a quasi-polynomial.
        __eq__ : bool
            Checks whether two quasi-polynomials are mathematically equal.
        pretty_print : str
            Transform a quasi-polynomial in the mathematical form suitable to be read by humans.
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
        integrate : QuasiPolynomial
            Integrate a quasi-polynomial with starting condition 0.
        get_constant : Fraction
            Returns the constant coefficient of the constant polynomial (alpha = 0).
    """

    def __init__(self, polynomial_list: List[Tuple[int, Polynomial]]) -> None:
        """
            Parameters
            ----------
            polynomial_list : List[Tuple[int, Polynomial]]
                The list of polynomials.
                The coefficient polynomial of exp(- alpha x) is stored as (alpha, polynomial).
        """

        self.polynomials = polynomial_list

    def __str__(self) -> str:
        """
            print(qp)

            Prints the coefficient array.

                Returns
                -------
                str
        """

        return str([(p[0], [str(coeff) for coeff in p[1].coefficients()]) for p in self.polynomials])  # TODO: Print alpha

    @staticmethod
    def zero() -> 'QuasiPolynomial':
        """
        QuasiPolynomial.zero()

        Creates an empty quasi-polynomial.

            Returns
            -------
            QuasiPolynomial
        """

        return QuasiPolynomial([])

    def simplify(self) -> 'QuasiPolynomial':
        """
        qp.simplify()

        Simplifies a quasi-polynomial by *removing* zero polynomials and adding polynomials with same exponent.

            Returns
            -------
            QuasiPolynomial
        """

        for polynomial in self.polynomials:
            polynomial[1].simplify()
        # Check whether the quasi-polynomial is empty.
        if len(self.polynomials) == 0:
            return QuasiPolynomial.zero()
        else:
            output = [self.polynomials[0]]
            for p in self.polynomials[1:]:
                if p[0] == output[-1][0]:
                    output[-1] = (output[-1][0], output[-1][1] + p[1])
                else:
                    output.append(p)
            # TODO: Remove empty polynomials in the middle.
            # Check whether the last polynomial is empty to remove it.
            while output[-1][1] == Polynomial.zero():
                output.pop(-1)
                # Recheck whether the quasi-polynomial is empty.
                if len(output) == 0:
                    return QuasiPolynomial.zero()
            return QuasiPolynomial(output)

    def sort(self) -> 'QuasiPolynomial':
        """
        qp.sort()

        Sorts a quasi-polynomial by exponential alpha.

            Returns
            -------
            QuasiPolynomial
        """

        self.polynomials.sort(key=lambda p: p[0])
        return self

    @staticmethod
    def new(coefficient_list: Union[List[List[Fraction]], List[List[int]], List[List[float]], List[List[str]]]) -> \
            'QuasiPolynomial':
        """
        new(List[List[scalar]])

        Creates a quasi-polynomial using a nested list of __private_coefficients.

            Parameters
            ----------
            coefficient_list

            Returns
            -------
            QuasiPolynomial
        """

        polynomial_list = [(alpha, Polynomial.new(coefficient_list[alpha])) for alpha in range(len(coefficient_list))]
        return QuasiPolynomial(polynomial_list).simplify()

    def copy(self) -> 'QuasiPolynomial':
        """
        copy(p)

        Copies a quasi-polynomial.

            Returns
            -------
            quasi-Polynomial
        """

        return QuasiPolynomial([(alpha, self.polynomials[alpha][1].copy()) for alpha in range(len(self.polynomials))])

    def __eq__(self, other: 'QuasiPolynomial') -> bool:
        """
        qp1 == qp2

        Checks whether two quasi-polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().polynomials, other.simplify().polynomials)

    def pretty_print(self) -> str:
        """
        qp.pretty_print()

        Transform a quasi-polynomial in the mathematical form suitable to be read by humans.

            Returns
            -------
            str
        """

        if self == QuasiPolynomial.zero():
            return '0'
        # Check whether the quasi-polynomial contains only the first polynomial.
        if len(self.polynomials) == 1:
            return self.polynomials[0][1].pretty_print()
        else:
            output = []
            if self.polynomials[0][1] != Polynomial.zero():
                output.append(self.polynomials[0][1].pretty_print())
            if self.polynomials[1][1] != Polynomial.zero():
                # Check whether the polynomial contains only the constant term to leave away the brackets.
                if self.polynomials[1][1].coefficients().size == 1:
                    # Check whether the polynomial contains only 1 to leave that away.
                    if self.polynomials[1][1].coefficients()[0] == 1:
                        output.append('exp(-x)')
                    else:
                        output.append(self.polynomials[1][1].pretty_print() + 'exp(-x)')
                else:
                    output.append('(' + self.polynomials[1][1].pretty_print() + ')exp(-x)')
            for alpha in range(2, len(self.polynomials)):
                # Check for the remaining polynomials whether they are zero to leave those away.
                if self.polynomials[alpha][1] != Polynomial.zero():
                    # Check for the remaining polynomials whether they contain only the constant term to leave away the
                    # brackets.
                    if len(self.polynomials[alpha][1].coefficients()) == 1:
                        # Check for the remaining polynomials whether they contain only 1 to leave that away.
                        if self.polynomials[alpha][1].coefficients()[0] == 1:
                            output.append('exp(-' + str(alpha) + 'x)')
                        else:
                            output.append(self.polynomials[alpha][1].pretty_print() + 'exp(-' + str(alpha) + 'x)')
                    else:
                        output.append('(' + self.polynomials[alpha][1].pretty_print() + ')exp(-' + str(alpha) + 'x)')
            return '+'.join(output).replace('+-', '-')

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

        return QuasiPolynomial(
            [(alpha, scalar * self.polynomials[alpha][1]) for alpha in range(len(self.polynomials))]).simplify()

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

        self_size = len(self.polynomials)
        other_size = len(other.polynomials)
        if self_size > other_size:
            output = [(alpha, self.polynomials[alpha][1] + other.polynomials[alpha][1]) for alpha in
                      range(other_size)] + self.polynomials[other_size:]
        else:
            output = [(alpha, self.polynomials[alpha][1] + other.polynomials[alpha][1]) for alpha in
                      range(self_size)] + other.polynomials[self_size:]
        print(QuasiPolynomial(output).simplify())
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
            output = [(p1[0] + p2[0], p1[1] * p2[1]) for p1 in self.polynomials for p2 in other.polynomials]
            return QuasiPolynomial(output).sort().simplify()
        # Check whether the second object is a polynomial and lift it to a quasi-polynomial.
        elif isinstance(other, Polynomial):
            return self * QuasiPolynomial([(0, other)])
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

    def integrate(self) -> 'QuasiPolynomial':
        """
        integrate(p)

        Integrate a polynomial with starting condition 0.

            Returns
            -------
            QuasiPolynomial
        """

        # Check whether the quasi-polynomial is empty.
        if self == QuasiPolynomial.zero():
            return QuasiPolynomial.zero()
        else:
            # Initiate the constant of integration.
            constant = Fraction(0)
            # Integrate the first polynomial (that has no exp).
            output = [(0, self.polynomials[0][1].integrate())]
            # Loop over all other polynomials to integrate them one at a time.
            for alpha in np.arange(1, len(self.polynomials)):
                if self.polynomials[alpha][1] == Polynomial.zero():
                    output.append((alpha, Polynomial.zero()))
                else:
                    # Give the polynomial a name to be able to differentiate it multiple times.
                    temp_polynomial = self.polynomials[alpha][1]
                    resulting_polynomial = -temp_polynomial * Fraction(1, alpha)
                    # Give the respective integration constant a name.
                    resulting_constant = temp_polynomial.coefficients()[0] * Fraction(1, alpha)
                    # Perform partial integration multiple times.
                    for n in np.arange(1, self.polynomials[alpha][1].coefficients().size):
                        temp_polynomial = temp_polynomial.diff()
                        resulting_polynomial = resulting_polynomial - (temp_polynomial * Fraction(1, alpha**(n + 1)))
                        resulting_constant = resulting_constant + temp_polynomial.coefficients()[0] * Fraction(1, alpha ** (n + 1))
                    constant = constant + resulting_constant
                    output.append((alpha, resulting_polynomial))
            return (QuasiPolynomial(output) + QuasiPolynomial.new([[constant]])).simplify()

    def get_constant(self) -> Fraction:
        """
        qp.get_constant()

        Returns the constant coefficient of the constant polynomial (alpha = 0).

            Returns
            -------
            Fraction
        """

        if self == QuasiPolynomial.zero():
            return Fraction(0)
        else:
            return self.polynomials[0][1].get_constant()
