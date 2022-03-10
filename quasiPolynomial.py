import numpy as np
from typing import List, Union


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
        zero : Polynomial

        copy : Polynomial
            Copies a polynomial.
        simplify : Polynomial
            Simplifies a polynomial by *removing* zeros.
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

    def __init__(self, coefficient_list: List[int]) -> None:
        """
            Parameters
            ----------
            coefficient_list : List[int]
                The array of coefficients.
                The coefficient of x^n is coefficient_list[n].
        """

        self.coefficients = np.asarray(coefficient_list).astype(int, copy=False)

    @staticmethod
    def zero() -> 'Polynomial':
        return Polynomial([])

    def to_list(self) -> List[int]:
        """
        p.to_list()

        Transforms a polynomial into the integer list used to construct it.

            Returns
            -------
            List[int]
        """

        return list(self.coefficients)

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
        return str(self.to_list())

    def simplify(self) -> 'Polynomial':
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
                self.coefficients = self.coefficients[:-1].copy()
                if self.coefficients.size == 0:
                    # Check whether the polynomial is empty.
                    break
        return self

    def pretty_print(self) -> str:
        """
        p.pretty_print()

        Transform a polynomial in the mathematical form suitable to be read by humans.

            Returns
            -------
            str
        """

        if self.coefficients.size == 0:
            # Check whether the polynomial is empty.
            return '0'
        elif self.coefficients.size == 1:
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

    def __eq__(self, other: 'Polynomial') -> bool:
        """
        p1 == p2

        Checks whether two polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().coefficients, other.simplify().coefficients)

    def scalar_multiplication(self, scalar: int) -> 'Polynomial':
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
        # TODO: Include scalar multiplication in __mul__

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
            # Check whether the right polynomial is shorter to add enough zeros to make them equally large.
            output = np.concatenate((other.coefficients, np.zeros(left_size - right_size, dtype=int)))
            return Polynomial(list(output + self.coefficients)).simplify()
        else:
            # Otherwise, the left polynomial is shorter and has to be extended by adding zeros.
            output = np.concatenate((self.coefficients, np.zeros(right_size - left_size, dtype=int)))
            return Polynomial(list(output + other.coefficients)).simplify()

    def __mul__(self, other: Union['Polynomial', int]) -> 'Polynomial':
        """
        p1 * p2 | p * int

        Multiplies two polynomials or a polynomial with a scalar.

            Returns
            -------
            Polynomial
        """

        if isinstance(other, Polynomial):
            # Check whether the second object is a polynomial.
            length = self.coefficients.size + other.coefficients.size - 1
            output = []
            for total in np.arange(length):
                mini = max(0, total - other.coefficients.size + 1)
                maxi = min(total, self.coefficients.size - 1)
                output.append(
                    sum([self.coefficients[exp1] * other.coefficients[- exp1 + total] for exp1 in
                         np.arange(mini, maxi + 1)], start=0))
            return Polynomial(output)
            # TODO: Maybe it is faster to Kronecker multiply the coefficient arrays and then sum over the resulting
            #  matrix.
        elif isinstance(other, int):
            # Check whether the second object is an integer.
            return self.scalar_multiplication(other)
        else:
            # If the second polynomial is not a polynomial (but e.g. a quasi-polynomial) return NotImplemented to
            # trigger the function __rmul__ of the other class.
            return NotImplemented

    def __rmul__(self, other: int) -> 'Polynomial':
        """
        int * p

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

        for polynomial in self.polynomials:
            # Simplify the remaining polynomials.
            polynomial.simplify()
        if self.polynomials.size == 0:
            # Check whether the quasi-polynomial is empty.
            return QuasiPolynomial([])
        else:
            while self.polynomials[-1].coefficients.size == 0:
                # Check whether the last polynomial is empty to remove it.
                self.polynomials = self.polynomials[:-1].copy()
                if self.polynomials.size == 0:
                    # Recheck whether the quasi-polynomial is empty.
                    return QuasiPolynomial([])
        return self

    @staticmethod
    def new(coefficient_list: List[List[int]]) -> 'QuasiPolynomial':
        """
        new(List[List[int]])

        Creates a quasi-polynomial using a nested list of coefficients.

            Parameters
            ----------
            coefficient_list

            Returns
            -------
            QuasiPolynomial
        """

        polynomial_list = [Polynomial(coeff) for coeff in coefficient_list]
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
                if self.polynomials[1].coefficients.size == 1:
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
                    if polynomial.coefficients.size == 1:
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

    def __eq__(self, other: 'QuasiPolynomial') -> bool:
        """
        qp1 == qp2

        Checks whether two quasi-polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        return np.array_equal(self.simplify().polynomials, other.simplify().polynomials)

    def scalar_multiplication(self, scalar: int) -> 'QuasiPolynomial':
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

        return QuasiPolynomial([p.scalar_multiplication(scalar) for p in self.polynomials])
        # TODO: Can this be faster?

    def __neg__(self) -> 'QuasiPolynomial':
        """
        qp.negation(int)

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
            new_other = np.concatenate(
                (other.polynomials,
                 np.array([Polynomial.zero()] * (left_size - right_size))))
            output = new_other + self.polynomials
        else:
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

        # TODO: Is this faster when defined without using add?
        return self + (-other)

    # TODO: Define multiplication of a polynomial with a quasi-polynomial.

    def __mul__(self, other: Union['QuasiPolynomial', Polynomial, int]) -> 'QuasiPolynomial':
        """
        qp1 * qp2 | qp * p | qp * int

        Multiplies two quasi-polynomials, a quasi-polynomial with a polynomial or a quasi-polynomial with a scalar.

            Returns
            -------
            QuasiPolynomial
        """

        if isinstance(other, QuasiPolynomial):
            # Check whether the second object is a quasi-polynomial.
            length = self.polynomials.size + other.polynomials.size - 1
            output = []
            for total in np.arange(length):
                mini = max(0, total - other.polynomials.size + 1)
                maxi = min(total, self.polynomials.size - 1)
                output.append(sum(
                    [self.polynomials[exp1] * other.polynomials[- exp1 + total] for exp1 in np.arange(mini, maxi + 1)],
                    start=Polynomial.zero()))
            return QuasiPolynomial(output).simplify()
            # TODO: Maybe it is faster to Kronecker multiply the coefficient arrays and then sum over the resulting
            #  matrix.
        elif isinstance(other, Polynomial):
            # Check whether the second object is a polynomial and lift it to a quasi-polynomial.
            return self * QuasiPolynomial([other])
        if isinstance(other, int):
            # Check whether the second object is an integer and call scalar_multiplication.
            return self.scalar_multiplication(other)
        else:
            return NotImplemented

    def __rmul__(self, other: Union[Polynomial, int]) -> 'QuasiPolynomial':
        """
        p * qp | int * qp

        Multiplies a polynomial with a quasi-polynomial or a scalar with a quasi-polynomial.

            Returns
            -------
            QuasiPolynomial
        """

        return self * other


def test_main():
    print((QuasiPolynomial.new([[1, 2], [3, 4]]) + QuasiPolynomial.new([[-1, -2], [-3, -4]])).polynomials[0])
    print((QuasiPolynomial.new([[1, 2], [3, 4]]) + QuasiPolynomial.new([[-1, -2], [-3, -4]])).polynomials[1])
    print(QuasiPolynomial.new([]))
