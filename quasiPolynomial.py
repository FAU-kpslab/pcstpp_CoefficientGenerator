from fractions import Fraction
from typing import List, Union, Tuple, Dict
from math import isclose

import numpy as np


def is_zero(scalar: Union[int, Fraction, float]):
    """
    is_zero(scalar)

    Checks for exact numbers whether they are equal to zero and for floats whether they are close to zero.

        Returns
        -------
        bool
    """

    if isinstance(scalar, (Fraction, int)):
        return scalar == 0
    elif isinstance(scalar, float):
        return isclose(scalar, 0, abs_tol=1e-09)


def are_close(scalar1: Union[int, Fraction, float], scalar2: Union[int, Fraction, float]):
    """
    are_close(scalar1, scalar2)

    Checks for non-zero exact numbers whether they are equal and for non-zero floats whether they are close.

        Returns
        -------
        bool
    """

    if isinstance(scalar1, (Fraction, int)) and isinstance(scalar2, (Fraction, int)):
        return scalar1 == scalar2
    elif isinstance(scalar1, float) or isinstance(scalar2, float):
        if is_zero(scalar1) or is_zero(scalar2):
            return isclose(scalar1, scalar2, abs_tol=1e-09)
        else:
            return isclose(scalar1, scalar2, rel_tol=1e-09)


def inverse(scalar: Union[int, Fraction, float]):
    """
    inverse(scalar)

    Calculates 1/scalar either exactly or approximately.

        Returns
        -------
        bool
    """

    if isinstance(scalar, (int, Fraction)):
        return Fraction(1, scalar)
    elif isinstance(scalar, float):
        return 1/scalar


class Polynomial:
    """
    Polynomial(coefficient_dict)

    A class used to represent a polynomial.

        Parameters
        ----------
        coefficient_list : List[Union[int, Fraction, float]]
            The list of coefficients.
            The coefficient of x^n is coefficient_array[n].

        Attributes
        ----------
        __private_coefficients : np.ndarray[Union[int, Fraction, float]]
            The numpy array of coefficients.
            The coefficient of x^n is __private_coefficients[n].

        Methods
        -------
        __str__ : str
            Prints the coefficient array.
        coefficients : np.ndarray[Union[int, Fraction, float]]
            Gets the coefficient array.
        zero : Polynomial
            Creates an empty polynomial.
        simplify : Polynomial
            Simplifies a polynomial by *removing* zeros.
        new : Polynomial
            Creates a quasi-polynomial using a list of __private_coefficients.
        copy : Polynomial
            Copies a polynomial.
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

    def __init__(self, coefficient_list: List[Union[Fraction, float]]) -> None:
        """
            Parameters
            ----------
            coefficient_list : List[Fraction]
                The list of coefficients.
                The coefficient of x^n is coefficient_dict[n].
        """

        if len(coefficient_list) == 0:
            self.__private_coefficients = np.asarray([]).astype(Fraction)
        else:
            self.__private_coefficients = np.asarray(coefficient_list)

    def __str__(self) -> str:
        """
        print(p)

        Prints the coefficient array.

            Returns
            -------
            str
        """

        return str([str(coeff) for coeff in self.__private_coefficients])

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
            while is_zero(self.__private_coefficients[-1]):
                self.__private_coefficients = self.__private_coefficients[:-1].copy()
                # Check whether the polynomial is empty.
                if self.__private_coefficients.size == 0:
                    return Polynomial.zero()
        return self

    @staticmethod
    def new(coefficient_list: List[Union[int, Fraction, float, str]]) -> 'Polynomial':
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

        coefficients = []
        for coeff in coefficient_list:
            if isinstance(coeff, (int, str)):
                coefficients.append(Fraction(coeff))
            elif isinstance(coeff, Fraction):
                coefficients.append(coeff)
            elif isinstance(coeff, float):
                coefficients.append(coeff)
            else:
                raise TypeError("Type {} is not supported for coefficients in `Polynomial`".format(type(coeff)))
        return Polynomial(coefficients).simplify()

    def copy(self) -> 'Polynomial':
        """
        p.copy()

        Copies a polynomial.

            Returns
            -------
            Polynomial
        """

        return Polynomial(self.__private_coefficients.copy())

    def __eq__(self, other: 'Polynomial') -> bool:
        """
        p1 == p2

        Checks whether two polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        if len(self.simplify().__private_coefficients) != len(other.simplify().__private_coefficients):
            return False
        else:
            for coeff in range(len(self.__private_coefficients)):
                coeff1 = self.__private_coefficients[coeff]
                coeff2 = other.__private_coefficients[coeff]
                if not are_close(coeff1, coeff2):
                    return False
        return True

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
            if not is_zero(self.__private_coefficients[0]):
                output.append(str(self.__private_coefficients[0]))
            if not is_zero(self.__private_coefficients[1]):
                # Check whether the coefficient is 1 or -1 to leave that away.
                if are_close(self.__private_coefficients[1], 1):
                    output.append('x')
                elif are_close(self.__private_coefficients[1], -1):
                    output.append('-x')
                else:
                    output.append(str(self.__private_coefficients[1]) + 'x')
            for exponent, coefficient in list(enumerate(self.__private_coefficients))[2:]:
                # Check for the remaining coefficients whether they are zero to leave those away.
                if not is_zero(coefficient):
                    # Check for the remaining coefficients whether they are 1 or -1 to leave that away.
                    if are_close(coefficient, 1):
                        output.append('x^' + str(exponent))
                    elif are_close(coefficient, -1):
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
            output = [sum(combinations.diagonal(exponent), Fraction(0)) for exponent in
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
    QuasiPolynomial(coefficient_dict)

    A class used to represent a quasi-polynomial.

        Parameters
        ----------
        polynomial_dict : Dict[Union[int, Fraction, float], Polynomial]
            The dictionary containing all polynomials.
            The coefficient polynomial of exp(- alpha x) is polynomial_dict[alpha].

        Attributes
        ----------
        polynomial_dict : Dict[Union[int, Fraction, float], Polynomial]
            The dictionary containing all polynomials.
            The coefficient polynomial of exp(- alpha x) is polynomial[alpha].
        polynomials : List[Tuple[Union[int, Fraction, float], Polynomial]]
            The list containing tuples of all exponents and their polynomials.

        Methods
        -------
        __str__ : str
            Prints the coefficients.
        zero : QuasiPolynomial
            Creates an empty quasi-polynomial.
        sort : QuasiPolynomial
            Sorts a quasi-polynomial by exponent alpha.
        simplify : QuasiPolynomial
            Simplifies a quasi-polynomial by *removing* zero polynomials.
        new_integer : QuasiPolynomial
            Creates a quasi-polynomial with integer exponents using a nested list of __private_coefficients.
        new: QuasiPolynomial
            Creates a quasi-polynomial using a nested list of (exponential, __private_coefficients).
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

    def __init__(self, polynomial_dict: Dict[Union[int, Fraction, float], Polynomial]) -> None:
        """
            Parameters
            ----------
            polynomial_dict : Dict[Union[int, Fraction], Polynomial]
                The dictionary containing all polynomials.
                The coefficient polynomial of exp(- alpha x) is polynomial_dict[alpha].
        """

        self.polynomial_dict = polynomial_dict
        self.polynomials = polynomial_dict.items()

    def __str__(self) -> str:
        """
            print(qp)

            Prints the coefficients.

                Returns
                -------
                str
        """

        return str(
            [(e, [str(coeff) for coeff in p.coefficients()]) for e, p in sorted(self.polynomials)])

    @staticmethod
    def zero() -> 'QuasiPolynomial':
        """
        QuasiPolynomial.zero()

        Creates an empty quasi-polynomial.

            Returns
            -------
            QuasiPolynomial
        """

        return QuasiPolynomial({})

    def sort(self) -> 'QuasiPolynomial':
        """
        qp.sort()

        Sorts a quasi-polynomial by exponent alpha.

            Returns
            -------
            QuasiPolynomial
        """

        return QuasiPolynomial({e: p for e, p in sorted(self.polynomials)})

    def simplify(self) -> 'QuasiPolynomial':
        """
        qp.simplify()

        Simplifies a quasi-polynomial by *removing* zero polynomials.

            Returns
            -------
            QuasiPolynomial
        """

        output = {}
        key_list = list(self.polynomial_dict.keys())
        for i in range(len(key_list)):
            # Insert the first element.
            if len(output) == 0:
                output[key_list[i]] = self.polynomial_dict[key_list[i]]
            else:
                # Check whether two exponentials e1 and e2 are almost equal.
                if are_close(key_list[i], key_list[i - 1]):
                    # Remove e1.
                    output.popitem()
                    # Add e1 + e2.
                    output[Fraction(1, 2) * (key_list[i - 1] + key_list[i])] = self.polynomial_dict[key_list[i - 1]] + \
                                                                               self.polynomial_dict[key_list[i]]
                else:
                    output[key_list[i]] = self.polynomial_dict[key_list[i]]
        return QuasiPolynomial({e: p.simplify() for e, p in output.items() if p != Polynomial.zero()})

    @staticmethod
    def new_integer(coefficient_list: List[List[Union[Fraction, int, float, str]]]) -> 'QuasiPolynomial':
        """
        new_integer(List[List[scalar]])

        Creates a quasi-polynomial with integer exponents using a nested list of __private_coefficients.

            Parameters
            ----------
            coefficient_list

            Returns
            -------
            QuasiPolynomial
        """

        polynomial_list = {alpha: Polynomial.new(coefficient_list[alpha]) for alpha in range(len(coefficient_list))}
        return QuasiPolynomial(polynomial_list).simplify()

    @staticmethod
    def new(coefficient_dict: Dict[Union[int, Fraction, float], List[Union[Fraction, int, float, str]]]) -> 'QuasiPolynomial':
        """
        new(Dict[scalar, List[scalar]])

        Creates a quasi-polynomial using a nested list of (exponential, __private_coefficients).

            Parameters
            ----------
            coefficient_dict

            Returns
            -------
            QuasiPolynomial
        """

        polynomial_dict = {e: Polynomial.new(p) for e, p in coefficient_dict.items()}
        return QuasiPolynomial(polynomial_dict).simplify()

    def copy(self) -> 'QuasiPolynomial':
        """
        copy(p)

        Copies a quasi-polynomial.

            Returns
            -------
            quasi-Polynomial
        """

        return QuasiPolynomial({e: p.copy() for e, p in self.polynomials})

    def __eq__(self, other: 'QuasiPolynomial') -> bool:
        """
        qp1 == qp2

        Checks whether two quasi-polynomials are mathematically equal.

            Returns
            -------
            bool
        """

        if len(self.polynomial_dict) != len(other.polynomial_dict):
            return False
        else:
            key_list1 = list(self.sort().polynomial_dict.keys())
            key_list2 = list(other.sort().polynomial_dict.keys())
            for i in range(len(self.polynomial_dict)):
                # Check whether the coefficients are close.
                if not are_close(key_list1[i], key_list2[i]):
                    return False
                # Check whether the polynomials are equal.
                elif self.polynomial_dict[key_list1[i]] != other.polynomial_dict[key_list2[i]]:
                    return False
            return True

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
        else:
            output = []
            for e, p in self.sort().simplify().polynomials:
                if is_zero(e):
                    exponent = ''
                    polynomial = p.pretty_print()
                else:
                    if are_close(e, 1):
                        exponent = 'exp(-x)'
                    else:
                        exponent = 'exp(-' + str(e) + 'x)'
                    # Check whether the polynomial contains only the constant term to leave away the brackets.
                    if p.coefficients().size == 1:
                        # Check whether the polynomial contains only 1 to leave that away.
                        if p.coefficients()[0] == 1:
                            polynomial = ''
                        else:
                            polynomial = p.pretty_print()
                    else:
                        polynomial = '(' + p.pretty_print() + ')'
                output.append(polynomial + exponent)
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

        return QuasiPolynomial({e: scalar * p for e, p in self.polynomials}).simplify()

    def __neg__(self) -> 'QuasiPolynomial':
        """
        -qp

        Multiplies a quasi-polynomial with -1.

            Returns
            -------
            QuasiPolynomial
        """

        return self.scalar_multiplication(Fraction(-1))

    def __add__(self, other: 'QuasiPolynomial') -> 'QuasiPolynomial':
        """
        qp1 + qp2

        Adds two quasi-polynomials.

            Returns
            -------
            QuasiPolynomial
        """

        keys = set(self.polynomial_dict.keys())
        keys.update(other.polynomial_dict.keys())
        output = {e: self.polynomial_dict.get(e, Polynomial.zero()) + other.polynomial_dict.get(e, Polynomial.zero())
                  for e in keys}
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

    def __mul__(self,
                other: Union['QuasiPolynomial', Polynomial, Fraction, int, float]) -> 'QuasiPolynomial':
        """
        qp1 * qp2 | qp * p | qp * scalar

        Multiplies two quasi-polynomials, a quasi-polynomial with a polynomial or a quasi-polynomial with a scalar.

            Returns
            -------
            QuasiPolynomial
        """

        # Check whether the second object is a quasi-polynomial.
        if isinstance(other, QuasiPolynomial):
            output = QuasiPolynomial.zero()
            for e1, p1 in self.polynomials:
                for e2, p2 in other.polynomials:
                    output = output + QuasiPolynomial({e1 + e2: p1 * p2})
            return output.simplify()
        # Check whether the second object is a polynomial and lift it to a quasi-polynomial.
        elif isinstance(other, Polynomial):
            return self * QuasiPolynomial({0: other})
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
            output = {}
            for e, p in self.polynomials:
                # Check whether the exponent is zero.
                if is_zero(e):
                    output[0] = p.integrate()
                else:
                    # Give the polynomial a name to be able to differentiate it multiple times.
                    temp_polynomial = p
                    resulting_polynomial = -temp_polynomial * inverse(e)
                    # Give the respective integration constant a name.
                    resulting_constant = temp_polynomial.coefficients()[0] * inverse(e)
                    # Perform partial integration multiple times.
                    for n in np.arange(1, p.coefficients().size):
                        temp_polynomial = temp_polynomial.diff()
                        resulting_polynomial = resulting_polynomial - (temp_polynomial * inverse(e)**(n+1))
                        resulting_constant = resulting_constant + temp_polynomial.coefficients()[0] * inverse(e)**(n+1)
                    constant = constant + resulting_constant
                    output[e] = resulting_polynomial
            return (QuasiPolynomial(output) + QuasiPolynomial.new_integer([[constant]])).simplify()

    def get_constant(self) -> Fraction:
        """
        qp.get_constant()

        Returns the constant coefficient of the constant polynomial (alpha = 0).

            Returns
            -------
            Fraction
        """

        return self.polynomial_dict.get(0, Polynomial.zero()).get_constant()
