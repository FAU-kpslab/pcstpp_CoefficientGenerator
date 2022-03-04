import numpy as np


class Polynomial:     # TODO: Think about making Polynomial a subclass of QuasiMonomial.
    """
    Polynomial(coefficient_array)

    A class used to represent a polynomial.

        Parameters
        ----------
        coefficient_array : np.ndarray[int]
            The array of coefficients.
            The coefficient of x^n is coefficient_array[n].

        Attributes
        ----------
        coefficient_array : np.ndarray[int]
            The array of coefficients.
            The coefficient of x^n is coefficient_array[n].

        Methods
        -------
        pretty_print : str
            Transform a polynomial in the mathematical form suitable to be read by humans.
    """

    def __init__(self, coefficient_array: np.ndarray) -> None:
        """
        Parameters
        ----------
        coefficient_array : np.ndarray[int]
            The array of coefficients.
            The coefficient of x^n is coefficient_array[n].
        """

        self.coefficients = coefficient_array

    def __str__(self) -> str:
        return str(self.coefficients.tolist())

    def pretty_print(self) -> str:
        """
        p.pretty_print()

        Transform a polynomial in the mathematical form suitable to be read by humans.
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

    # TODO: Define simplification.

    # TODO: Define multiplication with a scalar.

    # TODO: Define negation of a polynomial.

    # TODO: Define addition of two polynomials.

    # TODO: Define multiplication of two polynomials.


class QuasiPolynomial:
    """
        QuasiPolynomial(polynomial_array)

        A class used to represent a quasi-polynomial.

            Parameters
            ----------
            polynomial_array : np.ndarray[np.ndarray[int]]
                The array of polynomials.
                The polynomial in front of exp(-nx) is polynomial_array[n].

            Attributes
            ----------
            polynomial_array : np.ndarray[np.ndarray[int]]
                The array of polynomials.
                The polynomial in front of exp(-nx) is polynomial_array[n].

            Methods
            -------
            pretty_print : str
                Transform a quasi-polynomial in the mathematical form suitable to be read by humans.
        """

    def __init__(self, polynomial_array: np.ndarray) -> None:
        """
                Parameters
                ----------
                polynomial_array : np.ndarray[np.ndarray[int]]
                    The array of polynomials.
                    The polynomial in front of exp(-nx) is polynomial_array[n].
                """
        self.polynomials = polynomial_array

    def __str__(self) -> str:
        return str([polynomial.coefficients.tolist() for polynomial in self.polynomials])
        # TODO: What is "Convert method to property?"

    def pretty_print(self) -> str:
        """
        qp.pretty_print()

        Transform a quasi-polynomial in the mathematical form suitable to be read by humans.
        """
        if len(self.polynomials) == 0:
            # Check whether the quasi-polynomial is empty.
            return '0'
        if len(self.polynomials) == 1:
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

    # TODO: Define simplification.

    # TODO: Define multiplication with a scalar.

    # TODO: Define negation of a quasi-polynomial.

    # TODO: Define addition of two quasi-polynomials.

    # TODO: Define multiplication of a polynomial with a quasi-polynomial.

    # TODO: Define multiplication of two quasi-polynomials.



def test_main():
    qp = QuasiPolynomial(np.array([Polynomial(np.array([2, 1]).astype(int)), Polynomial(np.array([]).astype(int)),
                                   Polynomial(np.array([4]).astype(int))]))
    print(qp)
    print(qp.pretty_print())
    print(Polynomial(np.array([2, 1])))
    print(Polynomial(np.array([2, 1])).pretty_print())
