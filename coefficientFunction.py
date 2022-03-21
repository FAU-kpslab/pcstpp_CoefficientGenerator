import quasiPolynomial
from typing import List


class CoefficientFunction:
    """
    CoefficientFunction(m, f)

    A class used to define the coefficient functions f(ell; m), encoded by a list m denoting the operator sequence.

        Parameters
        ----------
        f : QuasiPolynomial
            The coefficient function.
        m : List
            The operator sequence m identifying the function f(ell; m).

        Attributes
        ----------
        __private_key : int
            The number corresponding to the sequence vector.
        function : QuasiPolynomial
            The corresponding coefficient function f(ell; m).

        Methods
        -------
        sequence : List
            Gets the operator sequence m identifying the function f(ell; m).
    """

    def __init__(self, m: List, f: quasiPolynomial) -> None:
        """
        Parameters
        ----------
        f : QuasiPolynomial
            The coefficient function.
        m : List
            The operator sequence m identifying the function f(ell; m).
        """

        self.__private_key = sequence_to_key(m)
        self.function = f

    def sequence(self) -> List:
        """
        cf.vector()

        Gets the vector m identifying the function f(ell; m).

            Returns
            -------
            List
        """

        return key_to_sequence(self.__private_key)


def sequence_to_key(m: List) -> int:
    """
    vector_to_key(m)

    Converts the operator sequence m into the key.

        Returns
        -------
        int
    """

    return m


def key_to_sequence(key: int) -> List:
    """
    key_to_sequence(key)

    Converts the key into the operator sequence m.

        Returns
        -------
        List
    """

    return key
