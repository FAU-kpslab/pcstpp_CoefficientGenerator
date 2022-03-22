from quasiPolynomial import QuasiPolynomial
from differentialEquation import differential_equation
from typing import Tuple, Dict


class CoefficientFunction:
    """
    CoefficientFunction(sequence, f)

    A class used to define the coefficient functions f(ell; m), encoded by a list m denoting the operator sequence.

        Parameters
        ----------
        f : QuasiPolynomial
            The coefficient function.
        sequence : Tuple
            The operator sequence m identifying the function f(ell; m).

        Attributes
        ----------
        __private_key : int
            The number corresponding to the sequence vector.
        function : QuasiPolynomial
            The corresponding coefficient function f(ell; m).

        Methods
        -------
        sequence : Tuple
            Gets the operator sequence m identifying the function f(ell; m).
        indices : Tuple
            Returns the indices in the operator sequence m.
    """

    def __init__(self, sequence: Tuple, f: QuasiPolynomial) -> None:
        """
        Parameters
        ----------
        f : QuasiPolynomial
            The coefficient function.
        sequence : Tuple
            The operator sequence m identifying the function f(ell; m).
        """

        self.__private_key = sequence_to_key(sequence)
        self.function = f

    def sequence(self) -> Tuple:
        """
        cf.vector()

        Gets the vector sequence identifying the function f(ell; sequence).

            Returns
            -------
            Tuple
        """

        return key_to_sequence(self.__private_key)

    def indices(self) -> Tuple:
        """
        cf.indices()

        Returns the indices in the operator sequence m.

            Returns
            -------
            Tuple
        """

        return sequence_to_indices(self.sequence())


class FunctionCollection:
    """
    FunctionCollection()

    A class used to store all calculated coefficient functions f(ell; m).

        Attributes
        ----------
        __private_collection : Dict
            The dictionary storing keys sequence_to_key(m) and values f(ell; m).

        Methods
        -------
        __contains__ : bool
            Checks for the operator sequence m whether the function f(ell; m) is already calculated.
        __setitem__ : None
            Saves the function f(ell; m).
        __getitem__ : CoefficientFunction
            Returns for the operator sequence m the function f(ell; m) and computes it if necessary.
    """

    def __init__(self) -> None:
        self.__private_collection = dict()

    def __contains__(self, sequence: Tuple) -> bool:
        """
        sequence in FunctionCollection

        Checks for the operator sequence m whether the function f(ell; m) is already calculated.

            Parameters
            ----------
            sequence : Tuple
                The operator sequence m identifying the function f(ell; m).

            Returns
            -------
            bool
        """

        return sequence_to_key(sequence) in self.__private_collection

    def __setitem__(self, sequence: Tuple, f: QuasiPolynomial) -> None:
        """
        FunctionCollection[sequence] = f

        Saves the function f(ell; m).

            Parameters
            ----------
            sequence : Tuple
                The operator sequence m identifying the function f(ell; m).
            f : QuasiPolynomial
                The function f(ell; m).
        """

        self.__private_collection[sequence_to_key(sequence)] = f

    def __getitem__(self, sequence: Tuple) -> CoefficientFunction:
        """
        FunctionCollection[sequence]

        Returns for the operator sequence m the function f(ell; m) and computes it if necessary.

            Parameters
            ----------
            sequence : Tuple
                The operator sequence m.

            Returns
            -------
            CoefficientFunction
        """

        if sequence in self:
            return CoefficientFunction(sequence, self.__private_collection[sequence_to_key(sequence)])
        else:
            # Calculate the function if it doesn't exist yet.
            self[sequence] = differential_equation(sequence)
            return self[sequence]


def sequence_to_key(sequence: Tuple) -> int:
    """
    vector_to_key(sequence)

    Converts the operator sequence m into the key.

        Returns
        -------
        int
    """

    return sequence


def key_to_sequence(key: int) -> Tuple:
    """
    key_to_sequence(key)

    Converts the key into the operator sequence m.

        Returns
        -------
        Tuple
    """

    return key


def sequence_to_indices(sequence: Tuple) -> Tuple:
    """
    sequence_to_indices(key)

    Converts the operator sequence into the indices of the operator sequence m.

        Returns
        -------
        Tuple
    """

    return sequence