from quasiPolynomial import QuasiPolynomial
from mathematics import *
from typing import Tuple, Dict, Optional


class CoefficientFunction:
    """
    CoefficientFunction(sequence, function)

    A class used to define the coefficient functions f(ell; m), encoded by a tuple m denoting the operator sequence.

        Parameters
        ----------
        sequence : Tuple[Tuple, Tuple]
            The operator sequence m identifying the function f(ell; m).
        function : QuasiPolynomial
            The corresponding coefficient function f(ell; m).

        Attributes
        ----------
        __private_key : Tuple[int, int]
            The number corresponding to the sequence of operators.
        function : QuasiPolynomial
            The corresponding coefficient function f(ell; m).

        Methods
        -------
        sequence : Tuple[Tuple, Tuple]
            Gets the operator sequence m identifying the function f(ell; m).
        __str__ : str
            Prints the operator sequence m and the coefficient array of the corresponding coefficient function
            f(ell; m).
        pretty_print() : str
            Prints the operator sequence m and corresponding coefficient function f(ell; m).
    """

    def __init__(self, sequence: Tuple[Tuple, Tuple], function: QuasiPolynomial) -> None:
        """
        Parameters
        ----------
        sequence : Tuple
            The operator sequence m identifying the function f(ell; m).
        function : QuasiPolynomial
            The corresponding coefficient function f(ell; m).
        """

        self.__private_key = sequence_to_key(sequence)
        self.function = function

    def sequence(self) -> Tuple[Tuple, Tuple]:
        """
        cf.sequence()

        Gets the operator sequence m identifying the function f(ell; m).

            Returns
            -------
            Tuple[Tuple, Tuple]
        """

        return key_to_sequence(self.__private_key)

    def __str__(self) -> str:
        """
        print(cf)

        Prints the operator sequence m and the coefficient array of the corresponding coefficient function f(ell; m).

            Returns
            -------
            str
        """

        return str(self.sequence()) + ': ' + str(self.function)

    def pretty_print(self) -> str:
        """
        cf.pretty_print()

        Prints the operator sequence m and corresponding coefficient function f(ell; m).

            Returns
            -------
            str
        """

        return str(self.sequence()) + ': ' + self.function.pretty_print()


class FunctionCollection:
    """
    FunctionCollection()

    A class used to store all calculated coefficient functions f(ell; m).

        Parameters
        ----------
        translation : Dict
            The dictionary assigning an index to every operator.

        Attributes
        ----------
        __private_collection : Dict
            The dictionary storing keys sequence_to_key(m) and values f(ell; m).
        translation : Dict
            The dictionary assigning an index to every operator.

        Methods
        -------
        __contains__ : bool
            Checks for the operator sequence m whether the function f(ell; m) is already calculated.
        __setitem__ : None
            Saves the function f(ell; m) if it is not already saved.
        __getitem__ : CoefficientFunction
            Returns for the operator sequence m the function f(ell; m) or None.
        keys :
            Returns all calculated operator sequences m.
        __str__ : str
            Prints the collection.
        pretty_print() : str
            Transform the collection in a form suitable to be read by humans.
    """

    def __init__(self, translation: Dict) -> None:
        self.__private_collection = dict()
        self.translation = translation

    def __contains__(self, sequence: Tuple[Tuple, Tuple]) -> bool:
        """
        sequence in FunctionCollection

        Checks for the operator sequence m whether the function f(ell; m) is already calculated.

            Parameters
            ----------
            sequence : Tuple[Tuple, Tuple]
                The operator sequence m identifying the function f(ell; m).

            Returns
            -------
            bool
        """

        return sequence_to_key(sequence) in self.__private_collection

    def __setitem__(self, sequence: Tuple[Tuple, Tuple], function: QuasiPolynomial) -> None:
        """
        FunctionCollection[sequence] = function

        Saves the function f(ell; m) if it is not already saved.

            Parameters
            ----------
            sequence : Tuple[Tuple, Tuple]
                The operator sequence m identifying the function f(ell; m).
            function : QuasiPolynomial
                The function f(ell; m).
        """

        if sequence not in self:
            self.__private_collection[sequence_to_key(sequence)] = function

    def __getitem__(self, sequence: Tuple[Tuple, Tuple]) -> Optional[CoefficientFunction]:
        """
        FunctionCollection[sequence]

        Returns for the operator sequence m the function f(ell; m) or None.

            Parameters
            ----------
            sequence : Tuple[Tuple, Tuple]
                The operator sequence m.

            Returns
            -------
            CoefficientFunction
        """

        if sequence in self:
            return CoefficientFunction(sequence, self.__private_collection[sequence_to_key(sequence)])
        else:
            return None

    def keys(self) -> List[Tuple[Tuple, Tuple]]:
        """
        FunctionCollection.keys()

        Returns all calculated operator sequences m.

            Returns
            -------
            List[Tuple[Tuple, Tuple]]
        """

        return [key_to_sequence(key) for key in self.__private_collection.keys()]

    def __str__(self) -> str:
        """
        print(FunctionCollection)

        Prints the collection.

            Returns
            -------
            str
        """

        output = []
        for key in self.keys():
            output.append(str(self[key]))
        return str(output)

    def pretty_print(self) -> str:
        """
        FunctionCollection.pretty_print()

        Transform the collection in a form suitable to be read by humans.

            Returns
            -------
            str
        """

        output = str()
        for key in self.keys():
            output = output + self[key].pretty_print() + '\n'
        return output


def sequence_to_key(sequence: Tuple[Tuple, Tuple]) -> Tuple[Tuple, Tuple]:  # TODO The key is supposed to be an integer.
    """
    vector_to_key(sequence)

    Converts the operator sequence m into the key.

        Returns
        -------
        Tuple[Tuple, Tuple]
    """

    return sequence


def key_to_sequence(key: Tuple[Tuple, Tuple]) -> Tuple[Tuple, Tuple]:  # TODO The key is supposed to be an integer.
    """
    key_to_sequence(key)

    Converts the key into the operator sequence m.

        Returns
        -------
        Tuple[Tuple, Tuple]
    """

    return key


def sequence_to_indices(sequence: Tuple[Tuple, Tuple], translation: Dict) -> Tuple[List, List]:
    """
    sequence_to_indices(key)

    Converts the operator sequence into the indices of the operator sequence m.

        Returns
        -------
        Tuple[List, List]
    """

    sequence_left = [translation[operator_left] for operator_left in sequence[0]]
    sequence_right = [translation[operator_right] for operator_right in sequence[1]]
    return sequence_left, sequence_right


def calc(sequence: Tuple[Tuple, Tuple], collection: FunctionCollection, translation: Dict,
         max_energy: int) -> QuasiPolynomial:
    """
    calc(sequence)

    Returns or calculates the function f(ell; m) corresponding to the operator sequence m.

        Returns
        -------
        QuasiPolynomial
    """

    # Check whether the function is already calculated.
    if collection[sequence] is not None:
        return collection[sequence].function
    else:
        partition_list = partitions(sequence)
        integrand = QuasiPolynomial.zero()
        for partition in partition_list:
            # Rename the operator sequences.
            s1 = partition[0]
            s2 = partition[1]
            # Check whether the required functions are already calculated.
            f1 = calc(partition[0], collection, translation, max_energy)
            f2 = calc(partition[1], collection, translation, max_energy)
            # Translate the operator sequences into its indices.
            m = sequence_to_indices(sequence, translation)
            m1 = sequence_to_indices(s1, translation)
            m2 = sequence_to_indices(s2, translation)
            # Only calculate non-vanishing contributions to the integrand.
            if (abs(energy(m1)) <= max_energy) & (abs(energy(m2)) <= max_energy):
                integrand = integrand + exponential(m, m1, m2) * signum(m1, m2) * f1 * f2
        result = integrand.integrate()
        # Insert the result into the collection.
        collection[sequence] = result
        return result


def trafo_calc(sequence: Tuple[Tuple, Tuple], trafo_collection: FunctionCollection, collection: FunctionCollection,
               translation: Dict, max_energy: int) -> QuasiPolynomial:
    """
    trafo_calc(sequence)

    Calculates the function G(ell; m) corresponding to the operator sequence m.

        Returns
        -------
        QuasiPolynomial
    """

    # Check whether the function is already calculated.
    if trafo_collection[sequence] is not None:
        return trafo_collection[sequence].function
    else:
        m = sequence_to_indices(sequence, translation)
        integrand = exponential(([], []), ([], []), m) * signum(([], []), m) * calc(sequence, collection, translation, max_energy)
        partition_list = partitions(sequence)
        for partition in partition_list:
            # Rename the operator sequences.
            s1 = partition[0]
            s2 = partition[1]
            # Check whether the required functions are already calculated.
            g1 = trafo_calc(partition[0], trafo_collection, collection, translation, max_energy)
            f2 = calc(partition[1], collection, translation, max_energy)
            # Translate the operator sequences into its indices.
            m1 = sequence_to_indices(s1, translation)
            m2 = sequence_to_indices(s2, translation)
            # Calculate the contributions to the integrand.
            integrand = integrand + exponential(([], []), ([], []), m2) * signum(([], []), m2) * g1 * f2
        result = integrand.integrate()
        # Insert the result into the collection.
        trafo_collection[sequence] = result
        return result
