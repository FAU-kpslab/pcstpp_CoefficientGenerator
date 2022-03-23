from quasiPolynomial import QuasiPolynomial
from mathematics import *
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
        #indices : List
        #    Returns the indices in the operator sequence m.
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

    def __str__(self) -> str:
        return str(self.sequence()) + ': ' + str(self.function)

    def pretty_print(self) -> str:
        return str(self.sequence()) + ': ' + self.function.pretty_print()


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
            Returns for the operator sequence m the function f(ell; m).
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

        Returns for the operator sequence m the function f(ell; m).

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
            self[sequence] = differential_equation(sequence, self, self.translation)
            return self[sequence]

    def keys(self) -> List[Tuple]:
        """
        FunctionCollection.keys()

        Returns all calculated operator sequences m.

            Returns
            -------
            List[Tuple]
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


def sequence_to_key(sequence: Tuple) -> Tuple:  # TODO The key is supposed to be an integer.
    """
    vector_to_key(sequence)

    Converts the operator sequence m into the key.

        Returns
        -------
        Tuple
    """

    return sequence


def key_to_sequence(key: Tuple) -> Tuple:  # TODO The key is supposed to be an integer.
    """
    key_to_sequence(key)

    Converts the key into the operator sequence m.

        Returns
        -------
        Tuple
    """

    return key


def sequence_to_indices(sequence: Tuple, translation: Dict) -> List:
    """
    sequence_to_indices(key)

    Converts the operator sequence into the indices of the operator sequence m.

        Returns
        -------
        List
    """

    return [translation[operator] for operator in sequence]


def differential_equation(sequence: Tuple, collection: FunctionCollection, translation: Dict) -> QuasiPolynomial:
    """
    differential_equation(sequence)

    Calculates the function f(ell; m) corresponding to the operator sequence m.

        Returns
        -------
        QuasiPolynomial
    """

    partition_list = partitions(sequence)
    integrand = QuasiPolynomial.zero()
    for partition in partition_list:
        # Rename the operator sequences.
        s1 = partition[0]
        s2 = partition[1]
        # Translate the operator sequences into its indices.
        m = sequence_to_indices(sequence, translation)
        m1 = sequence_to_indices(s1, translation)
        m2 = sequence_to_indices(s2, translation)
        integrand = integrand + exponential(m, m1, m2) * signum(m1, m2) * collection[s1].function * collection[
            s2].function
    return integrand.integrate()
