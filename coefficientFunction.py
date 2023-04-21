"""
Defines classes to store and calculate a single coefficient function $f(ell; m)$ 
and the collection of all $f(ell; m)$.
"""

from quasiPolynomial import QuasiPolynomial
from mathematics import *
from typing import Tuple, Dict, Optional, Callable


class CoefficientFunction:
    """
    CoefficientFunction(sequence, function)

    A class used to define the coefficient functions f(ell; m), encoded by a tuple m denoting the operator sequence.

    Parameters
    ----------
    sequence : Tuple[Tuple[int,...],...]
        The operator sequence m identifying the function f(ell; m).
    function : QuasiPolynomial
        The corresponding coefficient function f(ell; m).

    Attributes
    ----------
    __private_key : Tuple[Tuple[int,...],...]
        The number corresponding to the sequence of operators.
    function : QuasiPolynomial
        The corresponding coefficient function f(ell; m).

    Examples
    --------
    >>> print(CoefficientFunction(((1, 2), (0,)), QuasiPolynomial.new({0: [1, 2], 3: [4, 5]})))
    ((-1, 1), (0,)): [(0, ['1', '2']), (3, ['4', '5'])]

    Prints the operator sequence m and the coefficient array of the corresponding coefficient function f(ell; m).
    """

    def __init__(self, sequence: Sequence, function: QuasiPolynomial) -> None:
        """
        Parameters
        ----------
        sequence : Tuple[Tuple[int,...],...]
            The operator sequence m identifying the function f(ell; m).
        function : QuasiPolynomial
            The corresponding coefficient function f(ell; m).
        """

        self.__private_key = sequence_to_key(sequence)
        self.function = function

    def sequence(self) -> Sequence:
        """
        cf.sequence()

        Gets the operator sequence m identifying the function f(ell; m).

        Returns
        -------
        Tuple[Tuple[int,...],...]
            Operator sequence.
        """

        return key_to_sequence(self.__private_key)

    def __str__(self) -> str:
        """
        print(cf)

        Prints the operator sequence m and the coefficient array of the corresponding coefficient function f(ell; m).

        Returns
        -------
        str
            Operator sequence and coefficient array as string.
        """

        return str(self.sequence()) + ': ' + str(self.function)

    def pretty_print(self) -> str:
        """
        cf.pretty_print()

        Prints the operator sequence m and corresponding coefficient function f(ell; m).

        Returns
        -------
        str
            Operator sequence and human-readable quasi-polynomial.
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

    Examples
    --------
    >>> collection = FunctionCollection({1: -1, 2: 0, 3: 1})
    ... ((1,), ()) in collection
    False

    Checks for the operator sequence m whether the function f(ell; m) is already calculated.

    >>> collection[((1,), ())] = QuasiPolynomial.new({0: [1, 2], 3: [4, 5]})

    Saves the function f(ell; m) if it is not already saved.

    >>> collection[((1,), ())]

    Returns for the operator sequence m the function f(ell; m) or None.

    >>> print(collection)
    ["((1,), ()): [(0, ['1', '2']), (3, ['4', '5'])]"]

    Prints the collection.
    """

    def __init__(self, translation: Dict[int, Energy]) -> None:
        self.__private_collection = dict()
        self.translation = translation

    def __contains__(self, sequence: Sequence) -> bool:
        """
        sequence in FunctionCollection

        Checks for the operator sequence m whether the function f(ell; m) is already calculated.

        Parameters
        ----------
        sequence : Tuple[Tuple[int,...],...]
            The operator sequence m identifying the function f(ell; m).

        Returns
        -------
        bool
            `True` if function already in collection.
        """

        return sequence_to_key(sequence) in self.__private_collection

    def __setitem__(self, sequence: Sequence, function: QuasiPolynomial) -> None:
        """
        FunctionCollection[sequence] = function

        Saves the function f(ell; m) if it is not already saved.

        Parameters
        ----------
        sequence : Tuple[Tuple[int,...],...]
            The operator sequence m identifying the function f(ell; m).
        function : QuasiPolynomial
            The function f(ell; m).
        """

        if sequence not in self:
            self.__private_collection[sequence_to_key(sequence)] = function

    def __getitem__(self, sequence: Sequence) -> Optional[CoefficientFunction]:
        """
        FunctionCollection[sequence]

        Returns for the operator sequence m the function f(ell; m) or None.

        Parameters
        ----------
        sequence : Tuple[Tuple[int,...],...]
            The operator sequence m.

        Returns
        -------
        CoefficientFunction
            Function or none (if not in collection).
        """

        if sequence in self:
            return CoefficientFunction(sequence, self.__private_collection[sequence_to_key(sequence)])
        else:
            return None

    def keys(self) -> List[Sequence]:
        """
        FunctionCollection.keys()

        Returns all calculated operator sequences m.

        Returns
        -------
        List[Tuple[Tuple[int,...],...]]
            List of operator sequences in collection.
        """

        return [key_to_sequence(key) for key in self.__private_collection.keys()]

    def __str__(self) -> str:
        """
        print(FunctionCollection)

        Prints the collection.

        Returns
        -------
        str
            Collection as string.
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
            Collection as human-readable string.
        """

        output = str()
        for key in self.keys():
            output = output + self[key].pretty_print() + '\n'
        return output


def sequence_to_key(sequence: Sequence) -> Sequence:  
    """
    vector_to_key(sequence)

    Converts the operator sequence m into the key.

    Parameters
    ----------
    sequence
        Operator sequence whose function is to be calculated.

    Returns
    -------
    Tuple[Tuple[int,...],...]
        Key.
    """
    
    # To reduce memory consumption, the key could be of type `int`
    return sequence


def key_to_sequence(key: Sequence) -> Sequence:
    """
    key_to_sequence(key)

    Converts the key into the operator sequence m.

    Parameters
    ----------
    key
        Key whose function is to be calculated.

    Returns
    -------
    Tuple[Tuple[int,...],...]
        Operator sequence.
    """

    return key


def sequence_to_indices(sequence: Sequence, translation: Dict[int, Energy]) -> Indices[Energy]:
    """
    sequence_to_indices(sequence, translation)

    Converts the operator sequence into the indices of the operator sequence m
    using the translation dictionary.

    Parameters
    ----------
    sequence
        Operator sequence whose function is to be calculated.
    translation
        Dictionary translating operator indices to energy values.

    Returns
    -------
    Tuple[Tuple[Union[int,float,Fraction,complex],...],...]
        Operator sequence indices.
    """

    return tuple(tuple((translation[e] for e in s)) for s in sequence)


def calc(sequence: Sequence, collection: FunctionCollection, translation: Dict[int, Energy],
         signum_func: Union[
            Callable[[Indices[Energy_real], Indices[Energy_real]], int], Callable[
                [Indices[Union[complex, Expr]], Indices[Union[complex, Expr]]], Union[complex, Expr]]],
         energy_func: Callable[[Indices[Energy]], Energy],
         band_diagonality_func: Callable[[Indices[Energy]], bool]) -> QuasiPolynomial:
    """
    calc(sequence)

    Returns or calculates the function f(ell; m) corresponding to the operator sequence m.

    Parameters
    ----------
    sequence
        Operator sequence whose function is to be calculated.
    collection
        Function collection where the result is to be stored in.
    translation
        Dictionary translating operator indices to energy values.
    signum_func
        Signum function to be used (normal, broad or complex).
    energy_func
        Energy function to be used (normal or broad).
    band_diagonality_func
        Band diagonality function to be used (normal, broad or complex).

    Returns
    -------
    QuasiPolynomial
        Calculated function f(ell; m).
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
            f1 = calc(partition[0], collection, translation, signum_func, energy_func, band_diagonality_func)
            f2 = calc(partition[1], collection, translation, signum_func, energy_func, band_diagonality_func)
            # Translate the operator sequences into its indices.
            m = sequence_to_indices(sequence, translation)
            m1 = sequence_to_indices(s1, translation)
            m2 = sequence_to_indices(s2, translation)
            # Only calculate non-vanishing contributions to the integrand.
            if (band_diagonality_func(m1) and band_diagonality_func(m2)):
                integrand = integrand + exponential(m, m1, m2, energy_func) * signum_func(m1, m2) * f1 * f2
        result = integrand.integrate()
        # Insert the result into the collection.
        collection[sequence] = result
        return result


def trafo_calc(sequence: Sequence, trafo_collection: FunctionCollection, collection: FunctionCollection,
               translation: Dict[int, Energy],
               signum_func: Union[Callable[[Indices[Energy_real], Indices[Energy_real]], int], Callable[
                [Indices[Union[complex, Expr]], Indices[Union[complex, Expr]]], Union[complex, Expr]]],
               energy_func: Callable[[Indices[Energy]], Energy],
               band_diagonality_func: Callable[[Indices[Energy]], bool]) -> QuasiPolynomial:
    """
    trafo_calc(sequence)

    Calculates the function G(ell; m) corresponding to the operator sequence m.

    Parameters
    ----------
    sequence
        Operator sequence whose function is to be calculated.
    trafo_collection
        Function collection where the result is to be stored in.
    collection
        Function collection where the resulting f(ell; m) are to be stored in.
    translation
        Dictionary translating operator indices to energy values.
    signum_func
        Signum function to be used (normal, broad or complex).
    energy_func
        Energy function to be used (normal or broad).
    band_diagonality_func
        Band diagonality function to be used (normal, broad or complex).

    Returns
    -------
    QuasiPolynomial
        Calculated function G(ell; m).
    """

    # Check whether the function is already calculated.
    if trafo_collection[sequence] is not None:
        return trafo_collection[sequence].function
    else:
        m = sequence_to_indices(sequence, translation)
        # TODO: Why does `signum_func` gives a type checking error?
        integrand = (exponential(((), ()), ((), ()), m, energy_func) * signum_func(((), ()), m)
                     * calc(sequence, collection, translation, signum_func, energy_func,band_diagonality_func))
        partition_list = partitions(sequence)
        for partition in partition_list:
            # Rename the operator sequences.
            s1 = partition[0]
            s2 = partition[1]
            # Check whether the required functions are already calculated.
            g1 = trafo_calc(partition[0], trafo_collection, collection, translation, signum_func,
                            energy_func, band_diagonality_func)
            f2 = calc(partition[1], collection, translation, signum_func, energy_func, band_diagonality_func)
            # Translate the operator sequences into its indices.
            m1 = sequence_to_indices(s1, translation)
            m2 = sequence_to_indices(s2, translation)
            # Calculate the contributions to the integrand.
            integrand = integrand + exponential(((), ()), ((), ()), m2, energy_func) * signum_func(((), ()),
                                                                                                   m2) * g1 * f2
        result = integrand.integrate()
        # Insert the result into the collection.
        trafo_collection[sequence] = result
        return result
