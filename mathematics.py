from quasiPolynomial import QuasiPolynomial
import numpy as np
from typing import List, Tuple


def energy(indices: List) -> int:
    """
    energy(indices)

    Returns the sum M(m) of operator sequence indices.

        Returns
        -------
        int
    """

    return sum(indices)


def signum(indices1: List, indices2: List) -> int:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

        Returns
        -------
        int
    """

    return np.sign(energy(indices1)) - np.sign(energy(indices2))


def exponential(indices: List, indices1: List, indices2: List) -> QuasiPolynomial:
    """
    exponential(indices, indices1, indices2)

    Returns the exponential exp(- alpha x) with alpha = - (|M(sequence)| - |M(m1)| - |M(m2)|).

        Returns
        -------
        QuasiPolynomial
    """

    alpha = abs(energy(indices)) - abs(energy(indices1)) - abs(energy(indices2))
    coefficient_list = [[] for _ in range(- alpha)]
    coefficient_list.append([1])
    return QuasiPolynomial.new(coefficient_list)


def partitions(m: Tuple) -> List[Tuple[Tuple, Tuple]]:
    """
    partitions(sequence)

    Returns all partitions of the operator sequence m into (m1, m2).

        Returns
        -------
        List[Tuple[Tuple, Tuple]]
    """

    return [(m[:i+1], m[i+1:]) for i in range(len(m) - 1)]
