from itertools import product, combinations

from quasiPolynomial import QuasiPolynomial
import numpy as np
from typing import List, Tuple


def energy(indices: Tuple[List, List]) -> int:
    """
    energy(indices)

    Returns the sum M(m) of operator sequence indices.

        Returns
        -------
        int
    """

    indices_left = indices[0]
    indices_right = indices[1]
    return sum(indices_left) + sum(indices_right)


def signum(indices1: Tuple[List, List], indices2: Tuple[List, List]) -> int:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

        Returns
        -------
        int
    """

    return np.sign(energy(indices1)) - np.sign(energy(indices2))


def exponential(indices: Tuple[List, List], indices1: Tuple[List, List], indices2: Tuple[List, List]) -> QuasiPolynomial:
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


def partitions(sequence: Tuple[Tuple, Tuple]) -> List[Tuple[Tuple[Tuple, Tuple], Tuple[Tuple, Tuple]]]:
    """
    partitions(sequence)

    Returns all partitions of the operator sequence (m, n) into ((m1, n1), (m2, n2)).

        Returns
        -------
        List[Tuple[Tuple[Tuple, Tuple], Tuple[Tuple, Tuple]]]
    """

    sequence_left = sequence[0]
    sequence_right = sequence[1]
    partition_left = [(sequence_left[:i], sequence_left[i:]) for i in range(len(sequence_left) + 1)]
    partition_right = [(sequence_right[:i], sequence_right[i:]) for i in range(len(sequence_right) + 1)]
    return [((left1, right1), (left2, right2)) for left1, left2 in partition_left for right1, right2 in partition_right
            if (((left1, right1) != ((), ())) & ((left2, right2) != ((), ())))]
