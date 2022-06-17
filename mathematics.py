from functools import reduce
from itertools import product, combinations
import operator

from quasiPolynomial import QuasiPolynomial
import numpy as np
from typing import List, Tuple


def energy(indices: Tuple[Tuple[int,...],...]) -> int:
    """
    energy(indices)

    Returns the sum M(m) of operator sequence indices.

        Returns
        -------
        int
    """
    return sum(reduce(operator.add, indices))


def signum(indices1: Tuple[Tuple[int,...],...], indices2: Tuple[Tuple[int,...],...]) -> int:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

        Returns
        -------
        int
    """

    return np.sign(energy(indices1)) - np.sign(energy(indices2))


def exponential(indices: Tuple[Tuple[int,...],...], indices1: Tuple[Tuple[int,...],...], indices2: Tuple[Tuple[int,...],...]) -> QuasiPolynomial:
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


def partitions(sequence: Tuple[Tuple[int,...],...]) -> List[Tuple[Tuple[Tuple[int,...],...],Tuple[Tuple[int,...],...]]]:
    """
    partitions(sequence)

    Returns all partitions of the operator sequence (m, n, o, ...) into ((m1, n1, o1, ...), (m2, n2, o2, ...)).

        Returns
        -------
        List[Tuple[Tuple[Tuple[int,...],...],Tuple[Tuple[int,...],...]]]
    """
    # TODO: Why we have to look at all possible partitions, especially those where
    # we have different amount of terms on the left side for commuting Hilbert spaces?
    # Should this depend on the starting conditions e.g. is this needed for the Dicke model?
    partitions = [[(s[:i],s[i:]) for i in range(len(s) + 1)] for s in sequence]
    # skip edge cases of completely empty left or right side
    valid_partitions = list(product(*partitions))[1:-1]
    return [(tuple(l[0] for l in pr),tuple(r[1] for r in pr)) for pr in valid_partitions]
