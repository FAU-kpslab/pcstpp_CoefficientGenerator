"""
Defines basic mathematical functions for calculating the coefficient functions,
like the unperturbed eigenvalue change $M(m)$ and the occurring exponents in the
modified coefficient functions.
"""

from fractions import Fraction
from functools import reduce
from itertools import product
import operator

from quasiPolynomial import QuasiPolynomial, are_close, is_zero
import numpy as np
import sympy as sym
from typing import Callable, List, Tuple, Union, TypeVar, cast
from sympy.core.expr import Expr

Coeff = Union[int, float, Fraction, complex, Expr]
Energy = Coeff
Energy_real = Union[int, float, Fraction]
E = TypeVar('E', bound=Energy)
E_real = TypeVar('E_real', bound=Energy_real)
Indices = Tuple[Tuple[E, ...], ...]
Sequence = Tuple[Tuple[int, ...], ...]


def energy(indices: Indices[E]) -> E:
    """
    energy(indices)

    Returns the sum M(m) of operator sequence indices.

    Parameters
    ----------
    indices
        Operator sequence indices.

    Returns
    -------
    Union[int, float, Fraction, complex, Expr]
        Sum M(m).
    """

    return sum(reduce(operator.add, indices), cast(E, 0))


def energy_broad(indices: Indices[Energy_real], delta: Energy_real) -> Energy_real:
    """
    energy_broad(indices)

    Returns the *broadened* sum M(m)*theta(|M(m)|-delta) of operator sequence indices, where theta is the Heaviside
    step function.

    Parameters
    ----------
    indices
        Operator sequence indices.

    Returns
    -------
    Union[int, float, Fraction]
        Sum M(m)*theta(|M(m)|-delta).
    """

    e = energy(indices)
    # also return 0 if `abs(e)` is very close to `delta` to deal with
    # floating errors
    return e if abs(e) > delta and not are_close(abs(e), delta) else 0

def signum(indices1: Indices[Energy_real], indices2: Indices[Energy_real]) -> int:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

    Parameters
    ----------
    indices1
        Operator sequence indices whose sgn(M(m1)) is to be added.
    indices2
        Operator sequence indices whose sgn(M(m2)) is to be subtracted.

    Returns
    -------
    int
        Difference of signum functions.
    """

    return int(np.sign(energy(indices1))) - int(np.sign(energy(indices2)))


def signum_broad(indices1: Indices[Energy_real], indices2: Indices[Energy_real], delta: Energy_real) -> int:
    """
    signum_broad(indices1, indices2, delta)

    Returns the prefactor sgn_`delta`(M(m1)) - sgn_delta(M(m2)), where sgn_d is the broadened signum function with
    sgn_d(x) = 0 for |x| <= d and sgn_d(x) = sgn(x), otherwise.

    Parameters
    ----------
    indices1
        Operator sequence indices whose sgn_d(M(m1)) is to be added.
    indices2
        Operator sequence indices whose sgn_d(M(m2)) is to be subtracted.

    Returns
    -------
    int
        Difference of broad signum functions.
    """

    return int(np.sign(energy_broad(indices1, delta))) - int(np.sign(energy_broad(indices2, delta)))


def signum_complex(indices1: Indices[Union[complex, Expr]], indices2: Indices[Union[complex, Expr]]) -> Union[complex,
                                                                                                              Expr]:
    """
    signum_complex(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)) with the definition sgn(z) = z / |z|.

    Parameters
    ----------
    indices1
        Operator sequence indices whose complex sgn(M(m1)) is to be added.
    indices2
        Operator sequence indices whose complex sgn(M(m2)) is to be subtracted.

    Returns
    -------
    Union[complex, Expr]
        Difference of complex signum functions.
    """

    complex_sgn = lambda z: 0 if is_zero(z) else z.conjugate() / abs(z)
    return complex_sgn(energy(indices1)) - complex_sgn(energy(indices2))


def exponential(indices: Indices[Energy],
                indices1: Indices[Energy],
                indices2: Indices[Energy],
                energy_func: Callable[[Indices[Energy]], Energy]) -> QuasiPolynomial:
    """
    exponential(indices, indices1, indices2)

    Returns the exponential exp(- alpha x) with alpha = - (|M(m)| - |M(m1)| - |M(m2)|).

    Parameters
    ----------
    indices
        Operator sequence indices whose |M(m)| is to be added in the exponential.
    indices1
        First operator sequence indices whose |M(m1)| is to be subtracted in the exponential.
    indices2
        Second operator sequence indices whose |M(m2)| is to be subtracted in the exponential.
    energy_func
        Energy function to be used (normal or broad).

    Returns
    -------
    QuasiPolynomial
        Resulting quasi-polynomial.
    """

    alpha = abs(energy_func(indices)) - abs(energy_func(indices1)) - abs(energy_func(indices2))
    return QuasiPolynomial.new({-alpha: [1]})


def partitions(sequence: Sequence) -> List[Tuple[Sequence, Sequence]]:
    """
    partitions(sequence)

    Returns all partitions of the operator sequence (m, n, o, ...) into ((m1, n1, o1, ...), (m2, n2, o2, ...)).

    Parameters
    ----------
    sequence
        Operator sequence to be partitioned.

    Returns
    -------
    List[Tuple[Tuple[Tuple[int,...],...],Tuple[Tuple[int,...],...]]]
        List of partitions.
    """

    partitions = [[(s[:i], s[i:]) for i in range(len(s) + 1)] for s in sequence]
    # skip edge cases of completely empty left or right side
    valid_partitions = list(product(*partitions))[1:-1]
    return [(tuple(l[0] for l in pr), tuple(r[1] for r in pr)) for pr in valid_partitions]


def band_diagonality(indices: Indices[Energy_real], max_energy: Energy_real) -> bool:
    return abs(energy(indices)) <= abs(max_energy)

def band_diagonality_broad(indices: Indices[Energy_real], max_energy:Energy_real, delta:Energy_real) -> bool:
    # TODO: Is `delta * len(indices)` the correct factor or has it to depend on the
    # complete indices list
    return abs(energy(indices)) <= abs(max_energy) + delta * len(indices)

def band_diagonality_complex(indices: Indices[Energy]) -> bool:
    return True