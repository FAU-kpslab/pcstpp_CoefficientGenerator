from fractions import Fraction
from functools import reduce
from itertools import product, combinations
import operator
from cmath import isclose

from quasiPolynomial import QuasiPolynomial, are_close
import numpy as np
from typing import Callable, List, Tuple, Union, TypeVar

Coeff = Union[int,float,Fraction,complex]
Energy = Coeff
Energy_real = Union[int,float,Fraction]
E = TypeVar('E', bound= Energy)
Indices = Tuple[Tuple[E,...],...]
Sequence = Tuple[Tuple[int,...],...]

def energy(indices: Indices[Energy]) -> Energy:
    """
    energy(indices)

    Returns the sum M(m) of operator sequence indices.

        Returns
        -------
        Union[int, float, Fraction, complex]
    """
    return sum(reduce(operator.add, indices))

def energy_broad(indices: Indices[Energy_real], delta:Energy_real) -> Energy_real:
    """
    energy_broad(indices)

    Returns the 'broadened' sum M(m)*theta(|M(m)|-`delta`) of operator sequence 
    indices, where theta is the Heaviside step function.

        Returns
        -------
        Union[int, float, Fraction]
    """
    e = energy(indices)
    # also return 0 if `abs(e)` is very close to `delta` to deal with
    # floating errors
    return e if abs(e)>delta and not are_close(abs(e),delta) else 0

def signum(indices1: Indices[Energy_real], indices2: Indices[Energy_real]) -> int:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

        Returns
        -------
        int
    """
    # TODO: Maybe change to math.copysign https://stackoverflow.com/questions/1986152/why-doesnt-python-have-a-sign-function
    return int(np.sign(energy(indices1))) - int(np.sign(energy(indices2)))

def signum_broad(indices1: Indices[Energy_real], indices2: Indices[Energy_real], delta:Energy_real) -> int:
    """
    signum_broad(indices1, indices2, delta)

    Returns the prefactor sgn_`delta`(M(m1)) - sgn_`delta`(M(m2)), where sgn_d is
    the broadened signum function with sgn_d (x) = 0 for |x| <= d and sgn_d (x)=sgn (x),
    otherwise.

        Returns
        -------
        int
    """
    return int(np.sign(energy_broad(indices1,delta))) - int(np.sign(energy_broad(indices2,delta)))
   
def signum_complex(indices1: Indices[complex], indices2: Indices[complex]) -> complex:
    """
    signum_complex(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)) with the definition sgn(z) = z / |z|
    as used in the Ferkinghoff, Uhrig paper.

        Returns
        -------
        complex
    """
    complex_sgn = lambda z: 0 if np.abs(z) == 0 else np.conj(z)/np.abs(z)
    return complex_sgn(energy(indices1)) - complex_sgn(energy(indices2))

def exponential(indices: Indices[Energy], 
                indices1: Indices[Energy], 
                indices2: Indices[Energy],
                energy_func: Callable[[Indices[Energy]],Energy]) -> QuasiPolynomial:
    """
    exponential(indices, indices1, indices2)

    Returns the exponential exp(- alpha x) with alpha = - (|M(sequence)| - |M(m1)| - |M(m2)|).

        Returns
        -------
        QuasiPolynomial
    """

    alpha = abs(energy_func(indices)) - abs(energy_func(indices1)) - abs(energy_func(indices2))
    return QuasiPolynomial.new({-alpha:[1]})

def partitions(sequence: Sequence) -> List[Tuple[Sequence,Sequence]]:
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
    # -> Maybe add an option to restrict partitions depending on the model
    partitions = [[(s[:i],s[i:]) for i in range(len(s) + 1)] for s in sequence]
    # skip edge cases of completely empty left or right side
    valid_partitions = list(product(*partitions))[1:-1]
    return [(tuple(l[0] for l in pr),tuple(r[1] for r in pr)) for pr in valid_partitions]
