from fractions import Fraction
from functools import reduce
from itertools import product, combinations
import operator

from quasiPolynomial import QuasiPolynomial
import numpy as np
from typing import List, Tuple, Union


def energy(indices: Tuple[Tuple[Union[int,float,Fraction],...],...]) -> Union[int,float,Fraction]:
    """
    energy(indices)

    Returns the sum M(m) of operator sequence indices.

        Returns
        -------
        Union[int, float, Fraction]
    """
    return sum(reduce(operator.add, indices))


def signum(indices1: Tuple[Tuple[Union[int,float,Fraction],...],...], 
           indices2: Tuple[Tuple[Union[int,float,Fraction],...],...]) -> Union[int,float]:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

        Returns
        -------
        Union[int, float]
    """
    # TODO: Maybe change to math.copysign https://stackoverflow.com/questions/1986152/why-doesnt-python-have-a-sign-function
    return np.sign(energy(indices1)) - np.sign(energy(indices2))

def signum_broad(indices1: Tuple[Tuple[int,...],...], indices2: Tuple[Tuple[int,...],...], delta:int) -> int:
    """
    signum_broad(indices1, indices2, delta)

    Returns the prefactor sgn_`delta`(M(m1)) - sgn_`delta`(M(m2)), where sgn_d is
    the broadened signum function with sgn_d (x) = 0 for |x| <= d and sgn_d (x)=sgn (x),
    otherwise.

        Returns
        -------
        int
    """
    # np.int64(.) as np.heaviside outputs floating value
    return (np.sign(energy(indices1))*np.int64(np.heaviside(np.abs(energy(indices1))-delta,0)) 
            - np.sign(energy(indices2))*np.int64(np.heaviside(np.abs(energy(indices2))-delta,0)))


def signum_complex(indices1: Tuple[Tuple[complex,...],...], indices2: Tuple[Tuple[complex,...],...]) -> complex:
    """
    signum_complex(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)) with the definition sgn(z) = z / |z|
    as used in the Ferkinghoff, Uhrig paper.

        Returns
        -------
        complex
    """
    # TODO: Generalization needed: `energy` has to be able to output 
    # `complex`-typed variables (also in typing)
    # TODO: Problem that floating point precision is used. This leads to round-off
    # errors when comparing to the normal signum function
    complex_sgn = lambda z: 0 if np.abs(z) == 0 else z/np.abs(z)
    return complex_sgn(energy(indices1)) - complex_sgn(energy(indices2))

def exponential(indices: Tuple[Tuple[Union[int,float,Fraction],...],...], 
                indices1: Tuple[Tuple[Union[int,float,Fraction],...],...], 
                indices2: Tuple[Tuple[Union[int,float,Fraction],...],...]) -> QuasiPolynomial:
    """
    exponential(indices, indices1, indices2)

    Returns the exponential exp(- alpha x) with alpha = - (|M(sequence)| - |M(m1)| - |M(m2)|).

        Returns
        -------
        QuasiPolynomial
    """

    alpha = abs(energy(indices)) - abs(energy(indices1)) - abs(energy(indices2))
    return QuasiPolynomial.new({-alpha:[1]})


def partitions(sequence: Tuple[Tuple[Union[int,float,Fraction],...],...]) -> List[Tuple[Tuple[Tuple[Union[int,float,Fraction],...],...],Tuple[Tuple[Union[int,float,Fraction],...],...]]]:
    """
    partitions(sequence)

    Returns all partitions of the operator sequence (m, n, o, ...) into ((m1, n1, o1, ...), (m2, n2, o2, ...)).

        Returns
        -------
        List[Tuple[Tuple[Tuple[Union[int,float,Fraction],...],...],Tuple[Tuple[Union[int,float,Fraction],...],...]]]
    """
    # TODO: Why we have to look at all possible partitions, especially those where
    # we have different amount of terms on the left side for commuting Hilbert spaces?
    # Should this depend on the starting conditions e.g. is this needed for the Dicke model?
    # -> Maybe add an option to restrict partitions depending on the model
    partitions = [[(s[:i],s[i:]) for i in range(len(s) + 1)] for s in sequence]
    # skip edge cases of completely empty left or right side
    valid_partitions = list(product(*partitions))[1:-1]
    return [(tuple(l[0] for l in pr),tuple(r[1] for r in pr)) for pr in valid_partitions]
