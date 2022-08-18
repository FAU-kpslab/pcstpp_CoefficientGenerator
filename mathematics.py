from fractions import Fraction
from functools import reduce
from itertools import product
import operator

from quasiPolynomial import QuasiPolynomial, are_close, is_zero
import numpy as np
import sympy as sym
from typing import Callable, List, Tuple, Union, TypeVar, cast

Expr = sym.core.expr.Expr
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

        Returns
        -------
        Union[int, float, Fraction, complex, Expr]
    """

    return sum(reduce(operator.add, indices), cast(E, 0))


def energy_broad(indices: Indices[Energy_real], delta: Energy_real) -> Energy_real:
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
    return e if abs(e) > delta and not are_close(abs(e), delta) else 0

def energy_broad_expr(indices: Indices[Expr], delta: Expr) -> Expr:
    """
    energy_broad(indices, delta)

    Returns the 'broadened' sum M(m)*theta(|M(m)|-`delta`) of operator sequence 
    indices, where theta is the Heaviside step function using sympy functionality.

        Returns
        -------
        Expr
    """

    e = energy(indices)
    return sym.Piecewise((e,sym.Abs(e)>delta),(0,True))

def signum(indices1: Indices[Energy_real], indices2: Indices[Energy_real]) -> int:
    """
    signum(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)).

        Returns
        -------
        int
    """

    return int(np.sign(energy(indices1))) - int(np.sign(energy(indices2)))


def signum_broad(indices1: Indices[Energy_real], indices2: Indices[Energy_real], delta: Energy_real) -> int:
    """
    signum_broad(indices1, indices2, delta)

    Returns the prefactor sgn_`delta`(M(m1)) - sgn_`delta`(M(m2)), where sgn_d is
    the broadened signum function with sgn_d (x) = 0 for |x| <= d and sgn_d (x)=sgn (x),
    otherwise.

        Returns
        -------
        int
    """

    return int(np.sign(energy_broad(indices1, delta))) - int(np.sign(energy_broad(indices2, delta)))


def signum_broad_expr(indices1: Indices[Expr], indices2: Indices[Expr], delta: Expr) -> Expr:
    """
    signum_broad_expr(indices1, indices2, delta)

    Returns the prefactor sgn_`delta`(M(m1)) - sgn_`delta`(M(m2)) where sgn_d is
    the broadened signum function with sgn_d (x) = 0 for |x| <= d and sgn_d (x)=sgn (x)
    using sympy functionality.

        Returns
        -------
        Expr
    """
    expr_sgn = lambda z: sym.Piecewise((z.conjugate() / sym.Abs(z), sym.Abs(z)>delta),(0,True))
    return expr_sgn(energy(indices1)) - expr_sgn(energy(indices2))

# TODO: Update typing (exponent_zero)
def signum_complex(indices1: Indices[Union[complex, Expr]], indices2: Indices[Union[complex, Expr]]) -> Union[complex,
                                                                                                              Expr]:
    """
    signum_complex(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)) with the definition sgn(z) = z / |z|
    as used in the Ferkinghoff, Uhrig paper.

        Returns
        -------
        Union[complex, Expr]
    """

    complex_sgn = lambda z: 0 if is_zero(z) else z.conjugate() / abs(z)
    return complex_sgn(energy(indices1)) - complex_sgn(energy(indices2))

def signum_expr(indices1: Indices[Expr], indices2: Indices[Expr]) -> Expr:
    """
    signum_expr(indices1, indices2)

    Returns the prefactor sgn(M(m1)) - sgn(M(m2)) with the definition sgn(z) = z / |z|
    as used in the Ferkinghoff, Uhrig paper using sympy functionality.

        Returns
        -------
        Expr
    """

    expr_sgn = lambda z: sym.Piecewise((z.conjugate() / sym.Abs(z), sym.Ne(z,0)),(0,True))
    return expr_sgn(energy(indices1)) - expr_sgn(energy(indices2))


def exponential(indices: Indices[Energy],
                indices1: Indices[Energy],
                indices2: Indices[Energy],
                energy_func: Callable[[Indices[Energy]], Energy]) -> QuasiPolynomial:
    """
    exponential(indices, indices1, indices2)

    Returns the exponential exp(- alpha x) with alpha = - (|M(sequence)| - |M(m1)| - |M(m2)|).

        Returns
        -------
        QuasiPolynomial
    """

    alpha = abs(energy_func(indices)) - abs(energy_func(indices1)) - abs(energy_func(indices2))
    return QuasiPolynomial.new({-alpha: [1]})


def partitions(sequence: Sequence) -> List[Tuple[Sequence, Sequence]]:
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
    partitions = [[(s[:i], s[i:]) for i in range(len(s) + 1)] for s in sequence]
    # skip edge cases of completely empty left or right side
    valid_partitions = list(product(*partitions))[1:-1]
    return [(tuple(l[0] for l in pr), tuple(r[1] for r in pr)) for pr in valid_partitions]
