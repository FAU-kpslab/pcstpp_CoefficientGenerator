# Coefficient generator for pcst<sup>++</sup>

Coefficient generator for pcst<sup>++</sup> used in [L. Lenke, A. Schellenberger, K. P. Schmidt, "Series expansions in
closed and open quantum many-body systems with multiple quasiparticle types", (2023) arXiv:2302.01000 [cond-mat.str-el]
](https://doi.org/10.48550/arXiv.2302.01000)

## Table of contents

- [Coefficient generator for pcst++](#coefficient-generator-for-pcst)
  - [Table of contents](#table-of-contents)
  - [Description](#description)
  - [Instructions](#instructions)
    - [How to run this program](#how-to-run-this-program)
    - [Notation in code and documentation](#notation-in-code-and-documentation)
    - [Configuration](#configuration)
      - [Further remarks:](#further-remarks)
    - [Functionality](#functionality)
    - [Output format](#output-format)
    - [Examples](#examples)  
  - [Module specification](#module-specification)
  - [Authors](#authors)

> Both program and documentation rely on the mathematical notation used in our paper
> [arXiv:2302.01000](https://doi.org/10.48550/arXiv.2302.01000).

## Description

This program was implemented to calculate the transformations for our paper
[arXiv:2302.01000](https://doi.org/10.48550/arXiv.2302.01000).
This includes the coefficients, not the subsequent linked-cluster expansion.

## Instructions

These are instructions for the coefficient generator for pcst<sup>++</sup>, not for the method pcst<sup>++</sup> itself.
For that, we refer to our paper [arXiv:2302.01000](https://doi.org/10.48550/arXiv.2302.01000).

### How to run this program

This program can be run from the command line.
```Shell
$ python main.py
```

It accepts with multiple optional arguments:
- `-h` to show the help message.
- `-t` to calculate the coefficients of the transformation.
- `-f [FILE]` to use an external file for configuration. Default value is `config.yaml`.
- `-c` to generate an exemplary config file instead of running the program.
- Without any argument to run the program with the hard-coded default configuration.

### Notation in code and documentation

| Name                                       | Meaning                                    | (Equation in [arXiv:2302.01000](https://doi.org/10.48550/arXiv.2302.01000)) |
|--------------------------------------------|--------------------------------------------| --------------------------------------------------------------------------- | 
| Index of operator sequence                 | $\textbf{m}$                               | (9)                                                                         | 
| Operator sequence                          | Unique name of $T_\textbf{m}$              | (9)                                                                         |
| Key of operator sequence                   | Unique key of $T_\textbf{m}$, same as name |                                                                             |
| Coefficient function                       | $F(\ell; \textbf{m})$                      | (9)                                                                         |
| Starting condition                         | $F(0; \textbf{m})$                         | (13)                                                                        |
| Coefficient                                | $C_\textbf{m} = F(\infty; \textbf{m})$     |                                                                             |
| Transformation                             | $\mathcal{S}(\infty)$                      | (16)                                                                        |
| Coefficient function of the transformation | $G(\ell; \textbf{m})$                      | (16)                                                                        |
| Standard signum function                   | $\textrm{sgn}$ for Hermitian operators     |                                                                             |
| complex signum function                    | $\textrm{sgn}$ for non-Hermitian operators | (5)                                                                         |
| Broad signum function                      | $\textrm{sgn}_D$                           | (17)                                                                        |
| Delta                                      | $D$                                        | (17)                                                                        |

### Configuration

The descriptive configuration file reads as follows and can be found in `config-example.yaml`. 
```YAML
# This is an exemplary config file. Following the comments in this file, you can modify it for your purpose and rerun
# the program with the flag '--file'. For more infos, use flag '--help'.

# Enter the total order.
max_order: 4
# Give a unique name (integer) to every operator, so that you can distinguish them. You can take the operator index as
# its name, provided that they are unique. The operators can be separated in arbitrarily different lists which marks
# them as groups whose operators commute pairwise with those of other groups. 
operators: [[-1, -2, -3, -4, -5], [1, 2, 3, 4, 5]]
# Enter the operator indices, i.e., the unperturbed energy differences caused by the operators.
# The indices can be of type integer, float, Fraction (e.g. '1/2') or complex (e.g. (1+2j)).
indices:
  -1: -2
  -2: -1
  -3: 0
  -4: 1
  -5: 2
  1: -2
  2: -1
  3: 0
  4: 1
  5: 2
# Manually insert the solution for the coefficient functions with non-vanishing starting condition as string, integer,
# float or complex (e.g. (1+2j)).
starting_conditions:
  ((-1,), ()): '1'
  ((-3,), ()): '1'
  ((-5,), ()): '1'
  ((), (1,)): '-1'
  ((), (3,)): '-1'
  ((), (5,)): '-1'
  ((-2,), (4,)): '1'
  ((-4, -2), ()): '-1/2'
  ((), (2, 4)): '-1/2'
# Introduce band-diagonality, i.e., write down the largest absolute value of possible index sums occurring in the
# starting conditions.
max_energy: 2
# Optionally, specify the delta value for the 'broad signum' function, i.e., half of the width of the 0 level.
delta: 0
```

#### Further remarks:

- Specify the desired maximal order with `max_order`; all orders below that will also be calculated automatically.
- The number of different needed operator sequences reduces if some operators commute.
  Operators are therefore stored as a list of lists to indicate that.
- You might want to have multiple perturbation operators $T_m$ with the same index $m$.
  For that reason, every operator has a unique name that is an integer.
- The dictionary `indices` has these names as keys and the actual indices as values.
- For every coefficient function the program calculates, it assumes the starting condition $F(0, \mathbf{m}) = 0$.
  Every function that has a different starting condition has to be inserted into the function collection manually via
  `starting_conditions`.
- It is necessary to indicate non-zero coefficient functions $F(0, \mathbf{m}) = 0$ via `starting_conditions`, even when
  you want to calculate the coefficient functions of the transformation $G(\ell, \mathbf{m})$.
- TODO: Something about band-diagonality.
- If you don't want to use the broad signum function you can either specify `delta: 0` or don't specify delta at all.
- After running, the program will save the used input in a new file `config.yaml` for you to store with your result.
  This will overwrite any existing file `config.yaml`.
  You can use this file to check that you configured what you wanted to configure.

Depending on the data types you use in the configuration, the program will use different functions:

| Data types                 | Exact or up to precision `1e-09` | Signum function |
|----------------------------|----------------------------------|-----------------|
| `int`, `fraction.Fraction` | Exact                            | Normal / broad  |
| `float`                    | Up to precision `1e-09`          | Normal / broad  |
| `complex`                  | Up to precision `1e-09`          | Complex         |
| `sympy.core.expr.Expr`     | Exact                            | Complex         |

- `sympy.core.expr.Expr` are assumed to be exact.
- Numbers of type `fraction.Fraction` should be given as strings, so that they aren't evaluated as floats.
- If at least one value in `indices` or `starting_conditions` in the configuration is of type `sympy.core.expr.Expr`,
  all of them will be converted into that type.
- Elif at least one of those values are of type `complex`, all of them will be converted into that type.
- Elif at least one of those values are of type `float`, all of them will be converted into that type.
- Every value in `indices` in the configuration that is of type `str` will be converted into type `fraction.Fraction`.

### Functionality

This program solves differential equations recursively for coefficient functions.
It stores all calculated coefficient functions in a data type based on a dictionary.
The keys in this dictionary are the indices of operator sequences; the values are the coefficient functions.
After all necessary coefficient functions are computed, it prints the resulting coefficients to a file.

It takes advantage of the fact that the coefficient functions are quasi-polynomials.
$$\sum_{\mu \geq 0} P_\mu(\ell; \mathbf{m}) e^{- \mu \ell}$$
Quasi-polynomials are stored in a data type based on a dictionary.
The keys in this dictionary are the $\mu$ (Alpha); the values are the corresponding polynomials.
Polynomials $\sum_n c_n \ell^n$ are stored in a data type based on a list.
The position in this list is the power $n$; the values are the coefficients $c_n$.

More detailed descriptions can be found in the [module specifications](#module-specification). 

### Output format

The coefficients are stored in the file `result.txt`; a new run will overwrite any existing file `result.txt`.
Every line in this file corresponds to an operator sequence, sorted according to their order.
The structure of these lines is as follows.
```txt
order mk ... m2 m1 coefficient
```
- The first integer `order` indicates the order of the operator sequence.
- The next `order` integers are the inverted operator sequence.
  Inverted means that the left-most operator in the result file is the right-most in the actual operator sequence.
- The rest of the line contains the coefficient depending on the used type.
  - `int`, `fraction.Fraction`: Coefficient is displayed as two integer numbers `numerator denominator`.
    It can happen that the first-order coefficient is displayed as integer and not as fraction.
  - `float`, `complex`, `sympy.core.expr.Expr`: Coefficient is displayed as it is.

> If the used type is `fraction.Fraction`, but the starting condition for some coefficient functions is an integer, then
> the result file can contain the coefficient for those cases as integer and not as fraction.

> It the used type is `sympy.core.expr.Expr`, the coefficient is not simplified in the result file.
> Our suggestion is to simplify the needed coefficients automatically with another script.
> We found that `sympy` quickly reaches its limits, even when using `sympy.cancel()`.
> In such cases we converted the `sympy` expressions to `mathematica` and simplified them there.

### Examples

Config files for all exemplary models in our paper [arXiv:2302.01000](https://doi.org/10.48550/arXiv.2302.01000) are in the folder [config-examples](./config-examples/).
This includes the [staggered transverse-field Ising model](TODO),
the [non-Hermitian staggered transverse-field Ising model](./config-examples/config-NHTFIM.yaml),
the [spin-one transverse-field Ising model with single-ion anisotropy](TODO)
and the [Dissipative transverse-field Ising model](./config-examples/config-DTFIM.yaml).


## Module specification

- [Index](./docs/index.html)
  - [main](./docs/main.html)
  - [coefficientFunction](./docs/coefficientFunction.html)
  - [quasiPolynomial](./docs/quasiPolynomial.html)
  - [mathematics](./docs/mathematics.html)

## Authors

Lea Lenke and Andreas Schellenberger
