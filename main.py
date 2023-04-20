from fractions import Fraction
from numpy import iscomplex
import yaml
from yaml.loader import SafeLoader

import argparse

import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import Energy, Coeff, Expr, energy, energy_broad, signum, signum_broad, signum_complex
from itertools import product, chain
from typing import cast, Dict, Union, Sequence
from sympy.parsing.sympy_parser import parse_expr
import sympy as sym
from sympy import sympify

# Standard Symbol for symbolic calculations
a = sym.Symbol("a", positive=True)


def main(raw_args: Union[Sequence[str],None] = None):
    my_parser = argparse.ArgumentParser(description='Use pcst++ to block-diagonalize a Lindbladian or a Hamiltonian'
                                                    'with multiple particle types')
    my_parser.add_argument('-t', '--trafo', action='store_true', help='calculate the transformation directly')
    my_config = my_parser.add_mutually_exclusive_group()
    my_config.add_argument('-f', '--file', nargs='?', const='config.yaml', default=None,
                           help='pass configuration using the config file "config.yaml" '
                                'or a custom one given as an argument')
    my_config.add_argument('-c', '--config', action='store_true',
                           help='Writes an exemplary config file to "config.yaml" without performing '
                                'any calculations.')
    args = my_parser.parse_args(raw_args)

    if args.file is not None:
        print('You have decided to use a config file, titled "{}".'.format(args.file))
        config_file = open(args.file, "r")
        config = yaml.load(config_file, Loader=SafeLoader)
        max_order = config['max_order']
        operators = list(config['operators'])
        translation = config['indices']
        starting_conditions = config['starting_conditions']
        max_energy = config['max_energy']
        delta = config['delta'] if 'delta' in config else 0
        config_file.close()
        # postprocessing of complex values in starting_conditions 
        for (k, v) in starting_conditions.items():
            if isinstance(v, str) and "j" in v:
                starting_conditions[k] = complex(v)
        # postprocessing of Expr values in translation
        for (k, v) in translation.items():
            if isinstance(v, str) and ("a" in v or "I" in v):
                translation[k] = parse_expr(v, local_dict={"a": a})
        # postprocessing of complex values in translation
        for (k, v) in translation.items():
            if isinstance(v, str) and "j" in v:
                translation[k] = complex(v)
        # postprocessing of Expr value for max_energy
        if isinstance(max_energy, str) and ("a" in max_energy or "I" in max_energy):
            max_energy = parse_expr(max_energy, local_dict={"a": a})
    else:
        print("You have decided to use the default config values.")
        # Enter the total order.
        max_order = 4
        # Give a unique name (integer) to every operator, so that you can distinguish them. You can take the operator
        # index as its name, provided that they are unique. The operators can be separated in arbitrarily different
        # lists which marks them as groups whose operators commute pairwise with those of other groups.
        operators = [[-1, -2, -3, -4, -5], [1, 2, 3, 4, 5]]
        # Enter the operator indices, i.e., the unperturbed energy differences caused by the operators.
        # The indices can be of type integer, float, Fraction (e.g. '1/2') or complex (e.g. (1+2j)).
        translation = {
            -1: -2,
            -2: -1,
            -3: 0,
            -4: 1,
            -5: 2,
            1: -2,
            2: -1,
            3: 0,
            4: 1,
            5: 2
        }
        # Manually insert the solution for the coefficient functions with non-vanishing starting condition as strings.
        starting_conditions = {
            '((-1,), ())': '1',
            '((-3,), ())': '1',
            '((-5,), ())': '1',
            '((), (1,))': '-1',
            '((), (3,))': '-1',
            '((), (5,))': '-1',
            '((-2,), (4,))': '1',
            '((-4, -2), ())': '-1/2',
            '((), (2, 4))': '-1/2'}
        # Introduce band-diagonality, i.e., write down the largest sum of indices occurring in the starting conditions.
        max_energy = 2
        delta = 0
    # Defining the concrete types
    starting_conditions = cast(Dict[str, Union[Coeff, str]], starting_conditions)
    translation = cast(Dict[int, Energy], translation)

    # Calculation is skipped if solely a exemplary config file shall be created
    if not args.config:
        # Assuming expressions of type `Expr` to be exact
        # If needed, convert all starting_conditions to the same type
        if len([v for v in chain(starting_conditions.values(), translation.values()) if isinstance(v, Expr)]) > 0:
            type_to_use = Expr
            print("Forcing all coefficients to type {}".format(type_to_use))
            for k, v in starting_conditions.items():
                try:
                    starting_conditions[k] = sympify(v)
                except ValueError:
                    raise ValueError("Incompatible type in starting conditions at {}. Only use one datatype.".format(k))
            for k, v in translation.items():
                try:
                    translation[k] = sympify(v)
                except ValueError:
                    raise ValueError("Incompatible type in index {}. Only use one datatype.".format(k))
        elif len([v for v in chain(starting_conditions.values(), translation.values()) if
                  isinstance(v, (complex, float))]) > 0:
            type_to_use = (complex if len([v for v in chain(starting_conditions.values(), translation.values())
                                           if isinstance(v, complex)]) > 0 else float)
            print("Forcing all coefficients to type {}".format(type_to_use))
            # Only changing starting_conditions as it is sufficient to get
            # a consistent coefficient file.
            for k, v in starting_conditions.items():
                try:
                    starting_conditions[k] = type_to_use(v)
                except ValueError:
                    raise ValueError("Incompatible type in starting conditions at {}. Only use one datatype.".format(k))

        for k, v in translation.items():
            # If string, the value has to be converted to Fraction
            if isinstance(v, str):
                translation[k] = Fraction(v)
        # Typechecking and conversion done

        # Prepare the coefficient function storage.
        collection = coefficientFunction.FunctionCollection(translation)
        for sequence in starting_conditions:
            collection[eval(sequence)] = qp.new_integer([[starting_conditions[sequence]]])

        # Prepare the trafo coefficient function storage.
        if args.trafo:
            trafo_collection = coefficientFunction.FunctionCollection(translation)
            trafo_collection[tuple([()] * len(operators))] = qp.new_integer([['1']])

        # Choosing the signum for the generator
        if delta > 0:
            print("Using the broad signum function.")
            signum_func = lambda l, r: signum_broad(l, r, delta=delta)
            energy_func = lambda i: energy_broad(i, delta=delta)
        # check if any translation value has a non-vanishing imaginary part
        elif len([v for v in translation.values() if iscomplex(v)]) > 0:
            print("Using the complex signum function.")
            signum_func = signum_complex
            energy_func = energy
        elif len([v for v in translation.values() if isinstance(v, Expr)]) > 0:
            print("Using the complex signum function for symbolic calculations.")
            signum_func = signum_complex
            energy_func = energy
        else:
            print("Using the standard signum function.")
            signum_func = signum
            energy_func = energy

        # Combine all elements in `operators` to one list
        operators_all = [operator for operator_space in operators for operator in operator_space]

        for order in range(max_order + 1):
            print('Starting calculations for order ' + str(order) + '.')
            sequences = set(product(operators_all, repeat=order))
            for sequence in sequences:
                sequence_sorted = tuple(tuple([s for s in sequence if s in operator_space]) for operator_space in operators)
                if not args.trafo:
                    indices = coefficientFunction.sequence_to_indices(sequence_sorted, translation)
                    # Make use of block diagonality.
                    if coefficientFunction.is_zero(energy_func(indices)):
                        # As the band diagonality is only fulfilled up to a multiple of delta add + delta * max_order
                        # TODO: For the broad signum, max_energy should depend on the specific order used in
                        #  one calculation -> implement order-dependent max_energy in `calc` in `coefficientFunction.py`
                        # TODO: Maybe even make max_energy completely automatic?
                        coefficientFunction.calc(sequence_sorted, collection, translation,
                                                 max_energy + delta * max_order, signum_func, energy_func)
                else:
                    coefficientFunction.trafo_calc(sequence_sorted, trafo_collection, collection, translation,
                                                   max_energy + delta * max_order, signum_func, energy_func)

        print('Starting writing process.')
        # Write the results in a file.
        with open("result.txt", "w") as result:
            act_collection = trafo_collection if args.trafo else collection
            for sequence in act_collection.keys():
                # Only return the block-diagonal operator sequences.
                if args.trafo or coefficientFunction.is_zero(
                        energy_func(coefficientFunction.sequence_to_indices(sequence, translation))):
                    resulting_constant = act_collection[sequence].function.get_constant()
                    # Only return the non-vanishing operator sequences.
                    if not coefficientFunction.is_zero(resulting_constant):
                        # Reverse the operator sequences, because the Solver thinks from left to right.
                        inverted_sequence = [str(operator) for s in sequence for operator in s[::-1]]
                        if isinstance(resulting_constant, Fraction):
                            # Return 'order' 'sequence' 'numerator' 'denominator'.
                            output = [str(sum([len(seq) for seq in sequence]))] + inverted_sequence + [
                                str(resulting_constant.numerator), str(resulting_constant.denominator)]
                        else:
                            # Return 'order' 'sequence' 'coefficient'.
                            output = [str(sum([len(seq) for seq in sequence]))] + inverted_sequence + [
                                str(resulting_constant)]
                        print(' '.join(output), file=result)
            result.close()

    # Generate the config file.
    config_file = open("config.yaml", "w")
    print('---', file=config_file)
    print("# This is an exemplary config file. Following the comments in this file, you can modify it for your "
          "purpose and rerun\n"
          "# the program with the flag '--file'. For more infos, use flag '--help'.\n",
          file=config_file)
    print("# Enter the total order.", file=config_file)
    print('max_order: ' + str(max_order), file=config_file)
    print("# Give a unique name (integer) to every operator, so that you can distinguish them. You can take the "
          "operator index as\n"
          "# its name, provided that they are unique. The operators can be separated in arbitrarily different lists "
          "which marks\n"
          "# them as groups whose operators commute pairwise with those of other groups. ",
          file=config_file)
    print('operators: ' + str(list(operators)), file=config_file)
    print("# Enter the operator indices, i.e., the unperturbed energy differences caused by the operators.\n"
          "# The indices can be of type integer, float, Fraction (e.g. '1/2') or complex (e.g. (1+2j)).",
          file=config_file)
    print('indices:', file=config_file)
    for key in translation.keys():
        print('  ' + str(key) + ': ' + str(translation[key]), file=config_file)
    print("# Manually insert the solution for the coefficient functions with non-vanishing starting condition as "
          "string, integer,\n"
          "# float or complex (e.g. (1+2j)).", file=config_file)
    print('starting_conditions:', file=config_file)
    for sequence in starting_conditions:
        if isinstance(starting_conditions[sequence], str):
            print('  ' + sequence + ": '" + str(starting_conditions[sequence]) + "'", file=config_file)
        else:
            print('  ' + sequence + ": " + str(starting_conditions[sequence]), file=config_file)
    print("# Introduce band-diagonality, i.e., write down the largest absolute value of possible index sums occurring "
          "in the\n"
          "# starting conditions.", file=config_file)
    print('max_energy: ' + str(max_energy), file=config_file)
    print("# Optionally, specify the delta value for the 'broad signum' function, i.e., half of the width of "
          "the 0 level.", file=config_file)
    print('delta: ' + str(delta), file=config_file)
    print('...', file=config_file)
    config_file.close()

    if not args.config:
        print('The calculations are done. Your coefficient file is "result.txt". If you want to keep it, store it under'
              ' a different name before executing the program again.')
        print('The used configuration is found in "config.yaml", you can store that together with the results.')
    else:
        print(
            'The default configuration file is found in "config.yaml". It will be overwritten when rerunning the program'
            '.')


if __name__ == '__main__':
    main()
