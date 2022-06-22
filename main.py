import yaml
from yaml.loader import SafeLoader

import argparse

import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import energy
from itertools import product


def main():
    my_parser = argparse.ArgumentParser(description='Use pCUT to block-diagonalize a Lindbladian or a Hamiltonian with '
                                                    'two particle types')
    my_parser.add_argument('-t', '--trafo', action='store_true', help='calculate the transformation directly')
    my_config = my_parser.add_mutually_exclusive_group()
    my_config.add_argument('-f', '--file', nargs='?', const='config.yml', default=None,
                           help='pass configuration using the config file "config.yml" '
                                'or a custom one given as an argument')
    my_config.add_argument('-c', '--config', action='store_true',
                           help='Writes an exemplary config file to "config.yml" without performing '
                                'any calculations.')
    args = my_parser.parse_args()

    if args.file != None:
        print('You have decided to use a config file, titled "{}".'.format(args.file))
        config_file = open(args.file, "r")
        config = yaml.load(config_file, Loader=SafeLoader)
        max_order = config['max_order']
        operators = list(config['operators'])
        translation = config['indices']
        starting_conditions = config['starting_conditions']
        max_energy = config['max_energy']
        config_file.close()
    else:
        print("You have decided to use the default config values.")
        # Enter the total order.
        max_order = 4
        # Give a unique name (integer) to every operator, so that you can distinguish them. You can take the operator
        # index as its name, provided that they are unique. The operators can separated in arbitrarily different lists 
        # which marks them as groups whose operators commute pairwise with those of other groups.
        operators = [[-1, -2, -3, -4, -5], [1, 2, 3, 4, 5]]
        # Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the operators.
        # In Lea's case, enter the indices of the operators prior to transposition.
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
    
    if not args.config:
        # Prepare the coefficient function storage.
        collection = coefficientFunction.FunctionCollection(translation)
        for sequence in starting_conditions:
            collection[eval(sequence)] = qp.new([[starting_conditions[sequence]]])

        operators_all = [operator for operator_space in operators for operator in operator_space]

        if not args.trafo:
            for order in range(max_order + 1):
                print('Starting calculations for order ' + str(order) + '.')
                # TODO: This version is slower as needed as we do not use the arbitrary order of the commuting operators
                sequences = set(product(operators_all, repeat=order))
                for sequence in sequences:
                    sequence_sorted = tuple(tuple([s for s in sequence if s in o_s]) for o_s in operators)
                    indices = coefficientFunction.sequence_to_indices(sequence_sorted, translation)
                    # Make use of block diagonality.
                    if energy(indices) == 0:
                        coefficientFunction.calc(sequence_sorted, collection, translation, max_energy)
            # print(collection.pretty_print())
            print('Starting writing process.')
            # Write the results in a file.
            with open("result.txt", "w") as result:
                for sequence in collection.keys():
                    # Only return the block-diagonal operator sequences.
                    if energy(coefficientFunction.sequence_to_indices(sequence, translation)) == 0:
                        resulting_constant = collection[sequence].function.get_constant()
                        # Only return the non-vanishing operator sequences.
                        if resulting_constant != 0:
                            # Reverse the operator sequences, because the Solver thinks from left to right.
                            inverted_sequence = [str(operator) for s in sequence for operator in s[::-1]]
                            # Return 'order' 'sequence' 'numerator' 'denominator'.
                            output = [str(sum([len(seq) for seq in sequence]))] + inverted_sequence + [
                                str(resulting_constant.numerator), str(resulting_constant.denominator)]
                            print(' '.join(output), file=result)
                result.close()

        if args.trafo:
            # Prepare the trafo coefficient function storage.
            trafo_collection = coefficientFunction.FunctionCollection(translation)
            trafo_collection[tuple([()]*len(operators))] = qp.new([['1']])

            for order in range(max_order + 1):
                print('Starting calculations for order ' + str(order) + '.')
                sequences = set(product(operators_all, repeat=order))
                for sequence in sequences:
                    sequence_sorted = tuple(tuple([s for s in sequence if s in o_s]) for o_s in operators)
                    coefficientFunction.trafo_calc(sequence_sorted, trafo_collection, collection, translation, max_energy)
            # print(collection.pretty_print())
            print('Starting writing process.')
            # Write the results in a file.
            with open("result.txt", "w") as result:
                for sequence in trafo_collection.keys():
                    resulting_constant = trafo_collection[sequence].function.get_constant()
                    # Only return the non-vanishing operator sequences.
                    if resulting_constant != 0:
                        # Reverse the operator sequences, because the Solver thinks from left to right.
                        inverted_sequence = [str(operator) for s in sequence for operator in s[::-1]]
                        # Return 'order' 'sequence' 'numerator' 'denominator'.
                        output = [str(sum([len(seq) for seq in sequence]))] + inverted_sequence + [
                            str(resulting_constant.numerator), str(resulting_constant.denominator)]
                        print(' '.join(output), file=result)
                result.close()

    # Generate the config file.
    config_file = open("config.yml", "w")
    print('---', file=config_file)
    print("# This is an exemplary config file. If you use this program for the first time, you can also "
          "specify everything step\n"
          "# by step in the command line; the program will then generate the correct config file.\n",
          file=config_file)
    print("# Enter the total order.", file=config_file)
    print('max_order: ' + str(max_order), file=config_file)
    print("# Give a unique name (integer) to every operator, so that you can distinguish them. You can take the "
          "operator index as\n"
          "# its name, provided that they are unique. The operators can separated in arbitrarily different lists "
          "which marks them\n" 
          "# as groups whose operators commute pairwise with those of other groups. ",
          file=config_file)
    print('operators: ' + str(list(operators)), file=config_file)
    print("# Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the "
          "operators. In Lea's\n"
          "# case, enter the indices of the operators prior to transposition.", file=config_file)
    print('indices:', file=config_file)
    for key in translation.keys():
        print('  ' + str(key) + ': ' + str(translation[key]), file=config_file)
    print("# Manually insert the solution for the coefficient functions with non-vanishing starting condition as "
          "strings.",
          file=config_file)
    print('starting_conditions:', file=config_file)
    for sequence in starting_conditions:
        print('  ' + sequence + ": '" + str(starting_conditions[sequence] + "'"), file=config_file)
    print("# Introduce band-diagonality, i.e., write down the largest sum of indices occurring in the starting "
          "conditions.", file=config_file)
    print('max_energy: ' + str(max_energy), file=config_file)
    print('...', file=config_file)
    config_file.close()

    if not args.config:
        print('The calculations are done. Your coefficient file is "result.txt". If you want to keep it, store it under a '
            'different name before executing the program again.')
        print('The used configuration is found in "config.yml", you can store that together with the results.')
    else:
        print('The default configuration file is found in "config.yml". It will be overwritten when rerunning the program.')

if __name__ == '__main__':
    main()
