import yaml
from yaml.loader import SafeLoader

import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import energy
from itertools import product


def main():
    config = input("Do you want to use a config file? (y/n) ")

    if config == "y":
        print("You have decided to use a config file, titled 'config.yml'.")
        config_file = open("config.yml", "r")
        config = yaml.load(config_file, Loader=SafeLoader)
        max_order = config['max_order']
        operators_left = tuple(config['operators_left'])
        operators_right = tuple(config['operators_right'])
        translation = config['indices']
        starting_conditions = config['starting_conditions']
        max_energy = config['max_energy']
        config_file.close()
    else:
        print("You have decided to use the hard-coded config values.")
        # Enter the total order.
        max_order = 2
        # Give a unique name to every operator, so that you can distinguish them. You can take the operator index as its
        # name, provided that they are unique. The list 'operators_left' contains all operators on the left side of the
        # tensor product and the list 'operators_right' all operators on the right side of the tensor product.
        operators_left = (8, 9, 10, 11, 12)
        operators_right = (18, 19, 20, 21, 22)
        # Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the operators.
        # In Lea's case, enter the indices of the operators prior to transposition.
        translation = {
            8: -2,
            9: -1,
            10: 0,
            11: 1,
            12: 2,
            18: -2,
            19: -1,
            20: 0,
            21: 1,
            22: 2
        }
        # Manually insert the solution for the coefficient functions with non-vanishing starting condition.
        starting_conditions = {'((8,), ())': [[1]], '((10,), ())': [[1]], '((12,), ())': [[1]], '((), (18,))': [[-1]],
                               '((), (20,))': [[-1]], '((), (22,))': [[-1]], '((9,), (21,))': [[1]],
                               '((11, 9), ())': [[-1/2]], '((), (19, 21))': [[-1/2]]}
        # Introduce band-diagonality, i.e., write down the largest sum of indices occurring in the starting conditions.
        max_energy = 2

    # Prepare the coefficient function storage.
    collection = coefficientFunction.FunctionCollection(translation, max_energy)
    for sequence in starting_conditions:
        collection[eval(sequence)] = qp.new(starting_conditions[sequence])

    for order in range(max_order + 1):
        print('Starting calculations for order ' + str(order) + '.')
        for order_left in range(order + 1):
            order_right = order - order_left
            # Calculate all possible operator sequences with 'order_left' operators on the left of the tensor product.
            sequences_left = set(product(operators_left, repeat=order_left))
            sequences_right = set(product(operators_right, repeat=order_right))
            for sequence_left in sequences_left:
                for sequence_right in sequences_right:
                    sequence = (sequence_left, sequence_right)
                    indices = coefficientFunction.sequence_to_indices(sequence, translation)
                    # Make use of block diagonality.
                    if energy(indices) == 0:
                        collection[sequence] = coefficientFunction.differential_equation(sequence, collection,
                                                                                         translation, max_energy)
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
                    inverted_sequence = [str(operator) for operator in sequence[0][::-1]] + [str(operator) for operator
                                                                                             in sequence[1][::-1]]
                    # Return 'order' 'sequence' 'numerator' 'denominator'.
                    output = [str(len(sequence[0]) + len(sequence[1]))] + inverted_sequence + [
                        str(resulting_constant.numerator), str(resulting_constant.denominator)]
                    print(' '.join(output), file=result)
        result.close()

    # Generate the config file.
    config_file = open("config.yml", "w")
    print('---', file=config_file)
    print("# This is an exemplary config file. If you use this program for the first time, you can also "
          "specify everything directly\n"
          "# in the file 'main.py'; the program will then generate the correct config file.\n",
          file=config_file)
    print("# Enter the total order.", file=config_file)
    print('max_order: ' + str(max_order), file=config_file)
    print("# Give a unique name to every operator, so that you can distinguish them. You can take the operator index"
          " as its name,\n"
          "# provided that they are unique. The list 'operators_left' contains all operators on the left side of the "
          "tensor product\n"
          "# and the list 'operators_right' all operators on the right side of the tensor product.", file=config_file)
    print('operators_left: ' + str(list(operators_left)), file=config_file)
    print('operators_right: ' + str(list(operators_right)), file=config_file)
    print("# Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the "
          "operators. In Lea's\n"
          "# case, enter the indices of the operators prior to transposition.", file=config_file)
    print('indices:', file=config_file)
    for key in translation.keys():
        print('  ' + str(key) + ': ' + str(translation[key]), file=config_file)
    print("# Manually insert the solution for the coefficient functions with non-vanishing starting condition.",
          file=config_file)
    print('starting_conditions:', file=config_file)
    for sequence in starting_conditions:
        print('  ' + sequence + ': ' + str(collection[eval(sequence)].function), file=config_file)
    print("# Introduce band-diagonality, i.e., write down the largest sum of indices occurring in the starting"
          " conditions.", file=config_file)
    print('max_energy: ' + str(max_energy), file=config_file)
    print('...', file=config_file)
    config_file.close()

    print("The calculations are done. Your coefficient file is 'result.txt'. If you want to keep it, store it under a "
          "different name before executing the program again.")
    print("The used configuration is found in 'config.yml', you can store that together with the results.")


if __name__ == '__main__':
    main()
