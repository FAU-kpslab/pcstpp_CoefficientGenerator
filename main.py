import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import energy
from itertools import product


def main():
    # Give a unique name to every operator, so that you can distinguish them. You can take the operator index as its
    # name, provided that they are unique. If you want to pass the results to the solver, you need to take integers.
    # Otherwise, you can also take strings.
    operators = ('-2', '0', '2')
    # Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the operators.
    translation = dict([('-2', -2), ('0', 0), ('2', 2)])
    # Introduce band-diagonality.
    max_energy = 2

    # Prepare the coefficient function storage.
    collection = coefficientFunction.FunctionCollection(translation, max_energy)

    # Manually insert the solution for the coefficient functions with non-vanishing starting condition.
    collection[('-2',)] = qp.new([[1]])
    collection[('0',)] = qp.new([[1]])
    collection[('2',)] = qp.new([[1]])

    max_order = 4
    for order in range(max_order + 1):
        # Calculate all possible operator sequences of length order.
        for sequence in set(product(operators, repeat=order)):
            indices = coefficientFunction.sequence_to_indices(sequence, translation)
            # Make use of block diagonality.
            if energy(indices) == 0:
                collection[sequence] = coefficientFunction.differential_equation(sequence, collection, translation,
                                                                                 max_energy)
    # print(collection.pretty_print())
    # Print the block-diagonal operator sequences yielding the effective operator.
    with open("result.txt", "w") as result:
        for sequence in collection.keys():
            if energy(coefficientFunction.sequence_to_indices(sequence, translation)) == 0:
                if collection[sequence].function.get_constant() != 0:
                    print(str(sequence) + ': ' + str(collection[sequence].function.get_constant()), file=result)
        result.close()


if __name__ == '__main__':
    main()
