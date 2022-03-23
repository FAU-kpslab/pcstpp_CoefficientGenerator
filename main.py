import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import energy
from itertools import product


def main():

    operators = ('-2', '0', '2')
    translation = dict([('-2', -2), ('0', 0), ('2', 2)])
    # TODO: Band-diagonality
    max_energy = 2

    collection = coefficientFunction.FunctionCollection(translation, max_energy)

    # Starting conditions:
    collection[('-2',)] = qp.new([[1]])
    collection[('0',)] = qp.new([[1]])
    collection[('2',)] = qp.new([[1]])

    order = 4
    # Calculate all possible operator sequences of desired order.
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
