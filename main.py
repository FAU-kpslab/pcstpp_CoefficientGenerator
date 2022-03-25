import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import energy
from itertools import product


def main():
    # Give a unique name to every operator, so that you can distinguish them. You can take the operator index as its
    # name, provided that they are unique. If you want to pass the results to the solver, you need to take integers.
    # Otherwise, you can also take strings.
    operators_left = (8, 9, 10, 11, 12)
    operators_right = (18, 19, 20, 21, 22)
    # Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the operators. In
    # Lea's case, enter the indices of the operators prior to transposition.
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
    # Introduce band-diagonality.
    max_energy = 2

    # Prepare the coefficient function storage.
    collection = coefficientFunction.FunctionCollection(translation, max_energy)

    # Manually insert the solution for the coefficient functions with non-vanishing starting condition.
    collection[((8,), ())] = qp.new([[1]])
    collection[((10,), ())] = qp.new([[1]])
    collection[((12,), ())] = qp.new([[1]])
    collection[((), (18,))] = qp.new([[-1]])
    collection[((), (20,))] = qp.new([[-1]])
    collection[((), (22,))] = qp.new([[-1]])
    collection[((9,), (21,))] = qp.new([[1]])
    collection[((11, 9), ())] = qp.new([[-1/2]])
    collection[((), (19, 21))] = qp.new([[-1/2]])

    # TODO: The fractions data type overflows in max_order 8.
    max_order = 8
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
    # Print the block-diagonal operator sequences yielding the effective operator.
    with open("result.txt", "w") as result:
        for sequence in collection.keys():
            if energy(coefficientFunction.sequence_to_indices(sequence, translation)) == 0:
                resulting_constant = collection[sequence].function.get_constant()
                if resulting_constant != 0:
                    # Invert operator sequence, because the Solver thinks from left to right.
                    inverted_sequence = [str(operator) for operator in sequence[0][::-1]] + [str(operator) for operator
                                                                                             in sequence[1][::-1]]
                    output = [str(len(sequence[0]) + len(sequence[1]))] + inverted_sequence + [
                        str(resulting_constant.numerator), str(resulting_constant.denominator)]
                    print(' '.join(output), file=result)
        result.close()


if __name__ == '__main__':
    main()
