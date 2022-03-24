import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp
from mathematics import energy
from itertools import product


def main():
    # Give a unique name to every operator, so that you can distinguish them. You can take the operator index as its
    # name, provided that they are unique. If you want to pass the results to the solver, you need to take integers.
    # Otherwise, you can also take strings.
    operators_left = ('-2.id', '-1.id', '0.id', '1.id', '2.id')
    operators_right = ('id.-2', 'id.-1', 'id.0', 'id.1', 'id.2')
    # Enter the operator indices. In Andi's case, enter the unperturbed energy differences caused by the operators.
    translation = {
        '-2.id': -2,
        '-1.id': -1,
        '0.id': 0,
        '1.id': 1,
        '2.id': 2,
        'id.-2': -2,
        'id.-1': -1,
        'id.0': 0,
        'id.1': 1,
        'id.2': 2
    }
    # Introduce band-diagonality.
    max_energy = 2

    # Prepare the coefficient function storage.
    collection = coefficientFunction.FunctionCollection(translation, max_energy)

    # Manually insert the solution for the coefficient functions with non-vanishing starting condition.
    collection[(('-2.id',), ())] = qp.new([[1]])
    collection[(('0.id',), ())] = qp.new([[1]])
    collection[(('2.id',), ())] = qp.new([[1]])
    collection[((), ('id.-2',))] = qp.new([[-1]])
    collection[((), ('id.0',))] = qp.new([[-1]])
    collection[((), ('id.2',))] = qp.new([[-1]])
    collection[(('-1.id',), ('id.1',))] = qp.new([[1]])
    collection[(('1.id', '-1.id'), ())] = qp.new([[-1/2]])
    collection[((), ('id.-1', 'id.1'))] = qp.new([[-1/2]])

    # TODO: The fractions data type overflows in max_order 8.
    max_order = 8
    for order in range(max_order + 1):
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
    # Print the block-diagonal operator sequences yielding the effective operator.
    with open("TCDissipationO6.txt", "w") as result:
        for sequence in collection.keys():
            if energy(coefficientFunction.sequence_to_indices(sequence, translation)) == 0:
                resulting_constant = collection[sequence].function.get_constant()
                if resulting_constant != 0:
                    # Invert operator sequence, because the Solver thinks from left to right.
                    inverted_sequence = [operator for operator in sequence[0][::-1]] + [operator for operator in
                                                                                        sequence[1][::-1]]
                    output = [str(len(sequence[0]) + len(sequence[1]))] + inverted_sequence + [
                        str(resulting_constant.numerator), str(resulting_constant.denominator)]
                    print(' '.join(output), file=result)
        result.close()


if __name__ == '__main__':
    main()
