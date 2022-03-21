from coefficientFunction import CoefficientFunction as cf
from quasiPolynomial import QuasiPolynomial as qp


def main():
    initialm2 = cf([-2], qp.new([[1]]))
    initial0 = cf([0], qp.new([[1]]))
    initialp2 = cf([2], qp.new([[1]]))
    resultp20 = cf([2, 0], (initialp2.function * initial0.function).integrate())
    print(resultp20.function.pretty_print())
    resultp2m2 = cf([2, -2], (qp.new([[], [], [], [], [1]]) * 2 * initialp2.function * initialm2.function).integrate())
    print(resultp2m2.function.pretty_print())


if __name__ == '__main__':
    # TODO: What does __name__ do?
    main()