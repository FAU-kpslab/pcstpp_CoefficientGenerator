import coefficientFunction
from quasiPolynomial import QuasiPolynomial as qp


def main():
    collection = coefficientFunction.FunctionCollection()
    collection[(-2,)] = qp.new([[1]])
    collection[(0,)] = qp.new([[1]])
    collection[(2,)] = qp.new([[1]])
    collection[(1, 0)] = qp.zero()
    print(collection)


if __name__ == '__main__':
    main()
