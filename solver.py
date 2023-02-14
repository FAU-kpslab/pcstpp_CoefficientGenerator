import argparse

import subprocess

import sympy
from sympy import parse_expr

a = sympy.symbols('a', positive=True)
h0 = sympy.symbols('h0')

my_parser = argparse.ArgumentParser(description='Use the solver with sympy expressions.')
my_parser.add_argument('-f', '--file', nargs='?', const='result.txt', default=None, help='pass the coefficient file '
                                                                                         '"result.txt" or a custom one '
                                                                                         'given as an argument')
my_parser.add_argument('-s', '--solver', nargs='?', const='./', default=None, help='pass the folder containing the '
                                                                                   'solver if not in the same folder')
my_parser.add_argument('-1', '--step1', action='store_true', help='just calculate the solver results separately, '
                                                                  'without combining them with the coefficients')
args = my_parser.parse_args()

if args.file is not None:
    print('You have decided to use the coefficient file "{}".'.format(args.file))
    file = open(args.file, "r")
else:
    print('You have decided to use the coefficient file result.txt.')
    file = open("result.txt", "r")

if args.solver is not None:
    print('The path of the solver is "{}Solver".\n'.format(args.solver))
else:
    print('The path of the solver is "./Solver".\n')

coefficients = file.readlines()

with open("result_solver_temp.txt", "w") as solver_results:
    for sequence in coefficients:
        order = int(sequence[0])
        with open("temp.txt", "w") as temp_coefficient:
            temp_coefficient.write((" ".join(sequence.split(' ')[0:order + 1])) + " 1 1\n")
        result = subprocess.run("cd {}; ./Solver".format(args.solver), capture_output=True, text=True,
                                shell=True).stdout.replace('^', '**')
        if result[0] == ';':
            solver_results.write(result[2:])
        # Check for explicitly mentioned kets (and ignore them for now, i.e. add everything together).
        elif result.find('|') != -1:
            pos = result.find('|')
            solver_results.write(result[:pos] + '\n')
        else:
            solver_results.write(result)

print('The expectation values of the individual T-operator sequences can be found in "result_solver_temp.txt".')

if not args.step1:
    print('Now they are multiplied with their respective prefactor to obtain the final result.\n')
    intermediate_results = []

    with open("result_solver_temp.txt", "r") as solver_results:
        prefactors = solver_results.readlines()
        for row in range(len(coefficients)):
            order = int(coefficients[row][0])
            prefactor = parse_expr(" ".join(coefficients[row].split(' ')[order + 1:]), local_dict={"a": a})
            # Check for empty lines
            if prefactors[row] != '\n':
                expectation_value = parse_expr(prefactors[row], local_dict={"h0": h0})
                intermediate_results.append(sympy.simplify(prefactor) * expectation_value)
    print("The result reads:")
    print(sympy.simplify(sum(intermediate_results)))

file.close()
