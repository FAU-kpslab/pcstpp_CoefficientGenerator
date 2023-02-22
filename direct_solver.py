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
my_parser.add_argument('-p', '--path', nargs='?', const='./', default=None, help='pass the folder containing the '
                                                                                   'solver if not in the same folder')
my_parser.add_argument('-1', '--step1', action='store_true', help='just calculate the solver results separately, '
                                                                  'without combining them with the coefficients')
my_parser.add_argument('-b', '--direct', nargs='?', default=None, help='pass the number of bits')
args = my_parser.parse_args()

if args.file is not None:
    print('You have decided to use the coefficient file "{}".'.format(args.file))
    file = open(args.file, "r")
else:
    print('You have decided to use the coefficient file result.txt.')
    file = open("result.txt", "r")
    
variable_dict = {"a": a, "h0": h0}
    
if args.direct is not None:
    print('You have decided to use the direct solver with {} bits.'.format(args.direct))
    bit_number = int(format(args.direct))
    # Generate all variables k# and assign ket_binary to them. 
    for ket in range(2**bit_number):
        ket_binary = format(ket, '0' + str(bit_number) + 'b')
        exec("k" + str(ket) + " = sympy.symbols('k" + str(ket) + "')")
        variable_dict["k" + str(ket)] = eval("k" + str(ket))
        
if args.path is not None:
    print('The path of the solver is "{}Solver".\n'.format(args.path))
else:
    print('The path of the solver is "./Solver".\n')

coefficients = file.readlines()

with open("result_solver_temp.txt", "w") as solver_results:
    for sequence in coefficients:
        order = int(sequence[0])
        with open("temp.txt", "w") as temp_coefficient:
            temp_coefficient.write((" ".join(sequence.split(' ')[0:order + 1])) + " 1 1\n")
        result = subprocess.run(
            "cd {}; ./Solver".format(args.path), capture_output=True, text=True, shell=True).stdout.replace(
            '^', '**').replace('\n', '').replace(';', '').replace(' ', '').replace('|', '*|')
        solver_results.write(result + '\n')

print('The expectation values of the individual T-operator sequences can be found in "result_solver_temp.txt".')

if not args.step1:
    print('Now they are multiplied with their respective prefactor to obtain the final result.\n')
    intermediate_results = []

    with open("result_solver_temp.txt", "r") as solver_results:
        solver_results_lines = solver_results.readlines()
        for row in range(len(coefficients)):
            order = int(coefficients[row][0])
            prefactor = parse_expr(" ".join(coefficients[row].split(' ')[order + 1:]), variable_dict)
            # Check for empty lines
            if solver_results_lines[row] != '\n':
                expectation_value = solver_results_lines[row]
                for ket in range(2**bit_number):
                    ket_binary = format(ket, '0' + str(bit_number) + 'b')
                    expectation_value = expectation_value.replace('|' + ket_binary + '>', str(eval('k' + str(ket))))
                parsed_expectation_value = parse_expr(expectation_value, variable_dict)
                intermediate_results.append(prefactor * parsed_expectation_value)
    result = sum(intermediate_results)
    with open("result_solver.txt", "w") as result_file:
        result_for_file = ""
        # Loop through the different kets to display their prefactor
        for ket in range(2**bit_number):
            ket_binary = format(ket, '0' + str(bit_number) + 'b')
            partial_result = result
            # Set all other kets to zero
            for other_ket in range(2**bit_number):
                if not other_ket == ket:
                    partial_result = partial_result.subs(str(eval('k' + str(other_ket))), 0)
            if not partial_result == 0:
                result_for_file = result_for_file + '+(' + str(
                    partial_result.subs(eval('k' + str(ket)), 1)) + ')*|' + ket_binary + '>\n'
        result_file.write(result_for_file)
    print('The result can be found in "result_solver.txt".')

file.close()
