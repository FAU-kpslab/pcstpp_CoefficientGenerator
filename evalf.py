import argparse

import sympy
from sympy.parsing.sympy_parser import parse_expr

a = sympy.symbols('a', positive=True)

my_parser = argparse.ArgumentParser(description='Simplify the sympy expressions in the coefficient file.')
my_parser.add_argument('-f', '--file', nargs='?', const='result.txt', default=None, help='pass the coefficient file '
                                                                                         '"result.txt" or a custom one '
                                                                                         'given as an argument')
my_parser.add_argument('-N', '--accuracy', nargs='?', const=15, default=None, help='pass the accuracy in decimal digits')
args = my_parser.parse_args()

if args.file is not None:
    print('You have decided to use the coefficient file "{}".'.format(args.file))
    file = open(args.file, "r")
else:
    print('You have decided to use the coefficient file "result.txt".')
    file = open("result.txt", "r")
    
if args.accuracy is not None:
    print('You have decided to use an accuracy of {} decimal digits.'.formal(args.accuracy))
    N = args.accuracy
else:
    print('You have decided to use the custom accuracy of 15 decimal digits.')
    N = 15

lines = file.readlines()
file.close()

with open("result_evalf.txt", "w") as new_file:
    for line in lines:
        order = int(line[0])
        new_file.write((" ".join(line.split(' ')[0:order + 1])) + " " + str(
            sympy.N(parse_expr(" ".join(line.split(' ')[order + 1:]), local_dict={"a": a}), N)) + "\n")
