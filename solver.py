import argparse

import subprocess

import sympy
from sympy import I, sqrt

a = sympy.symbols('a',positive=True)

my_parser = argparse.ArgumentParser(description='Use the solver with sympy expressions.')
my_parser.add_argument('-f', '--file', nargs='?', const='result.txt', default=None, help='pass the coefficient file "result.txt" or a custom one given as an argument')
my_parser.add_argument('-s', '--solver', nargs='?', const='./', default=None, help='pass the folder containing the solver if not in the same folder')
args = my_parser.parse_args()

if args.file != None:
	print('You have decided to use the coefficient file "{}".'.format(args.file))
	file = open(args.file, "r")
else:
	print('You have decided to use the coefficient file result.txt.')
	file = open("result.txt", "r")
	
if args.solver != None:
	print('The path of the solver is "{}Solver".'.format(args.solver))
else:
	print('The path of the solver is "./Solver".')

lines = file.readlines()

with open("result_solver_temp.txt", "w") as solver_results:
	for line in lines:
		with open("temp.txt", "w") as temp_coefficient:
			order = int(line[0])
			temp_coefficient.write((" ".join(line.split(' ')[0:order+1])) + " 1 1")
			result = subprocess.run("cd {}; ./Solver".format(args.solver), capture_output=True, text=True, shell=True)
			solver_results.write(result.stdout)