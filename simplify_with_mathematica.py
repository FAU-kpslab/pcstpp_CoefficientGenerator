import argparse

import wolframclient
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl

import sympy
from sympy.parsing.sympy_parser import parse_expr

wolfram_session = WolframLanguageSession('~/.local/lib/python3.7/site-packages/wolframclient')

# Error message: Kernel not found...

print(wolfram_session.evaluate(wl.StringReverse('abc')))

#a = sympy.symbols('a', positive=True)
#
#my_parser = argparse.ArgumentParser(description='Simplify the sympy expressions in the coefficient file.')
#my_parser.add_argument('-f', '--file', nargs='?', const='result.txt', default=None, help='pass the coefficient file '
#                                                                                         '"result.txt" or a custom one '
#                                                                                         'given as an argument')
#args = my_parser.parse_args()
#
#if args.file is not None:
#    print('You have decided to use the coefficient file "{}".'.format(args.file))
#    file = open(args.file, "r")
#else:
#    print('You have decided to use the coefficient file "result.txt".')
#    file = open("result.txt", "r")
#
#lines = file.readlines()
#file.close()
#
#with open("result_simplified.txt", "w") as new_file:
#    for line in lines:
#        order = int(line[0])
#        new_file.write((" ".join(line.split(' ')[0:order + 1])) + " " + str(
#            sympy.cancel(parse_expr(" ".join(line.split(' ')[order + 1:]), local_dict={"a": a}))) + "\n")
#