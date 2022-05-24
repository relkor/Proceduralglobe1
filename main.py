import subprocess
import statistics

import numpy
from matplotlib import pyplot as plt


def compile_c_program(input_files, output, compiler="gcc", optimization_level="-O3"):
    tmp = subprocess.run([compiler, "-o", output, optimization_level, *input_files, "-fopenmp"])
    return tmp.returncode



def run_c_program(program_name, nthreads,n,matrixsize=1 ):
    tmp = subprocess.run([program_name, str(nthreads), str(n),str(matrixsize), ], stdout=subprocess.PIPE)
    return tmp.stdout.decode()


input_files = ["bigCalc.c"]
output = "./main"
compile_c_program(input_files, output)

print(run_c_program(output, 8,0, 8))


