# Calculate the large N limit SYK entropy, S/N, by numerically solving the
# Schwinger Dyson equations, and then carrying out numerical integrations as
# suggested in Maldacena, Stanford 'Comments on the SYK model' Appendix G.

import argparse
import scipy.integrate as integrate

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from SYK_functions import *

# Define initial guess for two point function: usually taken to be free theory
# two point function
def G_input(x):
    return 1 / 2 * np.sign(x)

def calculate_entropy(dt, q, temp, max_iterations):
    # Variance of couplings is given by 'J_squared * 3!/N^3'
    J_squared = 1

    t0 = -1 / temp
    t0 = round(t0, 0)  # rounding since we want dt to divide t0

    t, G, w, S, Sf = G_SD(t0, dt, G_input, q, max_iterations)

    a = (1/2) * np.log(2)
    b = (1/2) * sum(np.log(1 + Sf[1::2]/(1j * w[1::2])))
    c = -t0/2 * integrate.simps(G[(len(G)//2):].real * S[(len(G)//2):].real - (J_squared/q)*G[(len(G)//2):].real**q,
                                t[(len(t)//2):])

    result1 = (a + b - c).real
    result2 = np.sqrt(J_squared) * (-t0)/q * integrate.simps(G[(len(G)//2):].real**q, t[(len(t)//2):])

    return result1 - result2

def fermion_int(value):
    fermions = int(value)
    if fermions < 2 or (fermions % 2) != 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid fermion count")
    return fermions

parser = argparse.ArgumentParser(description="Calculate the large N limit SYK entropy.")
parser.add_argument("--timestep", metavar="DT", type=float, required=True,
        help="resolution of time for calculation")
parser.add_argument("--fermions", metavar="Q", type=fermion_int, required=True,
        help="number of fermions (even integer ≥2)")
parser.add_argument("--temperature", metavar="TEMP", type=float, required=True,
        help="temperature for calculation")
parser.add_argument("--iterations", metavar="MAXITER", default=20, type=int,
        help="maximum iterations to try")
parser.add_argument("--output", type=argparse.FileType('w'),
        help="file to store the calculated entropy")
args = parser.parse_args()

temp = 10 ** args.temperature
entropy = calculate_entropy(args.timestep, args.fermions, temp, args.iterations)
print(f"temp. = {temp}, entropy = {entropy}")
if args.output:
    args.output.write(f"{entropy}\n")
