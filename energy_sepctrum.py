from math import pi
import numpy as np
import typing
from numpy.core.numeric import NaN
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

"""
With this program we want to plot the energy spectrum of a partilce in a box
with parity symmetric boundaries gamma_+ = gamma_- = gamma.
"""

box_L = 1
nearly_pi = 3

def even_positive(x, gamma):
    return gamma/x - np.tan(x*box_L/2)

def odd_positive(x, gamma):
    return gamma/x + 1/np.tan(x*box_L/2)

def find_roots(func, num_req_roots=6, gamma_value=1, startvalue=0.5, max_tries_per_root=50, root_epsilon=0.0001):
    results = []
    guess = startvalue
    num_tries = 0
    while len(results) < num_req_roots:
        while num_tries < max_tries_per_root:
            newroot = float(fsolve(func, guess, args=gamma_value))
            isnewroot = True
            for root in results:
                if abs(root-newroot) < root_epsilon:
                    isnewroot = False
            if isnewroot:
                results.append(newroot)
                guess = newroot+1
                break
            guess += 1
            num_tries += 1
    return results

# For the x-koordinates of the plot
xaxis_array = -np.linspace(-nearly_pi/2, nearly_pi/2, 100)
# To recieve the gamma factors times 2 over L for the calculation of the energy eigenvalues
gamma_array = np.tan(xaxis_array)*2./box_L 

highest_energy_level = 5

final_array = []
for gamma in gamma_array:
    even_results = find_roots(even_positive, num_req_roots=int((highest_energy_level+1)/2), gamma_value=gamma)
    odd_results = find_roots(odd_positive, num_req_roots=int((highest_energy_level+1)/2), gamma_value=gamma)
    temp_tupellst = list(zip(even_results, odd_results))
    temp_final_array = []
    for tupel in temp_tupellst:
        temp_final_array.append(tupel[0])
        temp_final_array.append(tupel[1])
    final_array.append(temp_final_array)

energy_array_list = np.power(np.transpose(final_array), 2)

plt.plot(xaxis_array, energy_array_list[0], label="n=0")
plt.plot(xaxis_array, energy_array_list[1], label="n=1")
plt.plot(xaxis_array, energy_array_list[2], label="n=2")
plt.plot(xaxis_array, energy_array_list[3], label="n=3")
plt.plot(xaxis_array, energy_array_list[4], label="n=4")

plt.title("Energy spectrum: gamma_+ = gamma_-")
plt.xlabel("arctan(2*gamma/L)")
plt.ylabel("Energy")

plt.legend()

plt.show()