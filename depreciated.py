
from math import pi
import numpy as np
import typing
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

"""
With this program we want to plot the energy spectrum of a partilce in a box
with parity symmetric boundaries gamma_+ = gamma_- = gamma.
"""


box_L = 1
nearly_pi = 3.1415926535


def even_positive(x, gamma):
    return gamma/x - np.tan(x)

def odd_positive(x, gamma):
    return gamma/x + np.cot(x)

def zero_negative(x, gamma):
    return gamma/x + np.tanh(x)

def level_zero(x, gamma):
    if gamma > 0:
        return even_positive(x, gamma)
    elif gamma == 0:
        return 0
    elif gamma < 0:
        return zero_negative(x, gamma)
    else:
        raise ValueError("Something strange happened with the gamma in level_zero.")

def one_negative(x, gamma):
    return gamma/x + np.coth(x)

def level_one(x, gamma):
    if gamma > -2./box_L:
        return odd_positive(x, gamma)
    elif gamma == -2./box_L:
        return 0
    elif gamma < -2./box_L:
        return one_negative(x, gamma)
    else:
        raise ValueError("Something strange happened with the gamma in level_one.")

class EnergyLevel():
    def __init__(self, energy_level: int, gamma_array) -> None:
        self.n = energy_level
        self.gamma_array = gamma_array
        if energy_level < 0:
            raise ValueError("Negative energy level asked!")
        elif energy_level == 0:
            self.energy_func = level_zero
        elif energy_level == 1:
            self.energy_func = level_one
        elif (-1)**energy_level > 0:
            self.energy_func = even_positive
        elif (-1)**energy_level < 0:
            self.energy_func = odd_positive
        else:
            raise ValueError("Something strange happened with the energy level.")
    
    def find_energy(self, gamma):
        starting_guess = 0.1
        max_tries = 50
        num_tries = 0
        while num_tries < max_tries:
            num_tries += 1


    
    def calculate_array(self):
        self.array = 0 * self.gamma_array
        for gamma in self.gamma_array:
            self.find_energy(gamma)




xaxis_array = -np.linspace(-nearly_pi/2, nearly_pi/2, 100) # For the x-koordinates of the plot
gamma_array = np.tan(xaxis_array)*2./box_L # To recieve the gamma factors times 2 over L for the calculation of the energy eigenvalues

