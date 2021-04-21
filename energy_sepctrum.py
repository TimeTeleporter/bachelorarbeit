from abc import ABC
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

"""
With this program we want to plot the energy spectrum of a partilce in a box
with parity symmetric boundaries gamma_+ = gamma_- = gamma.
"""

BOX = 1
EPSILON = 0.00001
NEARLY_PI = 3.14

class energy_condition(ABC):
    def __init__(self, func, xmin=EPSILON, xmax=1000):
        self.xmin = xmin
        self.xmax = xmax
        self.condition_equation = func
    
    def calculate_eigenvalue(self, gamma_array):
        results = []
        for gamma in gamma_array:
            def func_with_gamma(x):
                return self.condition_equation(x, gamma)
            results.append(float(brentq(func_with_gamma, self.xmin, self.xmax)))
        return results

    def return_energy_spectrum(self, gamma_array):
        return np.power(np.array(self.calculate_eigenvalue(gamma_array))*2/BOX/np.pi, 2)

class even_positive_spectrum(energy_condition):
    def even_positive(self, x, gamma):
        return gamma/x - np.tan(x)

    def __init__(self, energy_level=2):
        if energy_level == 0:
            xmin = EPSILON
            xmax = np.pi / 2 * (1 - EPSILON)
        else:
            xmin = (2*abs(energy_level) - 1)/2 * np.pi * (1 + EPSILON)
            xmax = (2*abs(energy_level) + 1)/2 * np.pi * (1 - EPSILON)
        super().__init__(self.even_positive, xmin=xmin, xmax=xmax)

class odd_positive_spectrum(energy_condition):
    def odd_positive(self, x, gamma):
        return gamma/x + 1/np.tan(x)
    
    def __init__(self, energy_level: int =2):
        if energy_level == 0:
            xmin = EPSILON
        else:
            xmin = np.pi * abs(energy_level) * (1 + EPSILON)
        xmax = np.pi * (abs(energy_level) + 1) * (1 - EPSILON)
        super().__init__(self.odd_positive, xmin=xmin, xmax=xmax)

class even_negative_spectrum(energy_condition):
    def even_negative(self, x, gamma):
        return gamma/x + np.tanh(x)
    
    def __init__(self):
        super().__init__(self.even_negative)
    
    def return_energy_spectrum(self, gamma_array):
        positive_gamma = []
        negative_gamma = []
        include_negone = False
        for gamma in gamma_array:
            if gamma > 0:
                positive_gamma.append(gamma)
            elif gamma == 0:
                include_negone = True
            else:
                negative_gamma.append(gamma)
        positive_results = even_positive_spectrum(energy_level=0).return_energy_spectrum(positive_gamma)
        negative_results = -super().return_energy_spectrum(negative_gamma)
        if include_negone:
            return np.array(list(positive_results) + [0] + list(negative_results))
        else:
            return np.array(list(positive_results) + list(negative_results))

class odd_negative_spectrum(energy_condition):
    def odd_negative(self, x, gamma):
        return gamma/x + 1/np.tanh(x)
    
    def __init__(self):
        super().__init__(self.odd_negative)
    
    def return_energy_spectrum(self, gamma_array):
        positive_gamma = []
        negative_gamma = []
        include_negone = False
        for gamma in gamma_array:
            if gamma > -1:
                positive_gamma.append(gamma)
            elif gamma == -1:
                include_negone = True
            else:
                negative_gamma.append(gamma)
        positive_results = odd_positive_spectrum(energy_level=0).return_energy_spectrum(positive_gamma)
        negative_results = -super().return_energy_spectrum(negative_gamma)
        if include_negone:
            return np.array(list(positive_results) + [-1] + list(negative_results))
        else:
            return np.array(list(positive_results) + list(negative_results))

def gamma_from_xaxis(xaxis):
    return np.tan(xaxis)*BOX/2

xaxis_array = -np.linspace(-NEARLY_PI/2, NEARLY_PI/2, 100)
gamma_array = gamma_from_xaxis(xaxis_array)

n0 = even_negative_spectrum().return_energy_spectrum(gamma_array)
n1 = odd_negative_spectrum().return_energy_spectrum(gamma_array)
n2 = even_positive_spectrum(energy_level=1).return_energy_spectrum(gamma_array)
n3 = odd_positive_spectrum(energy_level=1).return_energy_spectrum(gamma_array)
n4 = even_positive_spectrum(energy_level=2).return_energy_spectrum(gamma_array)

plt.plot(xaxis_array, n0, label="n=0")
plt.plot(xaxis_array, n1, label="n=1")
plt.plot(xaxis_array, n2, label="n=2")
plt.plot(xaxis_array, n3, label="n=3")
plt.plot(xaxis_array, n4, label="n=4")

plt.title("Particle in a box energy spectrum with parity symmetric boundary conditions $\gamma_+ = \gamma_-$")
plt.xlabel("arctan($\gamma L/2$)")
plt.ylabel("Energy")

plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
plt.xlim(xaxis_array[-1], xaxis_array[0])
plt.xticks(np.linspace(xaxis_array[-1], xaxis_array[0], 5))
plt.ylim(bottom=-4, top=16)
plt.yticks(np.arange(-4, 16))
plt.grid(True)

plt.show()