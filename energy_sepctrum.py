import typing
from abc import ABC, abstractmethod
import numpy as np
from numpy.core.numeric import NaN
from scipy.optimize import brentq
import matplotlib.pyplot as plt

"""
With this program we want to plot the energy spectrum of a partilce in a box
with parity symmetric boundaries gamma_+ = gamma_- = gamma.
"""

BOX = 1
EPSILON = 0.00001
NEARLY_PI = 3.14

def gamma_from_xaxis(xaxis):
    return np.tan(xaxis)*BOX/2

class energy_condition(ABC):
    def __init__(self, func, xmin=0, xmax=2):
        self.xmin = xmin
        self.xmax = xmax
        self.condition_equation = func

    def return_energy_spectrum(self, gamma_array):
        results = []
        for gamma in gamma_array:
            def func_with_gamma(x):
                return self.condition_equation(x, gamma)
            results.append(float(brentq(func_with_gamma, self.xmin, self.xmax)))
        return np.power(np.array(results)*2/BOX/np.pi, 2)

class even_positive_spectrum(energy_condition):
    def even_positive(self, x, gamma):
        return gamma/x - np.tan(x)

    def __init__(self, energy_level: int =2):
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
    
    def __init__(self, xmax=1000):
        super().__init__(self.even_negative, xmin=EPSILON, xmax=xmax)

class odd_negative_spectrum(energy_condition):
    def odd_negative(self, x, gamma):
        return gamma/x + 1/np.tanh(x)
    
    def __init__(self, xmax=1000):
        super().__init__(self.odd_negative, xmin=EPSILON, xmax=xmax)

positive_to_near_zero = list(-np.linspace(-NEARLY_PI/2, -EPSILON, 100))
neg_near_zero_to_negative = list(-np.linspace(EPSILON, NEARLY_PI/2, 100))
xaxis_array = np.array(positive_to_near_zero + [0] + neg_near_zero_to_negative)

gamma_array = gamma_from_xaxis(xaxis_array)

spec0pos = even_positive_spectrum(energy_level=0)
spec0neg = even_negative_spectrum()
n0 = np.array(list(spec0pos.return_energy_spectrum(gamma_from_xaxis(positive_to_near_zero)))
            + [0] 
            + list(-spec0neg.return_energy_spectrum(gamma_from_xaxis(neg_near_zero_to_negative))))

n1_gamma_pos = []
n1_gamma_neg = []
for gamma in gamma_array:
    if gamma > -1.:
        n1_gamma_pos.append(gamma)
    if gamma < -1:
        n1_gamma_neg.append(gamma)



spec1pos = odd_positive_spectrum(energy_level=0)
spec1neg = odd_negative_spectrum()
n1 = np.array(list(spec1pos.return_energy_spectrum(n1_gamma_pos))
            + [0 for gamma in gamma_array if gamma == -1]
            + list(-spec1neg.return_energy_spectrum(n1_gamma_neg)))

n1_xaxis = [xaxis_array[index] for index in range(len(n1))]

spec2 = even_positive_spectrum(energy_level=1)
n2 = spec2.return_energy_spectrum(gamma_array)
spec3 = odd_positive_spectrum(energy_level=1)
n3 = spec3.return_energy_spectrum(gamma_array)
spec4 = even_positive_spectrum(energy_level=2)
n4 = spec4.return_energy_spectrum(gamma_array)

plt.plot(xaxis_array, n0, label="n=0")
plt.plot(n1_xaxis, n1, label="n=1")
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