"""
Program by Immanuel Albrecht
The energy_sepctrum module: Houses the classes for particle in a box energy spectrum calculation.

As of 20210422 the module implements the spectra for the parity symmetric case with the classes
    * even_positive_spectrum: implements gamma/x - tan(x).
    * odd_positive_spectrum: implements gamma/x + 1/tan(x).
    * even_split_spectrum: implements gamma/x + tanh(x) for gamma < 0.
    * odd_split_spectrum: implements gamma/x + 1/tanh(x) for gamma < 1.

By executing this module a plot gets created that shows the energy_spectrum of the first four energy levels.

It also houses the functions
    * calculate_parity_symmetric_energies
to import in other modules.

To be improved:
Introduce a @abstractmethod in the abstract base class and move the individual energy conditions
into a function that overwrites that abstract methood.
"""

from abc import ABC
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt



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

class parity_antisymmetric_positive(energy_condition):
    def __init__(self, energy_level: int):
        self.energy_level = energy_level
    
    def calculate_eigenvalue(self, gamma_array):
        return [self.energy_level * np.pi / 2 for gamma in gamma_array]

class parity_antisymmetric_negative(energy_condition):
    def __init__(self):
        self.energy_level = 0
    
    def calculate_eigenvalue(self, gamma_array):
        return [abs(gamma)  for gamma in gamma_array]
    
    def return_energy_spectrum(self, gamma_array):
        return -super().return_energy_spectrum(gamma_array)


class split_spectrum(energy_condition):
    def __init__(self, func):
        super().__init__(func)
    
    def return_energy_spectrum(self, gamma_array, splitter=0, energy_condition_obj=even_positive_spectrum(energy_level=0)):
        positive_gamma = []
        negative_gamma = []
        include_splitter = False
        for gamma in gamma_array:
            if gamma > splitter:
                positive_gamma.append(gamma)
            elif gamma == splitter:
                include_splitter = True
            else:
                negative_gamma.append(gamma)
        positive_results = energy_condition_obj.return_energy_spectrum(positive_gamma)
        negative_results = -super().return_energy_spectrum(negative_gamma)
        if include_splitter:
            return np.array(list(positive_results) + [splitter] + list(negative_results))
        else:
            return np.array(list(positive_results) + list(negative_results))


class even_split_spectrum(split_spectrum):
    def even_negative(self, x, gamma):
        return gamma/x + np.tanh(x)
    
    def __init__(self):
        super().__init__(self.even_negative)
    
    def return_energy_spectrum(self, gamma_array):
        return super().return_energy_spectrum(gamma_array, splitter=0, energy_condition_obj=even_positive_spectrum(energy_level=0))


class odd_split_spectrum(split_spectrum):
    def even_negative(self, x, gamma):
        return gamma/x + 1/np.tanh(x)
    
    def __init__(self):
        super().__init__(self.even_negative)
    
    def return_energy_spectrum(self, gamma_array):
        return super().return_energy_spectrum(gamma_array, splitter=-1, energy_condition_obj=odd_positive_spectrum(energy_level=0))



def calculate_parity_symmetric_energy(energy_level: int, gamma: float) -> float:
    '''Returns the energy value calculated for a given gamma on a given energy level.'''
    energy_level = int(abs(energy_level))
    isodd = bool(energy_level % 2)
    n = int(energy_level/2)
    if energy_level == 0:
        return float(even_split_spectrum().return_energy_spectrum([gamma]))
    elif energy_level == 1:
        return float(odd_split_spectrum().return_energy_spectrum([gamma]))
    elif isodd:
        return float(odd_positive_spectrum(n).return_energy_spectrum([gamma]))
    else:
        return float(even_positive_spectrum(n).return_energy_spectrum([gamma]))