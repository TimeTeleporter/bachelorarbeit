"""
Program by Immanuel Albrecht
The momentum_distribution module: 
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from energy_sepctrum import calculate_parity_symmetric_energy


BOX = 1


def pos_energy_momentumentum_distribution(momentum_k_array, energy_level=2, gamma=2):
    if gamma == 0:
        energy_k = energy_level * np.pi / BOX
    elif gamma == np.inf:
        energy_k = (energy_level+1) * np.pi / BOX
    elif gamma == -np.inf:
        energy_k = (energy_level-1) * np.pi / BOX
    else:
        energy_k = calculate_parity_symmetric_energy(energy_level, gamma) * np.pi / BOX
    
    def sine_stuff(energy_k, momentum_k, sign=1):
        k_stuff = momentum_k + sign * energy_k
        return np.sin((k_stuff) * BOX / 2)/(k_stuff)

    zaehler = np.power(sine_stuff(energy_k, momentum_k_array, 1) + np.power(-1, energy_level) * sine_stuff(energy_k, momentum_k_array, -1), 2)
    nenner = (BOX + np.power(-1, energy_level) * np.sin(energy_k * BOX) / energy_k)
    return zaehler / nenner


min_momentum_bin = -20
max_momentum_bin = 20

momentum_n_array = np.array(np.arange(min_momentum_bin, max_momentum_bin+1))
momentum_k_array = np.array(momentum_n_array) * np.pi/BOX
momentum_probability = pos_energy_momentumentum_distribution(momentum_k_array)

print(momentum_n_array)
print(momentum_probability)

plt.plot(momentum_n_array, momentum_probability)

plt.ylabel = "$\bra \phi_l | \phi_k \ket^2$"
plt.xlabel = "n"

plt.show