from energy_sepctrum import *

def gamma_from_xaxis(xaxis):
    return np.tan(xaxis)*BOX/2

xaxis_array = -np.linspace(-NEARLY_PI/2, NEARLY_PI/2, 100)
gamma_array = gamma_from_xaxis(xaxis_array)

n0 = parity_antisymmetric_negative().return_energy_spectrum(gamma_array)
n1 = parity_antisymmetric_positive(1).return_energy_spectrum(gamma_array)
n2 = parity_antisymmetric_positive(2).return_energy_spectrum(gamma_array)
n3 = parity_antisymmetric_positive(3).return_energy_spectrum(gamma_array)
n4 = parity_antisymmetric_positive(4).return_energy_spectrum(gamma_array)

plt.plot(xaxis_array, n0, label="n=0")
plt.plot(xaxis_array, n1, label="n=1")
plt.plot(xaxis_array, n2, label="n=2")
plt.plot(xaxis_array, n3, label="n=3")
plt.plot(xaxis_array, n4, label="n=4")

plt.title("Particle in a box energy spectrum with parity antisymmetric boundary conditions $\gamma_+ = -\gamma_-$")
plt.xlabel("arctan($\gamma L/2$)")
plt.ylabel("Energy")

plt.legend(bbox_to_anchor=(1.01,1), loc="upper left")
plt.xlim(xaxis_array[-1], xaxis_array[0])
plt.xticks(np.linspace(xaxis_array[-1], xaxis_array[0], 5))
plt.ylim(bottom=-4, top=17)
plt.yticks(np.arange(-4, 17))
plt.grid(True)

plt.show()