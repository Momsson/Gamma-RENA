import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constants
# -----------------------------
c = 3e8
m_e = 9.11e-31
e = 1.602e-19

# characteristic energy (7.3 MeV)
E0 = 7.3e6  # eV

# material parameters (example PMMA)
Et = 0.213e9  # V/m runaway threshold field
Nm = 7.229e21 # molecular density (cm^-3 -> approx scaling)

# applied electric field
Ea = 0.35e9

# -----------------------------
# electron energy distribution
# -----------------------------
def electron_distribution(E):
    return (1/E0) * np.exp(-E/E0)

# -----------------------------
# avalanche multiplication length
# -----------------------------
def avalanche_length(Ea, Et):
    if Ea <= Et:
        return np.inf
    return (E0/Ea) / (1 - Et/Ea)

# -----------------------------
# electron velocity
# relativistic approximation
# -----------------------------
def electron_velocity(E):
    gamma = 1 + E/(511e3)
    beta = np.sqrt(1 - 1/gamma**2)
    return beta * c

# -----------------------------
# simple bremsstrahlung model
# -----------------------------
def photon_production(E):
    v = electron_velocity(E)
    sigma = 1e-30 * E  # simplified cross section
    return electron_distribution(E) * sigma * v

# -----------------------------
# simulation
# -----------------------------
energies = np.linspace(1e3, 1e7, 1000)

photons = photon_production(energies)

lr = avalanche_length(Ea, Et)

print("Avalanche length:", lr)

plt.plot(energies/1e6, photons)
plt.xlabel("Electron energy (MeV)")
plt.ylabel("Photon production rate")
plt.title("Bremsstrahlung photon production")
plt.show()
