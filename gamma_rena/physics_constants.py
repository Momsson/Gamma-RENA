"""
Fyzikálne konštanty a ich hodnoty.
"""

import numpy as np

class PhysicsConstants:
    """Fyzikálne konštanty v SI jednotkách"""
    
    # Základné konštanty
    C = 3e8  # Rýchlosť svetla [m/s]
    E = 1.602e-19  # Elementárny náboj [C]
    H = 6.626e-34  # Planckova konštanta [J*s]
    HBAR = H / (2 * np.pi)  # Redukovaná Planckova konštanta
    K_B = 1.381e-23  # Boltzmannova konštanta [J/K]
    
    # Hmotnosti
    M_E = 9.109e-31  # Hmotnosť elektrónu [kg]
    M_P = 1.673e-27  # Hmotnosť protónu [kg]
    
    # Jednotky
    MEV_TO_J = 1.602e-13  # Konverzia MeV na Joule
    BARN = 1e-24  # Barn v cm²
    
    # Klasický polomer elektrónu
    R_E = (E**2) / (4 * np.pi * 8.854e-12 * M_E * C**2 * E)
    
    # Comptonova vlnová dĺžka
    LAMBDA_C = H / (M_E * C)
    
    # Fine structure constant
    ALPHA = 1 / 137.036
    
    # Thomson cross-section
    SIGMA_T = (8/3) * np.pi * R_E**2
    
    @staticmethod
    def rest_energy(mass_kg):
        """Pokojová energia v J"""
        return mass_kg * PhysicsConstants.C**2
    
    @staticmethod
    def rest_energy_mev(mass_amu):
        """Pokojová energia v MeV"""
        return mass_amu * 931.494
    
    @staticmethod
    def lorentz_factor(velocity_m_s):
        """Lorentzov faktor γ"""
        beta = velocity_m_s / PhysicsConstants.C
        return 1 / np.sqrt(1 - beta**2)
    
    @staticmethod
    def kinetic_energy(gamma):
        """Kinetická energia elektrónu [MeV]"""
        rest_energy = 0.511  # MeV
        return (gamma - 1) * rest_energy