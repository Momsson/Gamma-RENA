"""
Modelovanie Bremsstrahlung emisie - radiácia produkovaná pri brzdení nabitých частиц.
"""

import numpy as np
from scipy import integrate, interpolate
from .physics_constants import PhysicsConstants as PC
from .config import SimulationConfig

class Bremsstrahlung:
    """Modeluje Bremsstrahlung emisiu"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.pc = PC()
        
    def differential_cross_section(self, E_electron, E_photon, Z=7.5):
        """
        Diferenciálny účinný prierez Bremsstrahlung.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        E_photon : float
            Energia fotónu [MeV]
        Z : float
            Efektívny atómový počet (vzduchu ≈ 7.5)
        
        Returns:
        --------
        float
            Diferenciálny prierez [barn/MeV]
        """
        
        if E_photon >= E_electron or E_photon <= 0:
            return 0.0
        
        # Energia elektrónu po emitovaní fotónu
        E_prime = E_electron - E_photon
        
        # Momenty
        p_e = np.sqrt(E_electron**2 - 0.511**2)  # MeV/c
        p_prime = np.sqrt(E_prime**2 - 0.511**2) if E_prime > 0.511 else 1e-6
        
        # Lorentzov faktor
        gamma = E_electron / 0.511
        beta = p_e / E_electron
        
        # Coulomb correction
        d_e = 1.13 + 3.76 * (PC.ALPHA * Z)**2
        
        # Kvantový prierez - Bethe-Heitler formula
        if E_electron < 2.0:  # Low energy approximation
            sigma = (28.0/3.0) * PC.ALPHA * (PC.R_E * 1e15)**2 * Z**2 / E_photon
            sigma *= np.log(2 * E_electron / 0.511)
        else:
            # High energy: Bethe-Heitler
            u = E_photon / E_electron
            phi_1 = 8.0/3.0 * np.log((1 + u) / u) + 2*u - (2*u**2)/(1+u)**2
            phi_2 = (10.0/9.0 + 8*u**2/(1+u)**2) * np.log((1+u)/u)
            phi_2 -= 2.0/3.0 - 8*u/(3*(1+u))
            
            sigma = (PC.ALPHA * (PC.R_E * 1e15)**2 / (3 * np.pi * u)) * (
                Z**2 * (phi_1 + Z/137 * phi_2)
            )
        
        return max(0, sigma)
    
    def total_cross_section(self, E_electron, Z=7.5):
        """
        Celkový účinný prierez pre Bremsstrahlung.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        Z : float
            Atómový počet
        
        Returns:
        --------
        float
            Celkový prierez [barn]
        """
        
        # Integrácia od minima (0.01 MeV) po maximum (E_electron - 0.511)
        E_min = 0.01
        E_max = max(E_electron - 0.511, E_min)
        
        if E_max <= E_min:
            return 1e-6
        
        # Numerická integrácia
        def integrand(E_gamma):
            return self.differential_cross_section(E_electron, E_gamma, Z)
        
        result, _ = integrate.quad(integrand, E_min, E_max)
        return max(0, result)
    
    def mean_free_path(self, E_electron, density=1.225, Z=7.5, A=14.4):
        """
        Stredná voľná dráha elektrónu.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        density : float
            Hustota materiálu [kg/m³]
        Z : float
            Efektívny atómový počet
        A : float
            Stredná atómová hmotnosť
        
        Returns:
        --------
        float
            Stredná voľná dráha [cm]
        """
        
        sigma = self.total_cross_section(E_electron, Z)
        
        # Počet atómov na jednotku objemu
        mass_per_atom = A * 1.66e-27  # kg
        n = density / mass_per_atom * 1e-6  # cm⁻³
        
        if sigma <= 0 or n <= 0:
            return 1e10
        
        lambda_mfp = 1.0 / (n * sigma * self.pc.BARN)
        return lambda_mfp
    
    def photon_spectrum(self, E_electron, num_bins=50):
        """
        Generuje spektrum emitovaných fotónov.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        num_bins : int
            Počet energetických tried
        
        Returns:
        --------
        tuple
            (energy_bins, spectrum)
        """
        
        E_min = 0.01
        E_max = E_electron - 0.511
        
        if E_max <= E_min:
            return np.array([E_min]), np.array([0.0])
        
        E_bins = np.logspace(np.log10(E_min), np.log10(E_max), num_bins)
        spectrum = np.array([
            self.differential_cross_section(E_electron, E, 7.5) * (E - E_min)
            for E in E_bins
        ])
        
        # Normalizácia
        if np.max(spectrum) > 0:
            spectrum /= np.max(spectrum)
        
        return E_bins, spectrum
    
    def energy_loss_rate(self, E_electron, density=1.225):
        """
        Rýchlosť straty energie Bremsstrahlung.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        density : float
            Hustota [kg/m³]
        
        Returns:
        --------
        float
            Strata energie [MeV/cm]
        """
        
        # Stredná energia fotónu
        sigma_total = self.total_cross_section(E_electron)
        
        E_min = 0.01
        E_max = E_electron - 0.511
        
        def weighted_integrand(E_gamma):
            return E_gamma * self.differential_cross_section(E_electron, E_gamma, 7.5)
        
        if E_max > E_min:
            mean_photon_energy, _ = integrate.quad(weighted_integrand, E_min, E_max)
            mean_photon_energy /= (sigma_total + 1e-10)
        else:
            mean_photon_energy = 0.1
        
        # Počet interakcií na jednotku vzdialenosti
        lambda_mfp = self.mean_free_path(E_electron, density)
        
        if lambda_mfp > 0:
            loss_rate = mean_photon_energy / lambda_mfp
        else:
            loss_rate = 0.0
        
        return max(0, loss_rate)