"""
Simulácia fotoabsorpcie - proces absorpcie fotonov v materiáloch.
"""

import numpy as np
from scipy import integrate
from .physics_constants import PhysicsConstants as PC

class PhotoAbsorption:
    """Modeluje fotoabsorpciu"""
    
    def __init__(self):
        self.pc = PC()
    
    def compton_cross_section(self, E_photon, Z=1):
        """
        Comptonov rozptyl - Klein-Nishina formula.
        
        Parameters:
        -----------
        E_photon : float
            Energia fotónu [MeV]
        Z : float
            Atómový počet
        
        Returns:
        --------
        float
            Účinný prierez [barn]
        """
        
        # Redukovaná energia
        epsilon = E_photon / 0.511
        
        if epsilon <= 0:
            return 0.0
        
        # Klein-Nishina formula
        k = 1.0 / (1 + 2*epsilon)
        
        sigma_kn = 2 * np.pi * (self.pc.R_E * 1e15)**2 * (
            k * (2 + 2*epsilon - 2*k*epsilon) / epsilon**2 +
            np.log(2*epsilon*(1+epsilon)/(1+2*epsilon)) / epsilon -
            (1 + 2*epsilon) / ((1 + 2*epsilon)**2)
        )
        
        return max(0, sigma_kn * Z)
    
    def photoelectric_cross_section(self, E_photon, Z=7.5):
        """
        Fotoelektrický efekt.
        
        Parameters:
        -----------
        E_photon : float
            Energia fotónu [MeV]
        Z : float
            Atómový počet
        
        Returns:
        --------
        float
            Účinný prierez [barn]
        """
        
        if E_photon < 0.01:
            return 0.0
        
        # Aproximácia pre fotóny v MeV rozsahu
        sigma_pe = 10.0 * self.pc.ALPHA**3 * Z**4 / (E_photon)**3
        
        if E_photon > 1.0:
            # Vysokoenergetický režim
            sigma_pe *= np.exp(-E_photon / 2.0)
        
        return max(0, sigma_pe)
    
    def pair_production_cross_section(self, E_photon, Z=7.5):
        """
        Tvorba párov elektrónu-pozitrónu.
        
        Parameters:
        -----------
        E_photon : float
            Energia fotónu [MeV]
        Z : float
            Atómový počet
        
        Returns:
        --------
        float
            Účinný prierez [barn]
        """
        
        # Prah pre tvorbu párov
        threshold = 1.022  # MeV
        
        if E_photon < threshold:
            return 0.0
        
        # Bethe-Heitler formula pre tvorbu párov
        E_ratio = E_photon / threshold
        
        sigma_pp = (7.0/9.0) * self.pc.ALPHA * (self.pc.R_E * 1e15)**2 * Z**2 * (
            np.log(2*E_ratio) - 109.0/42.0
        )
        
        return max(0, sigma_pp)
    
    def total_attenuation_coefficient(self, E_photon, density=1.225, Z=7.5, A=14.4):
        """
        Celkový koeficient útlmu fotónu.
        
        Parameters:
        -----------
        E_photon : float
            Energia fotónu [MeV]
        density : float
            Hustota [kg/m³]
        Z : float
            Efektívny atómový počet
        A : float
            Stredná atómová hmotnosť
        
        Returns:
        --------
        float
            Útlmový koeficient [cm⁻¹]
        """
        
        # Celkový účinný prierez
        sigma_total = (
            self.compton_cross_section(E_photon, Z) +
            self.photoelectric_cross_section(E_photon, Z) +
            self.pair_production_cross_section(E_photon, Z)
        )
        
        # Počet atómov na jednotku objemu
        mass_per_atom = A * 1.66e-27  # kg
        n = density / mass_per_atom * 1e-2  # cm⁻³
        
        mu = n * sigma_total * self.pc.BARN  # cm⁻¹
        
        return max(0, mu)
    
    def attenuation_depth(self, E_photon, density=1.225):
        """
        Hĺbka útlmu (1/e hĺbka).
        
        Parameters:
        -----------
        E_photon : float
            Energia fotónu [MeV]
        density : float
            Hustota [kg/m³]
        
        Returns:
        --------
        float
            Hĺbka útlmu [cm]
        """
        
        mu = self.total_attenuation_coefficient(E_photon, density)
        
        if mu <= 0:
            return 1e10
        
        return 1.0 / mu
    
    def transmitted_intensity(self, E_photon, thickness, density=1.225):
        """
        Intenzita prejdeného žiarenia.
        
        Parameters:
        -----------
        E_photon : float
            Energia fotónu [MeV]
        thickness : float
            Hrúbka materiálu [cm]
        density : float
            Hustota [kg/m³]
        
        Returns:
        --------
        float
            Relatívna intenzita (0-1)
        """
        
        mu = self.total_attenuation_coefficient(E_photon, density)
        intensity = np.exp(-mu * thickness)
        
        return max(0, min(1, intensity))