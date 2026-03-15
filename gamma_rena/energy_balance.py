"""
Výpočet energetickej bilancie v systéme.
"""

import numpy as np
from scipy import integrate, optimize
from .bremsstrahlung import Bremsstrahlung
from .photo_absorption import PhotoAbsorption
from .config import SimulationConfig

class EnergyBalance:
    """Modeluje energetickú bilanciu"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.brem = Bremsstrahlung(config)
        self.photo = PhotoAbsorption()
    
    def electron_energy_loss(self, E_electron, material="air"):
        """
        Celková strata energie elektrónu.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        material : str
            Typ materiálu
        
        Returns:
        --------
        dict
            Zložky straty energie
        """
        
        # Bremsstrahlung
        loss_brem = self.brem.energy_loss_rate(E_electron)
        
        # Ionizácia a excitácia (Bethe-Bloch)
        loss_ionization = self._bethe_bloch_loss(E_electron)
        
        return {
            'bremsstrahlung': loss_brem,
            'ionization': loss_ionization,
            'total': loss_brem + loss_ionization
        }
    
    def _bethe_bloch_loss(self, E_electron):
        """
        Strata energie na ionizáciu a excitáciu (Bethe-Bloch formula).
        """
        
        # Stredná energia ionizácie
        I = 85.7  # eV pre vzduch
        I_mev = I / 1e6
        
        # Kinetická energia
        K = E_electron - 0.511
        
        if K <= 0:
            return 0.0
        
        # Lorentzov faktor
        beta_gamma = np.sqrt(K * (K + 2*0.511)) / 0.511
        
        # Bethe-Bloch
        loss = 1.38e-3 * (np.log(2*0.511*beta_gamma**2*K/I_mev**2) - 2*beta_gamma**2/(1-beta_gamma**2))
        
        return max(0, loss)
    
    def cascade_evolution(self, E_initial, num_steps=100):
        """
        Evolúcia energetickej kaskády elektrónu.
        
        Parameters:
        -----------
        E_initial : float
            Počiatočná energia [MeV]
        num_steps : int
            Počet krokov simulácie
        
        Returns:
        --------
        dict
            Výsledky kaskády
        """
        
        energies = np.zeros(num_steps)
        photons = []
        energies[0] = E_initial
        
        for i in range(1, num_steps):
            E = energies[i-1]
            
            if E < 1.0:  # Minimálna energia
                energies[i:] = E
                break
            
            # Strata energie
            loss = self.electron_energy_loss(E)['total']
            delta_E = loss * 0.01  # Malý krok
            
            energies[i] = max(0.511, E - delta_E)
        
        return {
            'energies': energies,
            'photons_emitted': len(photons),
            'final_energy': energies[-1],
            'energy_transferred': E_initial - energies[-1]
        }
    
    def energy_transfer_efficiency(self, E_electron):
        """
        Účinnosť prenosu energie na žiarenie.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        
        Returns:
        --------
        float
            Účinnosť (0-1)
        """
        
        loss_data = self.electron_energy_loss(E_electron)
        
        if loss_data['total'] <= 0:
            return 0.0
        
        efficiency = loss_data['bremsstrahlung'] / loss_data['total']
        
        return max(0, min(1, efficiency))
    
    def critical_energy(self):
        """
        Kritická energia - kde je ionizácia a Bremsstrahlung rovnomerne.
        
        Returns:
        --------
        float
            Kritická energia [MeV]
        """
        
        def equation(E):
            loss = self.electron_energy_loss(E)
            return loss['bremsstrahlung'] - loss['ionization']
        
        try:
            result = optimize.brentq(equation, 1.0, 10.0)
            return result
        except:
            return 5.0  # Default
    
    def thermalization_time(self, E_electron):
        """
        Čas potrebný na termalizáciu elektrónu.
        
        Parameters:
        -----------
        E_electron : float
            Počiatočná energia [MeV]
        
        Returns:
        --------
        float
            Čas termalizácie [s]
        """
        
        # Odhad na základe strednej voľnej dráhy
        mfp = self.brem.mean_free_path(E_electron)
        
        # Rýchlosť elektrónu (relativistická)
        pc = np.sqrt(E_electron**2 - 0.511**2)
        gamma = E_electron / 0.511
        beta = pc / E_electron
        velocity = beta * 3e8  # m/s
        
        thermalization_distance = 1000 * mfp / 100  # m
        thermalization_time = thermalization_distance / max(velocity, 1e6)
        
        return max(0, thermalization_time)