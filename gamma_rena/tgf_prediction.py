"""
Predikcia Terrestrial Gamma-ray Flashes (TGF).
"""

import numpy as np
from scipy import stats
from .energy_balance import EnergyBalance
from .config import SimulationConfig

class TGFPredictor:
    """Predpovedá Terrestrial Gamma-ray Flashes"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.energy_balance = EnergyBalance(config)
    
    def runaway_electron_avalanche(self, E_seed, num_generations=10, branching_ratio=1.5):
        """
        Modeluje multiplikáciu elektrónu v elektrickej pole (runaway avalanche).
        
        Parameters:
        -----------
        E_seed : float
            Energia seed elektrónu [MeV]
        num_generations : int
            Počet generácií multiplikácie
        branching_ratio : float
            Počet dcérskych elektrónov na jeden parent elektrónu
        
        Returns:
        --------
        dict
            Informácie o lavíne
        """
        
        electron_count = np.zeros(num_generations)
        electron_count[0] = 1
        
        total_energy = E_seed
        
        for gen in range(1, num_generations):
            # Exponenciálny rast s rozptylom
            decay_factor = np.exp(-gen * 0.3)  # Zoslabovanie
            electron_count[gen] = electron_count[gen-1] * branching_ratio * decay_factor
            
            # Energia elektrónu rastie
            E_avg = E_seed * (1 + gen * 0.1)
            total_energy += electron_count[gen] * E_avg
        
        return {
            'generation_count': electron_count,
            'total_electrons': int(np.sum(electron_count)),
            'total_energy': total_energy,
            'growth_rate': branching_ratio
        }
    
    def threshold_field(self, temperature=293.15, pressure=101.325):
        """
        Elektrické pole potrebné na spustenie runaway avalanchy.
        
        Parameters:
        -----------
        temperature : float
            Teplota [K]
        pressure : float
            Tlak [kPa]
        
        Returns:
        --------
        float
            Prahové pole [V/m]
        """
        
        # Paskov model pre vzduch
        E_k = 327.8  # V/m*Pa (kritické pole)
        
        # Normalizácia na tlak
        p_norm = pressure / 101.325
        
        # Korrekcía teploty
        T_factor = (temperature / 293.15)
        
        E_threshold = E_k * p_norm * T_factor
        
        return E_threshold
    
    def tgf_probability(self, electric_field, altitude=15000):
        """
        Pravdepodobnosť vzniku TGF na danom mieste.
        
        Parameters:
        -----------
        electric_field : float
            Elektrické pole [V/m]
        altitude : float
            Nadmorská výška [m]
        
        Returns:
        --------
        float
            Pravdepodobnosť (0-1)
        """
        
        # Prahové pole
        E_th = self.threshold_field()
        
        # Normalizované pole
        field_ratio = electric_field / E_th
        
        if field_ratio < 1.0:
            prob = 0.0
        else:
            # Sigmoidálna funkcia
            prob = 1.0 / (1.0 + np.exp(-5.0 * (field_ratio - 1.2)))
        
        # Korrekcía výšky (TGF sú časnejšie vo vyšších nadmorských výškach)
        altitude_factor = 1.0 - 0.001 * (20000 - altitude) / 20000
        
        prob *= max(0.1, altitude_factor)
        
        return max(0, min(1, prob))
    
    def photon_yield(self, E_electron):
        """
        Počet gama fotónov na jeden elektrónu.
        
        Parameters:
        -----------
        E_electron : float
            Energia elektrónu [MeV]
        
        Returns:
        --------
        float
            Priemerne emitované fotóny
        """
        
        # Počet Bremsstrahlung interakcií
        mfp = self.energy_balance.brem.mean_free_path(E_electron)
        
        # Stredná voľná dráha em. kaskády
        cascade_length = 1000 * mfp / 100  # m
        
        # Fotóny na jednotku vzdialenosti
        photon_rate = 1.0 / mfp if mfp > 0 else 0
        
        # Celkový počet fotónov
        n_photons = photon_rate * cascade_length
        
        return max(0, n_photons)
    
    def spectrum_hardness(self, electron_energy):
        """
        Tvrdosť spektra (index tvrdosti fotónov).
        
        Parameters:
        -----------
        electron_energy : float
            Energia elektrónu [MeV]
        
        Returns:
        --------
        float
            Index tvrdosti (vyšší = tvrdší spektrum)
        """
        
        # Empirická korelacia
        hardness = np.log10(max(1.0, electron_energy)) * 0.5 + 0.5
        
        return hardness
    
    def simulate_tgf_event(self, duration=1e-3, E_field=500000):
        """
        Kompletná simulácia TGF udalosti.
        
        Parameters:
        -----------
        duration : float
            Trvanie udalosti [s]
        E_field : float
            Elektrické pole [V/m]
        
        Returns:
        --------
        dict
            Výsledky simulácie
        """
        
        # Kontrola či je pole nadprahové
        E_th = self.threshold_field()
        
        if E_field < E_th:
            return {
                'occurred': False,
                'reason': 'Below threshold field',
                'photon_yield': 0,
                'energy_radiated': 0
            }
        
        # Vytvorenie avalanchy
        seed_energy = 1.0  # MeV
        avalanche_data = self.runaway_electron_avalanche(
            seed_energy, 
            num_generations=8
        )
        
        # Spektrum
        photon_yield = sum([
            self.photon_yield(seed_energy * (1 + i*0.1))
            for i in range(8)
        ])
        
        return {
            'occurred': True,
            'electron_count': avalanche_data['total_electrons'],
            'total_electron_energy': avalanche_data['total_energy'],
            'photon_yield': photon_yield,
            'energy_radiated': avalanche_data['total_energy'] * 0.8,
            'duration': duration,
            'field_strength': E_field
        }