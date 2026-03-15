"""
Pomocné funkcie a hlavný simulátor.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from .config import SimulationConfig
from .bremsstrahlung import Bremsstrahlung
from .photo_absorption import PhotoAbsorption
from .energy_balance import EnergyBalance
from .tgf_prediction import TGFPredictor
from .radiation_dose import RadiationDose

class Simulator:
    """Hlavný simulátor - kombinácia všetkých modulov"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.brem = Bremsstrahlung(self.config)
        self.photo = PhotoAbsorption()
        self.energy = EnergyBalance(self.config)
        self.tgf = TGFPredictor(self.config)
        self.dose = RadiationDose(self.config)
        
        self.results = None
    
    def set_energy_range(self, E_min, E_max):
        """Nastavenie energetického rozsahu"""
        self.config.energy_min = E_min
        self.config.energy_max = E_max
    
    def set_gas_properties(self, pressure=None, temperature=None, density=None):
        """Nastavenie vlastností plynu"""
        if pressure is not None:
            self.config.gas_pressure = pressure
        if temperature is not None:
            self.config.gas_temperature = temperature
        if density is not None:
            self.config.gas_density = density
    
    def run(self):
        """Spustí kompletnu simuláciu"""
        
        # Energetické body
        energies = np.logspace(
            np.log10(self.config.energy_min),
            np.log10(self.config.energy_max),
            self.config.energy_bins
        )
        
        # Výsledky
        results = {
            'energies': energies,
            'bremsstrahlung_cross_section': [],
            'energy_loss_rate': [],
            'photon_yield': [],
            'cascade_evolution': []
        }
        
        # Pre každú energiu
        for E in energies:
            # Bremsstrahlung
            sigma = self.brem.total_cross_section(E)
            results['bremsstrahlung_cross_section'].append(sigma)
            
            # Strata energie
            loss = self.energy.electron_energy_loss(E)
            results['energy_loss_rate'].append(loss['total'])
            
            # Výťažok fotónov
            photons = self.tgf.photon_yield(E)
            results['photon_yield'].append(photons)
        
        # Kaskádna evolúcia
        cascade = self.energy.cascade_evolution(self.config.energy_max)
        results['cascade_evolution'] = cascade
        
        self.results = results
        return results
    
    def plot_results(self):
        """Vykreslí výsledky simulácie"""
        
        if self.results is None:
            print("Spustite simuláciu predtým: sim.run()")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        E = self.results['energies']
        
        # Bremmstrahlung cross-section
        axes[0, 0].loglog(E, self.results['bremsstrahlung_cross_section'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Energia elektrónu [MeV]')
        axes[0, 0].set_ylabel('Účinný prierez [barn]')
        axes[0, 0].set_title('Bremsstrahlung - Účinný prierez')
        axes[0, 0].grid(True, which='both', alpha=0.3)
        
        # Strata energie
        axes[0, 1].loglog(E, self.results['energy_loss_rate'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Energia elektrónu [MeV]')
        axes[0, 1].set_ylabel('Strata energie [MeV/cm]')
        axes[0, 1].set_title('Strata energie')
        axes[0, 1].grid(True, which='both', alpha=0.3)
        
        # Výťažok fotónov
        axes[1, 0].plot(E, self.results['photon_yield'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Energia elektrónu [MeV]')
        axes[1, 0].set_ylabel('Počet fotónov na elektrónu')
        axes[1, 0].set_title('Výťažok gama fotónov')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Kaskádna evolúcia
        cascade = self.results['cascade_evolution']
        axes[1, 1].plot(cascade['energies'], 'k-', linewidth=2)
        axes[1, 1].set_xlabel('Časový krok')
        axes[1, 1].set_ylabel('Energia elektrónu [MeV]')
        axes[1, 1].set_title('Evolúcia energetickej kaskády')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """Vytlačí súhrn výsledkov"""
        
        if self.results is None:
            print("Spustite simuláciu predtým: sim.run()")
            return
        
        print("\n" + "="*60)
        print("GAMMA-RENA SIMULAČNÝ REPORT")
        print("="*60)
        
        print("\n📊 Konfigurácia:")
        print(f"  Energetický rozsah: {self.config.energy_min} - {self.config.energy_max} MeV")
        print(f"  Tlak plynu: {self.config.gas_pressure} kPa")
        print(f"  Teplota: {self.config.gas_temperature} K")
        
        E_max = self.config.energy_max
        
        print("\n🔬 Fyzikálne procesy pri E_max = {:.2f} MeV:".format(E_max))
        
        sigma = self.brem.total_cross_section(E_max)
        print(f"  Bremsstrahlung σ_total: {sigma:.2e} barn")
        
        loss = self.energy.electron_energy_loss(E_max)
        print(f"  Strata energie: {loss['total']:.2e} MeV/cm")
        
        mfp = self.brem.mean_free_path(E_max)
        print(f"  Stredná voľná dráha: {mfp:.2e} cm")
        
        photons = self.tgf.photon_yield(E_max)
        print(f"  Výťažok fotónov: {photons:.2f} fotón/elektrónu")
        
        print("\n⚡ TGF Analýza:")
        E_th = self.tgf.threshold_field()
        print(f"  Prahové pole: {E_th:.2f} V/m")
        
        tgf_sim = self.tgf.simulate_tgf_event(E_field=500000)
        if tgf_sim['occurred']:
            print(f"  TGF pravdepodobnosť: Vysoká")
            print(f"  Počet elektrónov v lavíne: {tgf_sim['electron_count']}")
        else:
            print(f"  TGF výskyt: Nepravdepodobný")
        
        print("\n☢️ Radiačné dávky:")
        dose_rate = self.dose.altitude_dose_rate(10000)  # 10 km
        print(f"  Dávková rýchlosť v 10 km: {dose_rate:.2f} mSv/rok")
        
        print("\n" + "="*60 + "\n")