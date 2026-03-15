"""
Kalkulácia radiačnej záťaže a ekvivalentnej dávky.
"""

import numpy as np
from .photo_absorption import PhotoAbsorption
from .config import SimulationConfig

class RadiationDose:
    """Modeluje radiačnú záťaž"""
    
    # Quality factors (RBE)
    QUALITY_FACTORS = {
        'photon': 1.0,
        'electron': 1.0,
        'proton': 2.0,
        'alpha': 20.0,
        'neutron': 3.0  # priemer
    }
    
    # Organ weighting factors
    ORGAN_FACTORS = {
        'bone_marrow': 0.12,
        'colon': 0.12,
        'lung': 0.12,
        'stomach': 0.12,
        'breast': 0.12,
        'remainder': 0.16,
        'thyroid': 0.04,
        'bladder': 0.04,
        'esophagus': 0.04,
        'liver': 0.04,
        'bone_surface': 0.01,
        'skin': 0.01,
        'lens': 0.01
    }
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.photo = PhotoAbsorption()
    
    def absorbed_dose(self, energy_deposited, mass):
        """
        Absorbovaná dávka (Gray).
        
        Parameters:
        -----------
        energy_deposited : float
            Depozitovaná energia [J]
        mass : float
            Hmotnosť tkaniva [kg]
        
        Returns:
        --------
        float
            Absorbovaná dávka [Gy]
        """
        
        if mass <= 0:
            return 0.0
        
        dose = energy_deposited / mass  # J/kg = Gy
        
        return max(0, dose)
    
    def equivalent_dose(self, absorbed_dose, particle_type='photon'):
        """
        Ekvivalentná dávka (Sievert).
        
        Parameters:
        -----------
        absorbed_dose : float
            Absorbovaná dávka [Gy]
        particle_type : str
            Typ čiastice
        
        Returns:
        --------
        float
            Ekvivalentná dávka [Sv]
        """
        
        Q = self.QUALITY_FACTORS.get(particle_type, 1.0)
        equiv_dose = absorbed_dose * Q  # Sv
        
        return max(0, equiv_dose)
    
    def effective_dose(self, organ_doses):
        """
        Efektívna dávka (whole body equivalent).
        
        Parameters:
        -----------
        organ_doses : dict
            {organ_name: dose_in_Sv}
        
        Returns:
        --------
        float
            Efektívna dávka [Sv]
        """
        
        eff_dose = 0.0
        
        for organ, dose in organ_doses.items():
            weight = self.ORGAN_FACTORS.get(organ, 0.0)
            eff_dose += weight * dose
        
        return eff_dose
    
    def dose_rate(self, photon_flux, energy_spectrum, exposure_time=1.0):
        """
        Dávková rýchlosť.
        
        Parameters:
        -----------
        photon_flux : float
            Tok fotónov [fotón/m²*s]
        energy_spectrum : np.ndarray
            Spektrum energií [MeV]
        exposure_time : float
            Čas expozície [s]
        
        Returns:
        --------
        dict
            Dávkové údaje
        """
        
        # Priemerne energia fotónu
        mean_energy = np.mean(energy_spectrum)
        
        # Celková energia na jednotku plochy a času
        energy_flux = photon_flux * mean_energy  # MeV/(m²*s)
        
        # Konverzia na W/m²
        energy_flux_si = energy_flux * 1.602e-13  # W/m²
        
        # Dávka v Gy (hrubá aproximácia)
        dose_rate_gy_s = energy_flux_si / 1000  # Gy/s
        
        # Celková dávka
        total_dose = dose_rate_gy_s * exposure_time
        
        return {
            'dose_rate': dose_rate_gy_s,
            'total_dose': total_dose,
            'mean_photon_energy': mean_energy,
            'photon_flux': photon_flux
        }
    
    def altitude_dose_rate(self, altitude_m, solar_activity=1.0):
        """
        Dávková rýchlosť v letektve (reálne dáta).
        
        Parameters:
        -----------
        altitude_m : float
            Nadmorská výška [m]
        solar_activity : float
            Index slnečnej aktivity (1.0 = normál)
        
        Returns:
        --------
        float
            Dávková rýchlosť [mSv/rok]
        """
        
        # Empirické údaje z leteckého výskumu
        # Na úrovni mora: ~0.03 mSv/rok
        # V letectve: ~0.05-0.1 mSv/rok na 10 km
        
        if altitude_m < 1000:
            dose_rate = 0.03
        elif altitude_m < 5000:
            dose_rate = 0.04 + (altitude_m - 1000) / 4000 * 0.02
        elif altitude_m < 10000:
            dose_rate = 0.06 + (altitude_m - 5000) / 5000 * 0.03
        else:
            dose_rate = 0.09 + (altitude_m - 10000) / 30000 * 0.01
        
        # Korekcija solárnej aktivity
        dose_rate *= solar_activity
        
        return dose_rate
    
    def health_risk_assessment(self, dose_sv):
        """
        Posúdenie zdravotného rizika podľa dávky.
        
        Parameters:
        -----------
        dose_sv : float
            Dávka v Sievertoch
        
        Returns:
        --------
        dict
            Zdravotné rizika
        """
        
        risk_levels = {
            'dose': dose_sv,
            'dose_category': '',
            'health_effects': [],
            'severity': 'none'
        }
        
        if dose_sv < 0.05:
            risk_levels['dose_category'] = 'Bezpečná'
            risk_levels['health_effects'] = ['Žiadne pozorovateľné účinky']
            risk_levels['severity'] = 'none'
        elif dose_sv < 0.25:
            risk_levels['dose_category'] = 'Nízka'
            risk_levels['health_effects'] = ['Minimálne biologické zmeny']
            risk_levels['severity'] = 'minimal'
        elif dose_sv < 1.0:
            risk_levels['dose_category'] = 'Stredná'
            risk_levels['health_effects'] = [
                'Náusenosť', 'Únava',
                'Zvýšené riziko rakoviny'
            ]
            risk_levels['severity'] = 'moderate'
        elif dose_sv < 6.0:
            risk_levels['dose_category'] = 'Vysoká'
            risk_levels['health_effects'] = [
                'Akútny radiačný syndróm',
                'Zvýšená mortalita'
            ]
            risk_levels['severity'] = 'severe'
        else:
            risk_levels['dose_category'] = 'Fatálna'
            risk_levels['health_effects'] = ['Vysoká pravdepodobnosť smrti']
            risk_levels['severity'] = 'fatal'
        
        return risk_levels