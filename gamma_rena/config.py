"""
Konfiguračné parametre a nastavenia pre Gamma-RENA simulátor.
"""

class SimulationConfig:
    """Konfigurácia simulácie"""
    
    def __init__(self):
        # Energetické rozsahy [MeV]
        self.energy_min = 0.1
        self.energy_max = 10.0
        self.energy_bins = 100
        
        # Vlastnosti plynu
        self.gas_pressure = 101.325  # kPa (normálny tlak)
        self.gas_temperature = 293.15  # K (20°C)
        self.gas_density = 1.225  # kg/m³ (suchý vzduch pri 20°C)
        
        # Parametry elektrónov
        self.electron_energy = 1.0  # MeV
        self.electron_count = 1e6
        self.electron_momentum = 0.0
        
        # Vlastnosti materiálu
        self.material_type = "air"  # "air", "water", "ice"
        self.material_thickness = 1.0  # cm
        self.material_density = 1.225  # g/cm³
        
        # Časové parametre
        self.time_step = 1e-6  # s
        self.total_time = 1e-3  # s
        
        # TGF parametre
        self.tgf_threshold = 0.5  # MeV
        self.tgf_altitude = 15000  # m
        
        # Radiačné parametre
        self.dose_calculation = True
        self.dose_type = "absorbed"  # "absorbed" alebo "equivalent"
        
    def to_dict(self):
        """Vráti konfiguráciu ako slovník"""
        return self.__dict__
    
    def update(self, **kwargs):
        """Aktualizuje konfiguráciu"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Neznámy parameter: {key}")