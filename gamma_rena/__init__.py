"""
Gamma-RENA: Advanced kinetics model of relativistic runaway electrons 
and gamma-ray emission simulation tool.
"""

__version__ = "0.1.0"
__author__ = "Momsson"

from .bremsstrahlung import Bremsstrahlung
from .photo_absorption import PhotoAbsorption
from .energy_balance import EnergyBalance
from .tgf_prediction import TGFPredictor
from .radiation_dose import RadiationDose
from .utils import Simulator

__all__ = [
    'Bremsstrahlung',
    'PhotoAbsorption',
    'EnergyBalance',
    'TGFPredictor',
    'RadiationDose',
    'Simulator'
]