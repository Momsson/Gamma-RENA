import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve

# =================================================================
# 1. FYZIKÁLNE DÁTA (Pasko 2023, 2026)
# =================================================================
MATERIALS = {
    "BGO": {
        "lr_min": 0.011e-2, "Et": 8.65e8, "gamma": 1.0e-3, "Z": 83, "rho": 7.13,
        "name": "Bismuth Germanate"
    },
    "Quartz": {
        "lr_min": 0.046e-2, "Et": 3.63e8, "gamma": 1.5e-3, "Z": 14, "rho": 2.65,
        "name": "Silicon Dioxide"
    },
    "Air (SL)": {
        "lr_min": 100.0, "Et": 2.76e5, "gamma": 1.0e-4, "Z": 7.4, "rho": 0.0012,
        "name": "Atmospheric Air"
    }
}

class ResearchEngine:
    def __init__(self, mat_key="BGO"):
        self.set_material(mat_key)
        self.c = 3e8  # Rýchlosť svetla

    def set_material(self, key):
        self.m = MATERIALS[key]
        self.key = key

    # --- PILIER A: STABILITA (INCEPTION) ---
    def get_lambda_r(self, E):
        """Kineticky korigovaná dĺžka lavíny."""
        if E <= self.m["Et"]: return 1e10
        return self.m["lr_min"] / (1 - (self.m["Et"] / E))

    def solve_inception(self, e_ratio_range):
        """Vypočíta kritickú hrúbku L pre rozsah polí E/Et."""
        L_crits = []
        for er in e_ratio_range:
            lr = self.get_lambda_r(er * self.m["Et"])
            # Podmienka: L = lr * ln(1/gamma + 1)
            L_crits.append(lr * np.log(1/self.m["gamma"] + 1))
        return np.array(L_crits)

    # --- PILIER B: SPEKTRÁLNA ANALÝZA ---
    def bremsstrahlung_spectrum(self, energies):
        """Zjednodušený model spektrálnej hustoty (Kramersov zákon)."""
        E_max = 10.0 # MeV (typické pre RREA)
        spectrum = np.where(energies < E_max, (E_max - energies) / energies, 0)
        # Útlm v materiáli (Mass attenuation coefficient aproximácia)
        mu = 0.05 * (self.m["Z"]**2) / (energies**2.5 + 0.1)
        attenuation = np.exp(-mu * self.m["rho"] * 0.01) # Pre 1cm vzorku
        return spectrum * attenuation

    # --- PILIER C: ČASOVÁ DYNAMIKA ---
    def simulate_pulse(self, L, E_ratio, duration_ns=100):
        """Simulácia nárastu počtu elektrónov v čase."""
        E = E_ratio * self.m["Et"]
        lr = self.get_lambda_r(E)
        v_drift = 0.9 * self.c # Relativistická rýchlosť
        
        t = np.linspace(0, duration_ns * 1e-9, 1000)
        # N(t) = exp(v*t / lr) * termín spätnej väzby
        # Zjednodušený model exponenciálneho rastu pri uzavretej väzbe
        growth_rate = v_drift / lr
        n_t = np.exp(growth_rate * t)
        return t * 1e9, n_t

# =================================================================
# 2. VÝSKUMNÝ PANEL (VÝSTUPY)
# =================================================================
def run_full_analysis():
    engine = ResearchEngine("BGO")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#f4f4f4')

    # GRAF 1: MAPA STABILITY (Inception Map)
    e_ratios = np.linspace(1.1, 2.5, 100)
    for mat in MATERIALS:
        engine.set_material(mat)
        lc = engine.solve_inception(e_ratios)
        scale = 100 if "Air" not in mat else 1
        axs[0, 0].plot(e_ratios, lc * scale, label=mat, lw=2)
    
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title("A: Fázová mapa stability (Inception)", fontweight='bold')
    axs[0, 0].set_ylabel("Kritická dĺžka L [cm / m]")
    axs[0, 0].set_xlabel("Prepätie E/Et")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # GRAF 2: SPEKTRUM ŽIARENIA
    engine.set_material("BGO")
    energies = np.linspace(0.1, 10, 200)
    spec = engine.bremsstrahlung_spectrum(energies)
    axs[0, 1].fill_between(energies, spec, color='red', alpha=0.3)
    axs[0, 1].plot(energies, spec, color='red', lw=2)
    axs[0, 1].set_title("B: Energetické spektrum fotónov (BGO)", fontweight='bold')
    axs[0, 1].set_xlabel("Energia [MeV]")
    axs[0, 1].set_ylabel("Relatívna intenzita (Counts)")
    axs[0, 1].grid(True, alpha=0.3)

    # GRAF 3: ČASOVÝ PULZ (OSCILOSKOP)
    t, n = engine.simulate_pulse(L=0.01, E_ratio=1.3)
    axs[1, 0].plot(t, n, color='blue', lw=2)
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title("C: Dynamika pulzu (Log scale)", fontweight='bold')
    axs[1, 0].set_xlabel("Čas [ns]")
    axs[1, 0].set_ylabel("Počet elektrónov N(t)")
    axs[1, 0].grid(True, alpha=0.3)

    # GRAF 4: POROVNANIE MATERIÁLOVÝCH PARAMETROV
    names = list(MATERIALS.keys())
    ets = [MATERIALS[m]["Et"]/1e6 for m in names]
    bars = axs[1, 1].bar(names, ets, color=['orange', 'cyan', 'green'])
    axs[1, 1].set_title("D: Prahové polia Et materiálov", fontweight='bold')
    axs[1, 1].set_ylabel("Et [MV/cm pre tuhé, kV/m pre vzduch]")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_analysis()