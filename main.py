import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

class GammaRENA_V3:
    def __init__(self, root):
        self.root = root
        self.root.title("Gamma-RENA v3.0: 3D Monte Carlo & Optimizer")
        self.root.geometry("1400x900")
        
        self.E0 = 7.3e6
        self.materials = {
            "Vzduch (STP)": {"Et": 2.84e5, "rho": 1.225, "beta": 1e-4},
            "PMMA (Plexi)": {"Et": 0.213e9, "rho": 1180, "beta": 1e-4},
            "BGO Kryštál":  {"Et": 0.865e9, "rho": 7130, "beta": 5e-3},
        }

        self.setup_ui()

    def setup_ui(self):
        # Layout
        left_panel = ttk.Frame(self.root, padding="15")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        right_panel = ttk.Frame(self.root, padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Ovládanie
        ctrl_frame = ttk.LabelFrame(left_panel, text=" Nastavenia experimentu ", padding="10")
        ctrl_frame.pack(fill=tk.X)

        ttk.Label(ctrl_frame, text="Materiál:").pack(anchor="w")
        self.mat_var = tk.StringVar(value="BGO Kryštál")
        self.mat_combo = ttk.Combobox(ctrl_frame, textvariable=self.mat_var, values=list(self.materials.keys()))
        self.mat_combo.pack(fill=tk.X, pady=5)

        ttk.Label(ctrl_frame, text="Hrúbka (cm):").pack(anchor="w")
        self.dist_scale = ttk.Scale(ctrl_frame, from_=0.1, to=50, orient=tk.HORIZONTAL)
        self.dist_scale.set(10)
        self.dist_scale.pack(fill=tk.X)

        # --- NOVÉ FUNKCIE ---
        opt_frame = ttk.LabelFrame(left_panel, text=" Nástroje ", padding="10")
        opt_frame.pack(fill=tk.X, pady=10)

        ttk.Button(opt_frame, text="🔍 Nájsť optimálnu hrúbku", command=self.optimize_thickness).pack(fill=tk.X, pady=2)
        ttk.Button(opt_frame, text="⚡ Simulovať 3D trajektórie", command=self.simulate_3d).pack(fill=tk.X, pady=2)

        self.res_text = tk.Text(left_panel, height=15, width=40, font=("Consolas", 9))
        self.res_text.pack(pady=10)

        # Grafy (2 subplots: 2D Gain a 3D Trajektórie)
        self.fig = plt.figure(figsize=(8, 10))
        self.ax2d = self.fig.add_subplot(211)
        self.ax3d = self.fig.add_subplot(212, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_physics_params(self):
        mat = self.materials[self.mat_var.get()]
        dist_m = float(self.dist_scale.get()) / 100
        return mat['Et'], mat['beta'], dist_m

    def optimize_thickness(self):
        """Hľadá minimálnu hrúbku d, pri ktorej G >= 1 pri 1.5x Et."""
        Et, beta, _ = self.get_physics_params()
        target_field = Et * 1.5
        lr = (self.E0 / target_field) / (1.0 - Et / target_field)
        
        # Rovnica: 1 = beta * exp(d_opt / lr)  => d_opt = ln(1/beta) * lr
        d_opt_m = np.log(1.0 / beta) * lr
        d_opt_cm = d_opt_m * 100

        self.dist_scale.set(d_opt_cm)
        self.res_text.insert(tk.END, f"\n[OPTIMALIZÁCIA]\nCieľ: G=1 pri 1.5*Et\nNavrhovaná hrúbka: {d_opt_cm:.2f} cm\n")
        self.update_plots_base()

    def simulate_3d(self):
        """Simuluje náhodný rozptyl častíc v 3D priestore."""
        Et, beta, dist_m = self.get_physics_params()
        self.ax3d.clear()
        
        # Simulujeme 10 častíc
        for _ in range(10):
            steps = 100
            z = np.linspace(0, dist_m, steps)
            # Náhodný rozptyl (Brownov pohyb simulujúci kolízie)
            x = np.cumsum(np.random.normal(0, 0.002, steps))
            y = np.cumsum(np.random.normal(0, 0.002, steps))
            
            self.ax3d.plot(x, y, z, alpha=0.7)

        self.ax3d.set_title("3D Model rozptylu elektrónov")
        self.ax3d.set_xlabel("X (m)")
        self.ax3d.set_ylabel("Y (m)")
        self.ax3d.set_zlabel("Hĺbka (m)")
        self.canvas.draw()

    def update_plots_base(self):
        # Základný 2D graf (podobný ako v v2.0)
        Et, beta, dist_m = self.get_physics_params()
        fields = np.linspace(Et * 1.05, Et * 3, 200)
        lr = (self.E0 / fields) / (1.0 - Et / fields)
        gains = beta * np.exp(dist_m / lr)

        self.ax2d.clear()
        self.ax2d.semilogy(fields/1e6, gains, label="Gain")
        self.ax2d.axhline(1, color='red', ls='--')
        self.ax2d.set_title("Analýza zisku")
        self.ax2d.set_xlabel("Pole (MV/m)")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = GammaRENA_V3(root)
    root.mainloop()