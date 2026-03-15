# Basic Simulation

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
time = np.linspace(0, 10, 100)
output = np.sin(time)

# Plotting the simulation results
plt.plot(time, output)
plt.title('Basic Simulation')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.show()