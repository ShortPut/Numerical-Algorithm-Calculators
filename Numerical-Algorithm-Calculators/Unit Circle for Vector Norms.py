import numpy as np
import matplotlib.pyplot as plt

# Create array of 181 values from 0 to 2pi
theta = np.linspace(0, 2 * np.pi, 181)

c = np.cos(theta)
s = np.sin(theta)

absc = np.absolute(c)
abss = np.absolute(s)

# CHOOSE norm p here
p = 1

# Compute radial distance in polar coordinates
r = 1 / (absc ** p + abss ** p) ** (1 / p)

# Plot unit circle of p-th norm on cartesian graph
plt.plot(r * c, r * s)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Unit Circle for p-th norm")
plt.grid(True)
plt.show()
