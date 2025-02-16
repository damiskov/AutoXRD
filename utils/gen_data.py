import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Sample XRD data
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, x.size)

# Detect peaks
peaks, _ = find_peaks(y, height=0)

# Plotting
plt.plot(x, y)
plt.plot(x[peaks], y[peaks], "x")
plt.title('Peak Detection in XRD Data')
plt.xlabel('2θ (degrees)')
plt.ylabel('Intensity')
plt.show()