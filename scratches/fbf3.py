import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

es = np.linspace(1, 4, 100)
fs = np.linspace(0, 1, 100)

def poly(f):
    return 240+1260*f+484.2*f**2-1333.8*f**3-945*f**4+745.2*f**5+113.4*f**6

plt.plot(fs, poly(fs), label='thing')
plt.legend()
plt.show()