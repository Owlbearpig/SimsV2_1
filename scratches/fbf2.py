import numpy as np
from numpy import pi
import matplotlib.pyplot as plt


def ete(eps):
    return 1+f*(eps-1)
def etm(eps):
    return eps / (f * (1 - eps) + eps)

es = np.linspace(1, 4, 100)
fs = np.linspace(0, 1, 100)




grid = np.zeros((100, 100))
for i, e in enumerate(es):
    for j, f in enumerate(fs):
        r = 0.6
        d = (1/3)*r ** 2 * pi ** 2 * f ** 2 * (1 - f) ** 2
        c = (1/3)*r ** 2 * pi ** 2
        val = -(f-1)**2*f/((f*(1-e)+e)**2*d) - (f**2*(e**2-1)+f*(-e**2+4*e+1)-2*e)/(f*(1-e)+e)**4
        t = e**2*(3*c*f**2+f)+2*e-2*e**2-6*c*f**2-2*e*c-4*f*e
        t2 = 2*e*(c*f**3+f**2+1)+2*f-4*e*f-2*f*c-2*f**2
        t3 = c*f**3*e**2+4*c*f**2*e+c*f**2+f**2+f**2*e**2+e**2-2*f*e**2-f**3*c-e**2*c*f**2-2*e*c*f
        t4 = (f**2*(e**2-1)+f*(-e**2+4*e+1)-2*e)*c*f+(f*(1-e)+e)**2
        t5 = 2*c*f**3*e+4*c*f**2+2*f**2*e+2*e-4*e*f-2*e*f**2*c-2*c*f
        t6 = 6*c*f**2*e+8*c*f+4*f*e-4*e-4*e*f*c-2*c
        t7 = 2*c*f**2*e*(f-1)+2*e*(1+f**2-2*f)+2*c*f*(2*f-1)
        val = t7
        if val > 0:
            val = val
        if val < 0:
            val = val#-1
        grid[i, j] = val

plt.imshow(grid)
plt.colorbar()
plt.show()