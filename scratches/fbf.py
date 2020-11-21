from numpy import pi
import numpy as np
import matplotlib.pyplot as plt


"""
l1 = 7
l2 = 2
l = l1 + l2
lwr = 0.5 # l to wavelength ratio
f1, f2 = l1/l, l2/l
"""



#rhs = f1**2*eps**3+f2**2*eps+2*f1*eps**2*f2+(1/c)*(1-eps+(eps-1)**2 / (f1+eps*f2)**2)

#print((1/eps - 1)**2)
#print((eps-1)**2)
#print(c)eps_te
#print(eps_tm**3)
#print(eps_te)

def rhs(c, f1, f2):
    eps_tm = eps / (f1 + eps * f2)
    eps_te = f1 * eps + f2
    res = np.sqrt(eps_te + (1 / 3) * c * (eps - 1) ** 2) - np.sqrt(eps_tm + (1 / 3) * c * ((1 / eps) - 1) ** 2 * eps_tm ** 3 * eps_te)
    res2 = np.sqrt(eps_te) - np.sqrt(eps_tm)
    return res

for n in (np.arange(1,4,0.5)):
    eps = n**2
    grid = np.zeros((100, 100))
    for i, lwr in enumerate(np.linspace(0, 1, 100)):
        for j, f1 in enumerate(np.linspace(0, 1, 100)):
            f2 = 1 - f1
            c = (lwr * pi * f1 * f2) ** 2
            val = rhs(c, f1, f2)
            if val < 0:
                print(f1, f2)
                print(val)
            if abs(val) < 10:
                grid[i, j] = val
    lwr = np.linspace(0, 1, 100)
    f1 = np.linspace(0, 1, 100)
    plt.pcolormesh(lwr,f1, grid, cmap="jet")
    plt.xlabel("f1_mat (%)")
    plt.ylabel("L/l (%)")
    plt.colorbar()
    plt.title("Refractive index: " + str(n))
    plt.show()