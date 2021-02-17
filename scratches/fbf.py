from numpy import pi
import numpy as np
import matplotlib.pyplot as plt


#rhs = f1**2*eps**3+f2**2*eps+2*f1*eps**2*f2+(1/c)*(1-eps+(eps-1)**2 / (f1+eps*f2)**2)

#print((1/eps - 1)**2)
#print((eps-1)**2)
#print(c)eps_te
#print(eps_tm**3)
#print(eps_te)

def ri(eps):
    abs_val = np.sqrt(eps.real**2+eps.imag**2)
    return np.sqrt(abs_val + eps.real) / np.sqrt(2)

def eq(c, f1, f2, eps):
    eps_tm = eps / (f1 + eps * f2)
    eps_te = f1 * eps + f2
    eps_p = eps_te + (1 / 3) * c * (eps - 1) ** 2
    eps_s = eps_tm + (1 / 3) * c * ((1 / eps) - 1) ** 2 * eps_tm ** 3 * eps_te
    res = eps_p - eps_s
    dn = ri(eps_p) - ri(eps_s)
    #res2 = np.sqrt(eps_te) - np.sqrt(eps_tm)
    #res3 = np.sqrt((1 / 3) * c * (eps - 1) ** 2) - np.sqrt((1 / 3) * c * ((1 / eps) - 1) ** 2 * eps_tm ** 3 * eps_te)

    return dn


lwr = 0.6
grid = np.zeros((100, 100))
for i, eps in enumerate(np.linspace(1.05, 4, 100)):
    for j, f in enumerate(np.linspace(0, 1, 100)):
        c = (lwr * pi * f * (1-f)) ** 2
        val = eq(c, f, 1-f, eps + 0.045j)
        if val < 0:
            print(f, 1-f)
            print(val)
            val = -5
        grid[i, j] = val

eps = np.linspace(1.05, 4, 100)
f1 = np.linspace(0, 1, 100)
#plt.imshow(grid)
plt.pcolormesh(f1, eps, grid, cmap="jet")
plt.xlabel("f")
plt.ylabel("eps")
plt.colorbar()
#plt.title(f"Eps: {eps}")
plt.show()