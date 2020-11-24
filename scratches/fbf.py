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

def eq(c, f1, f2):
    eps_tm = eps / (f1 + eps * f2)
    eps_te = f1 * eps + f2
    res = eps_te + (1 / 3) * c * (eps - 1) ** 2 - eps_tm - (1 / 3) * c * ((1 / eps) - 1) ** 2 * eps_tm ** 3 * eps_te
    #res2 = np.sqrt(eps_te) - np.sqrt(eps_tm)
    #res3 = np.sqrt((1 / 3) * c * (eps - 1) ** 2) - np.sqrt((1 / 3) * c * ((1 / eps) - 1) ** 2 * eps_tm ** 3 * eps_te)
    return res

e = 4
eps = e
f = 0.69

e_tm = e / (f*(1-e)+e)
e_te = f*(e-1)+1

def ete(eps):
    return 1+f*(eps-1)
def etm(eps):
    return eps / (f * (1 - eps) + eps)

r = 0.6
d = (1/3)*(r*pi*f*(1-f))**2
c = (r * pi * f * (1-f)) ** 2

es = np.linspace(1, 5, 100)
lhs = (ete(es)-etm(es))/(d*(es-1)**2) + 1 - ete(es)*etm(es)**3/es**2

up = ete(es)+(1/3)*(r*pi*f*(1-f)*(es-1))**2
down = etm(es)+(1/3)*(r*pi*f*(1-f)*(1/es - 1))**2*ete(es)*etm(es)**3
diff1 = ete(es)+(1/3)*(r*pi*f*(1-f)*(es-1))**2 - etm(es)-(1/3)*(r*pi*f*(1-f)*(1/es - 1))**2*ete(es)*etm(es)**3
d = (1/3)*(r*pi*f*(1-f))**2
diff2 = (ete(es) - etm(es))/(d*(es-1)**2) + 1 - ete(es)*etm(es)**3/es**2
e = es
t3 = (f**2*(e**2-1)+f*(-e**2+4*e+1)-2*e)*c*f+(f*(1-e)+e)**2

print(eq(c, f, 1-f))
#plt.plot(es, up, label='lhs')
#plt.plot(es, down, label='rhs')
#plt.plot(es, diff1, label='diff')
#plt.plot(es, diff2, label='help me')
#plt.plot(es, lhs, label='wtf is going on')
plt.plot(es, t3, label='e dep')
plt.legend()
plt.show()

exit(129042)

for eps in (np.arange(1,4.5,0.5)):
    grid = np.zeros((100, 100))
    for i, lwr in enumerate(np.linspace(0, 0.61, 100)):
        for j, f in enumerate(np.linspace(0, 1, 100)):
            #f2 = 1 - f1
            c = (lwr * pi * f * (1-f)) ** 2
            val = eq(c, f, 1-f)
            if val < 0:
                print(f, 1-f)
                print(val)
                val = -5
            grid[i, j] = val
    lwr = np.linspace(0, 1, 100)
    f1 = np.linspace(0, 1, 100)
    plt.pcolormesh(lwr,f1, grid, cmap="jet")
    plt.xlabel("f1_mat (%)")
    plt.ylabel("L/l (%)")
    plt.colorbar()
    plt.title(f"Eps: {eps}")
    plt.show()