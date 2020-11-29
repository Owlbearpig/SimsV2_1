import numpy as np

from numpy import exp, pi, log10
from scipy.constants import c

freq = 1.0*10**12
wl = c/freq

eps_3eck = 3.75411825e+000 + 3.74745906e-002j
eps_4eck = 3.76539922e+000 + 2.64906370e-002j

n_r_3eck = np.sqrt((np.abs(eps_3eck)+eps_3eck.real)/2)
n_r_4eck = np.sqrt((np.abs(eps_4eck)+eps_4eck.real)/2)

T_3eck = 1-np.abs((1-n_r_3eck)/(1+n_r_3eck))**2
T_4eck = 1-np.abs((1-n_r_4eck)/(1+n_r_4eck))

alpha_3eck = (4*pi/wl)*(np.sqrt((np.abs(eps_3eck)-eps_3eck.real)/2))
alpha_4eck = (4*pi/wl)*(np.sqrt((np.abs(eps_4eck)-eps_4eck.real)/2))

d = 0.0025  # 2.5 mm

eck3 = T_3eck*exp(-alpha_3eck*d)
eck4 = T_4eck*exp(-alpha_4eck*d)

print(10*log10(eck3), 10*log10(eck4))
