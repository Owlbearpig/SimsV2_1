import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c


abs_coeff = np.array([2.5, 5, 6, 11, 13, 16, 20, 22])*100
freqs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])*10**12
eps_i = abs_coeff*c / (4*np.pi*freqs)
print(freqs, eps_i)
from scipy.interpolate import interp1d
#f2 = interp1d(freqs, eps_i, kind='cubic', fill_value="extrapolate")
new_series = np.polynomial.polynomial.Polynomial.fit(freqs, eps_i, 2)
poly_coeffs = new_series.convert().coef
def fit(poly_coeffs):
    def f(x):
        res = 0
        for i, coeff in enumerate(list(poly_coeffs)):
            res += coeff*x**i
        return res
    return f

f2 = fit(poly_coeffs)

plt.plot(freqs, eps_i)
plt.plot(freqs, f2(freqs))
plt.show()

freqs_ = np.linspace(0.2, 2.0, 190)*10**12

header = 'freq,ref_ind,epsilon_r,epsilon_i\n'

f = open('new_pla_data.csv', 'w')
f.write(header)
for freq in freqs_:
    s = f'{freq},{1.89},{3.5721},{f2(freq)}\n'
    f.write(s)

f.close()
