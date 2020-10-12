import numpy as np
from scipy.constants import c
from py_pol.mueller import Mueller
from py_pol.stokes import Stokes
from py_pol.utils import degrees
import matplotlib.pyplot as plt
import pandas
from py_pol.jones_vector import Jones_vector
from py_pol.jones_matrix import Jones_matrix
from functools import reduce
from modules.utils.constants import *


arctan = np.arctan
exp = np.exp
power = np.power
sqrt = np.sqrt
sin = np.sin
cos = np.cos
outer = np.outer

sqrt2 = sqrt(2)
pi = np.pi
MHz = 10 ** 6
GHz = 10 ** 9
THz = 10 ** 12
um = 10 ** -6


# berechnet brechungsindices (l1, l2 keine arrays)
def calc_wp_deltas(l_mat1, l_mat2):

    a = (1 / 3) * power(l_mat1 * l_mat2 * pi / (wls * (l_mat1 + l_mat2)), 2)

    # first order s and p
    wp_eps_s_1 = eps_mat2 * eps_mat1 * (l_mat2 + l_mat1) / (eps_mat2 * l_mat1 + eps_mat1 * l_mat2)

    wp_eps_p_1 = eps_mat1 * l_mat1 / (l_mat2 + l_mat1) + eps_mat2 * l_mat2 / (l_mat2 + l_mat1)

    # 2nd order
    wp_eps_s_2 = wp_eps_s_1 + (a * power(wp_eps_s_1, 3) * wp_eps_p_1 * power((1 / eps_mat1 - 1 / eps_mat2), 2))
    wp_eps_p_2 = wp_eps_p_1 + (a * power((eps_mat1 - eps_mat2), 2))

    # returns
    n_p, n_s = sqrt(abs(wp_eps_p_2) + wp_eps_p_2.real) / sqrt2, sqrt(
        abs(wp_eps_s_2) + wp_eps_s_2.real) / sqrt2
    k_p, k_s = sqrt(abs(wp_eps_p_2) - wp_eps_p_2.real) / sqrt2, sqrt(
        abs(wp_eps_s_2) - wp_eps_s_2.real) / sqrt2

    return n_p, n_s, k_p, k_s


# x is along s (perpendicular to stripes), y along p (parallel to stripes)
def jones_wp(theta, d):
    phi_s, phi_p = 2 * d * n_s * pi / wls, 2 * d * n_p * pi / wls
    alpha_s, alpha_p = -2 * pi * k_s * d / wls, -2 * pi * k_p * d / wls
    x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p

    j00 = exp(x) * cos(theta) ** 2 + exp(y) * sin(theta) ** 2
    j01 = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
    j10 = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
    j11 = exp(x) * sin(theta) ** 2 + exp(y) * cos(theta) ** 2

    return np.array([[j00, j01],
                     [j10, j11]])


def mueller_wp(theta, d):
    phi_s, phi_p = 2 * d * n_s * pi / wls, 2 * d * n_p * pi / wls
    alpha_s, alpha_p = -2 * pi * k_s * d / wls, -2 * pi * k_p * d / wls

    p = sqrt(exp(2 * alpha_s) + exp(2 * alpha_p))
    gamma = arctan(exp(alpha_p) * exp(-alpha_s))
    phi = phi_s - phi_p

    m00 = np.ones(len(wls))
    m01 = cos(2 * gamma) * cos(2 * theta)
    m02 = -cos(2 * gamma) * sin(2 * theta)
    m03 = np.zeros(len(wls))

    m10 = cos(2 * theta) * cos(2 * gamma)
    m11 = cos(2 * theta) ** 2 + cos(phi) * sin(2 * gamma) * sin(2 * theta) ** 2
    m12 = -cos(2 * theta) * sin(2 * theta) + sin(2 * theta) * cos(2 * theta) * cos(phi) * sin(2 * gamma)
    m13 = sin(2 * theta) * sin(2 * gamma) * sin(phi)

    m20 = -cos(2 * gamma) * sin(2 * theta)
    m21 = -sin(2 * theta) * cos(2 * theta) + cos(2 * theta) * sin(2 * theta) * cos(phi) * sin(2 * gamma)
    m22 = sin(2 * theta) ** 2 + sin(2 * gamma) * cos(phi) * cos(2 * theta) ** 2
    m23 = cos(2 * theta) * sin(2 * gamma) * sin(phi)

    m30 = np.zeros(len(wls))
    m31 = -sin(2 * theta) * sin(phi) * sin(2 * gamma)
    m32 = -cos(2 * theta) * sin(phi) * sin(2 * gamma)
    m33 = sin(2 * gamma) * cos(phi)

    m = 0.5 * p ** 2 * np.array([[m00, m01, m02, m03],
                                 [m10, m11, m12, m13],
                                 [m20, m21, m22, m23],
                                 [m30, m31, m32, m33]])

    return m


f_start = 0.150 * THz
f_end = 0.5 * THz

f_pnts = int((f_end - f_start) / (5 * GHz))
freqs = np.linspace(f_start, f_end, f_pnts)

n1 = 1.94  # material
n2 = 1  # luft

# n1**2+0.053j
eps_mat1 = ((n1 ** 2 + 0.053j) * np.ones(freqs.shape, dtype=np.float))
eps_mat2 = ((n2 ** 2 + 0.000j) * np.ones(freqs.shape, dtype=np.float))

wls = c / freqs

angles = np.deg2rad(np.array([45]))
d = np.array([3500]) * um

wp_cnt = len(d)

l_mat1 = 500 * um  # *75*um  # 64.8*um # mat1: nicht luft
l_mat2 = 300 * um  # *61*um  # 45.6*um # mat2: luft

n_p, n_s, k_p, k_s = calc_wp_deltas(l_mat1, l_mat2)

wp1_j = jones_wp(angles[0], d[0])
#wp2_j = jones_wp(angles[1], d[1])

wp1_m = mueller_wp(angles[0], d[0])
#wp2_m = mueller_wp(angles[1], d[1])

x_linear_j = np.array([1, 0])
x_linear_m = np.array([1, 1, 0, 0])

int_x_j, int_y_j, int_x_m, int_y_m = [], [], [], []
for i in range(len(wls)):
    final_j_x = np.dot(np.dot(x_pol_j, wp1_j[:, :, i]), x_linear_j)
    final_j_y = np.dot(np.dot(y_pol_j, wp1_j[:, :, i]), x_linear_j)
    int_x_j.append(final_j_x[0]*np.conjugate(final_j_x[0]) + final_j_x[1]*np.conjugate(final_j_x[1]))
    int_y_j.append(final_j_y[0]*np.conjugate(final_j_y[0]) + final_j_y[1]*np.conjugate(final_j_y[1]))

    final_m_x = np.dot(np.dot(x_pol_m, wp1_m[:, :, i]), x_linear_m)
    final_m_y = np.dot(np.dot(y_pol_m, wp1_m[:, :, i]), x_linear_m)
    int_x_m.append(final_m_x[0])
    int_y_m.append(final_m_y[0])

int_x_j, int_y_j = 10*np.log10(np.array(int_x_j)), 10*np.log10(np.array(int_y_j))
int_x_m, int_y_m = 10*np.log10(np.array(int_x_m)), 10*np.log10(np.array(int_y_m))

#plt.plot(freqs, int_x_j, label='int_x_j')
#plt.plot(freqs, int_y_j, label='int_y_j')

#plt.plot(freqs, int_x_m, label='int_x_m')
#plt.plot(freqs, int_y_m, label='int_y_m')

plt.plot(freqs, n_s**2, label='eps_s')
plt.plot(freqs, n_p**2, label='eps_p')

plt.legend()
plt.show()

print(wp1_j.shape)
