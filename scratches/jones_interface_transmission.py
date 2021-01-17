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
from sympy.solvers import nsolve
from sympy import Symbol, tan, Rational
eps = Symbol("eps")


dot = np.dot
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


# calculate ri (l1, l2 float)
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

    return n_s, n_p, k_s, k_p


def rytov_two_layer_numerical(l1, l2, eps1, eps2, f, x0):
    k = 2 * pi * f / c

    r = Rational(1, 2)

    # parallel (first return value) (p-pol.)
    eps_p = nsolve(
        k * (eps2 - eps) ** r * tan(k * (eps2 - eps) ** r * l2 * r) + k * (
                    eps1 - eps) ** r * tan(k * (eps1 - eps) ** r * l1 * r)
        , eps, x0[0])

    # perpendicular (second return value) (s-pol.)
    eps_s = nsolve(
        (1/eps2) * k * (eps2 - eps) ** r * tan(k * (eps2 - eps) ** r * l2 * r) + (1/eps1) * k * (
                    eps1 - eps) ** r * tan(k * (eps1 - eps) ** r * l1 * r)
        , eps, x0[1])

    return  eps_s ** 0.5, eps_p ** 0.5


def real_ri(l_mat1, l_mat2, eps1, eps2, frequencies):
    n_s, n_p, k_s, k_p = calc_wp_deltas(l_mat1, l_mat2)
    n_s, n_p = n_s.real, n_p.real

    prev_sol = [n_s[0], n_p[0]]

    n_p_n, n_s_n = [], []
    for i, freq in enumerate(frequencies):
        print(i, len(frequencies))

        for delta in np.linspace(0, 1.5, 10):
            start_val = (prev_sol[0] + delta, prev_sol[1] + delta)
            try:
                num_res = rytov_two_layer_numerical(l_mat1, l_mat2, eps1, eps2, freq, x0=start_val)

                if all([i ** 2 > 0 for i in num_res]):
                    if num_res[0] > 1 and num_res[1] > 1:
                        n_s_n.append(np.float(num_res[0]))
                        n_p_n.append(np.float(num_res[1]))
                        break
            except ValueError:  # not all start values lead to convergence (secant method)
                continue

        prev_sol = (n_p_n[-1], n_s_n[-1])

    return np.array(n_s_n), np.array(n_p_n)


# x is along s (perpendicular to stripes), y along p (parallel to stripes)
def jones_wp_no_t_loss(theta, d, n_s, n_p):
    phi_s, phi_p = 2 * d * n_s * pi / wls, 2 * d * n_p * pi / wls
    alpha_s, alpha_p = -2 * pi * k_s * d / wls, -2 * pi * k_p * d / wls
    x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p

    j00 = exp(x) * cos(theta) ** 2 + exp(y) * sin(theta) ** 2
    j01 = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
    j10 = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
    j11 = exp(x) * sin(theta) ** 2 + exp(y) * cos(theta) ** 2

    return np.array([[j00, j01],
                     [j10, j11]])


# works for one wp
def jones_wp_t_loss(theta, d, n_s, n_p):
    phi_s, phi_p = 2 * d * n_s * pi / wls, 2 * d * n_p * pi / wls
    alpha_s, alpha_p = -2 * pi * k_s * d / wls, -2 * pi * k_p * d / wls
    x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p
    t_x, t_y = (4 * n_p * n_s ** 2) / (n_s ** 2 + n_p) ** 2, (4 * n_p ** 3) / (n_p + n_p ** 2) ** 2

    j00 = t_x * exp(x) * cos(theta) ** 2 + t_y * exp(y) * sin(theta) ** 2
    j01 = 0.5 * sin(2 * theta) * (t_y * exp(y) - t_x * exp(x))
    j10 = 0.5 * sin(2 * theta) * (t_y * exp(y) - t_x * exp(x))
    j11 = t_x * exp(x) * sin(theta) ** 2 + t_y * exp(y) * cos(theta) ** 2

    return np.array([[j00, j01],
                     [j10, j11]])


# calculate transmission matrix (E = T*E_0(lab system)). alpha, theta angle of wp initially and transmitted
def transmission_matrix(alpha, theta, n_s, n_p):
    a_1, a_2 = n_s ** 2 * cos(theta) ** 2 + n_p ** 2 * sin(theta) ** 2, \
               n_s ** 2 * cos(alpha) ** 2 + n_p ** 2 * sin(alpha) ** 2
    b_1, b_2 = 0.5 * sin(2 * theta) * (n_p ** 2 - n_s ** 2), \
               0.5 * sin(2 * alpha) * (n_p ** 2 - n_s ** 2)
    c_1, c_2 = sin(theta) ** 2 * n_s ** 2 + cos(theta) ** 2 * n_p ** 2, \
               sin(alpha) ** 2 * n_s ** 2 + cos(alpha) ** 2 * n_p ** 2
    a, b, c_ = a_1 + a_2, b_1 + b_2, c_1 + c_2

    j00 = b * b_2 - a_2 * c_
    j01 = b * c_2 - b_2 * c_
    j10 = b * a_2 - b_2 * a
    j11 = b * b_2 - c_2 * a

    return (2 / (b ** 2 - c_ * a)) * np.array([[j00, j01], [j10, j11]])


def t_wp_air(n_s, n_p):
    j00 = 2*n_s**2/(n_p+n_s**2)
    j01 = j10 = np.zeros(n_p.shape)
    j11 = 2*n_p/(1+n_p)

    return np.array([[j00, j01], [j10, j11]])


def t_air_wp(n_s, n_p):
    j00 = 2*n_p/(n_p+n_s**2)
    j01 = j10 = np.zeros(n_p.shape)
    j11 = 2/(1+n_p)

    return np.array([[j00, j01], [j10, j11]])


def single_wp_w_t_loss(theta, d, n_s, n_p):
    wp = jones_wp_no_t_loss(theta, d, n_s, n_p)
    t_a_wp = t_air_wp(n_s, n_p)
    t_wp_a = t_wp_air(n_s, n_p)

    return np.einsum('abf,bcf,cdf->adf', t_wp_a, wp, t_a_wp)


def calc_intensity(wp_, only_x=False):
    if only_x:
        final_j_x = np.einsum('ab,bce,c->ae', x_pol_j, wp_, x_linear_j)
        int_j_x = final_j_x[0] * np.conjugate(final_j_x[0]) + final_j_x[1] * np.conjugate(final_j_x[1])

        return 10*np.log10(int_j_x)

    else:
        final_j_x = np.einsum('ab,bce,c->ae', x_pol_j, wp_, x_linear_j)
        final_j_y = np.einsum('ab,bce,c->ae', y_pol_j, wp_, x_linear_j)

        int_j_x = final_j_x[0] * np.conjugate(final_j_x[0]) + final_j_x[1] * np.conjugate(final_j_x[1])
        int_j_y = final_j_y[0] * np.conjugate(final_j_y[0]) + final_j_y[1] * np.conjugate(final_j_y[1])

        return 10*np.log10(int_j_x), 10*np.log10(int_j_y)


def optimal_angle_dev(n_s, n_p):
    n_s, n_p = n_s * np.ones(f_pnts), n_p * np.ones(f_pnts)
    angles = np.deg2rad(np.linspace(35, 55, 200))
    minima_t_loss, minima_no_t_loss = [], []
    for angle in angles:
        wp_method_loss = jones_wp_t_loss(angle, d[0], n_s, n_p)
        int_j_x_loss = calc_intensity(wp_method_loss, only_x=True)
        minima_t_loss.append(int_j_x_loss.min())

        #wp_no_loss = jones_wp_no_t_loss(angle, d[0], n_s, n_p)
        #int_j_x_no_loss, int_j_y_no_loss = calc_intensity(wp_no_loss)
        #minima_no_t_loss.append(min(int_j_x_no_loss))

    #minima_t_loss, minima_no_t_loss = np.array(minima_t_loss), np.array(minima_no_t_loss)

    return 180*np.abs(angles[np.argmin(minima_t_loss)]-pi/4)/pi


def optimal_angle_dev2(n_s, n_p):
    t_x = (4 * n_p * n_s ** 2) / (n_s ** 2 + n_p) ** 2
    t_y = (4 * n_p ** 3) / (n_p + n_p ** 2) ** 2
    best_angle = np.arctan(np.sqrt(t_x / t_y))

    return np.abs(45 - np.rad2deg(best_angle))

def optimal_angle_dev3(d, k_s, k_p):
    alpha_s, alpha_p = -2 * pi * k_s * d / wls, -2 * pi * k_p * d / wls
    best_angle = np.arctan(np.sqrt(exp(alpha_s) / exp(alpha_p)))

    return np.abs(45 - np.rad2deg(best_angle))

def single_wp_intensity(theta, d, n_s, n_p, t_loss=True):
    phi_s, phi_p = 2 * d * n_s * pi / wls, 2 * d * n_p * pi / wls
    x, y = 1j * phi_s, 1j * phi_p
    if t_loss:
        t_x, t_y = (4 * n_p * n_s ** 2) / (n_s ** 2 + n_p) ** 2, (4 * n_p ** 3) / (n_p + n_p ** 2) ** 2
    else:
        t_x, t_y = np.ones_like(n_p), np.ones_like(n_p)

    e_0 = t_x * exp(x) * cos(theta) * cos(theta) + t_y * exp(y) * sin(theta) * sin(theta)

    return 10*np.log10(e_0*np.conjugate(e_0))


f_start = 1.0 * THz
f_end = 1.5 * THz

f_pnts = int((f_end - f_start) / (10 * GHz))
freqs = np.linspace(f_start, f_end, f_pnts)

angles = [45] #[95.68, 290.49, 134.65, 332.32, 348.36]
angles = np.deg2rad(angles)

d = [430] #[590., 600., 570., 400., 600.]
d = np.array(d) * um

wp_cnt = len(d)

l_mat1 = 50 * um  # *75*um  # 64.8*um # mat1: nicht luft
l_mat2 = 30 * um  # *61*um  # 45.6*um # mat2: luft

x_linear_j = np.array([1, 0])
x_linear_m = np.array([1, 1, 0, 0])

"""
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# eps_1,2 from material parameters

data_file_path = Path('/home/alex/Desktop/Projects/SimsV2_1/modules/material_manager/data_folders/FusedSilica/3Eck_D=1028.csv')
# data_file_path = Path(r'E:\CURPROJECT\SimsV2_1\modules\material_manager\data_folders\FusedSilica\3Eck_D=1028.csv')

df = pandas.read_csv(data_file_path)
eps_silica_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
eps_silica_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

constants = {}
freq_dict_key = [key for key in df.keys() if "freq" in key][0]

freqs = np.array(df[freq_dict_key])

data_slice = np.where((freqs > f_start) & (freqs < f_end))
data_slice = data_slice[0]
freqs = freqs[data_slice]

constants["eps_silica_r"] = np.array(df[eps_silica_r_key])[data_slice]
constants["eps_silica_i"] = np.array(df[eps_silica_i_key])[data_slice]

f_pnts = len(freqs)
freqs = freqs

eps_mat1 = (constants["eps_silica_r"] + constants["eps_silica_i"] * 1j)
eps_mat2 = (np.ones(eps_mat1.shape, dtype=eps_mat1.dtype))
"""
#"""
# const eps_1,2
n1 = 2  #1.89  # material
n2 = 1  # luft

# n1**2+0.053j
eps_mat1 = ((n1 ** 2 + 0.000j) * np.ones(freqs.shape, dtype=np.float))
eps_mat2 = ((n2 ** 2 + 0.000j) * np.ones(freqs.shape, dtype=np.float))
#"""

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

wls = (c / freqs)
n_s, n_p, k_s, k_p = calc_wp_deltas(l_mat1, l_mat2)
n_s_n, n_p_n = real_ri(l_mat1, l_mat2, eps_mat1[0].real, eps_mat2[0].real, freqs)

#"""
plt.plot(freqs, n_s, label="s-second order", color="blue")
plt.plot(freqs, n_p, label="p-second order", color="red")
plt.plot(freqs, n_s_n, 'b*', label="s-numerical")
plt.plot(freqs, n_p_n, 'r+', label="p-numerical")
plt.plot(freqs, n_p-n_s, label="bf 2nd order")
plt.plot(freqs, n_p_n-n_s_n, label="bf numerical")
plt.legend()
plt.show()
#"""

# simple plots
wp = jones_wp_no_t_loss(angles, d, n_s, n_p)
wp_n = jones_wp_no_t_loss(angles, d, n_s_n, n_p_n)

int_x, int_y = calc_intensity(wp)
int_x_n, int_y_n = calc_intensity(wp_n)

plt.plot(freqs, int_x, label='int after x_polarizer')
plt.plot(freqs, int_y, label='int after y_polarizer')
plt.plot(freqs, int_x_n, label='int after x_polarizer (numerical)')
plt.plot(freqs, int_y_n, label='int after y_polarizer (numerical)')
plt.legend()
plt.show()

"""
# save ri to file for import in cst
Hz_to_THZ = 10**-12
with open('anisotropic_material_data.txt', 'a') as file:
    for freq, n_x, n_y, n_z in zip(freqs, n_s, n_p, n_p):
        line = f'{freq*Hz_to_THZ}    {n_x**2}    {0}    {n_y**2}    {0}    {n_z**2}    {0}\n'
        file.write(line)
print(n_s, n_p)
"""

# int l2 fp result with and without t losses
"""
wps = []
for angle, width in zip(angles, d):
    wps.append(jones_wp_no_t_loss(angle, width, n_s, n_p))
wps = np.array(wps)

wp_freq = []
from functools import reduce
for i in range(len(freqs)):
    wp_freq.append(reduce(dot, wps[:, :, :, i]))
wp_freq = np.array(wp_freq)

einsum_str = 'abg,bcg,cdg,deg,efg->afg'
wp_stack = np.einsum(einsum_str, *wps)

final_j_x = np.einsum('ab,bce,c->ae', x_pol_j, wp_stack, x_linear_j)
final_j_y = np.einsum('ab,bce,c->ae', y_pol_j, wp_stack, x_linear_j)

int_j_x_ohne = final_j_x[0] * np.conjugate(final_j_x[0]) + final_j_x[1] * np.conjugate(final_j_x[1])
int_j_y_ohne = final_j_y[0] * np.conjugate(final_j_y[0]) + final_j_y[1] * np.conjugate(final_j_y[1])

int_j_x_ohne, int_j_y_ohne = 10 * np.log10(np.array(int_j_x_ohne)), 10 * np.log10(np.array(int_j_y_ohne))

int_j_x, int_j_y = [], []
for i in range(len(freqs)):
    wp0 = wps[0, :, :, i]
    wp1 = wps[1, :, :, i]
    wp2 = wps[2, :, :, i]
    wp3 = wps[3, :, :, i]
    wp4 = wps[4, :, :, i]

    t_aw = t_air_wp(n_s, n_p)[:, :, i]
    t10 = transmission_matrix(angles[0], angles[1], n_s, n_p)[:, :, i]
    t21 = transmission_matrix(angles[1], angles[2], n_s, n_p)[:, :, i]
    t32 = transmission_matrix(angles[2], angles[3], n_s, n_p)[:, :, i]
    t43 = transmission_matrix(angles[3], angles[4], n_s, n_p)[:, :, i]
    t_wa = t_wp_air(n_s, n_p)[:, :, i]
    #print(t10)
    #print(t21)
    #print(t32)
    #print(t43)

    e0 = dot(t_aw, jones_initial)
    ef = dot(t_wa, dot(wp4, dot(t43, dot(wp3, dot(t32, dot(wp2, dot(t21, dot(wp1, dot(t10, dot(wp0, e0))))))))))

    final_j_x = dot(x_pol_j, ef)
    final_j_y = dot(y_pol_j, ef)

    int_j_x.append(final_j_x[0] * np.conjugate(final_j_x[0]) + final_j_x[1] * np.conjugate(final_j_x[1]))
    int_j_y.append(final_j_y[0] * np.conjugate(final_j_y[0]) + final_j_y[1] * np.conjugate(final_j_y[1]))

int_j_x, int_j_y = 10 * np.log10(np.array(int_j_x)), 10 * np.log10(np.array(int_j_y))

plt.plot(freqs, int_j_x, label='int_j_x')
plt.plot(freqs, int_j_y, label='int_j_y')
plt.plot(freqs, int_j_x_ohne, label='int_j_x_ohne')
plt.plot(freqs, int_j_y_ohne, label='int_j_y_ohne')

plt.legend()
plt.show()
"""
"""
# n_s y-axis, n_p x-axis, angle dev. from 45deg optimum img.

len_x, len_y = 1000, 1000
cnt = 0
img = np.zeros((len_x, len_y))
n_s_lst, n_p_lst = np.linspace(1.1, 3.5, len_x), np.linspace(1.1, 3.5, len_y)

for x, n_s in enumerate(n_s_lst):
    for y, n_p in enumerate(n_p_lst):
        cnt += 1
        print(f'{cnt}/{len_x*len_y}')
        if x != y:
            img[x, y] = optimal_angle_dev2(n_s, n_p)
        else:
            img[x, y] = 0

np.save('image_t_loss.npy', img)
"""
"""
img = np.load('image_t_loss.npy')
# autumn, gnuplot, brg, gist_rainbow, rainbow, nipy_spectral, jet, RdBu, coolwarm
im = plt.imshow(img, extent=[1.1, 3.5, 1.1, 3.5], cmap=plt.cm.RdBu)
plt.xlabel('n_p')
plt.ylabel('n_s')
plt.colorbar(im)
plt.show()
"""

# x-axis: frequency, y-axis: intensity, for single bf., no loss and with t-loss
"""
n_s, n_p = 1.2*np.ones(f_pnts), 1.55*np.ones(f_pnts)
wp_t_loss = jones_wp_t_loss(np.pi/4, d[0], n_s, n_p)
int_j_x_loss, int_j_y_loss = calc_intensity(wp_t_loss)

wp_no_loss = jones_wp_no_t_loss(np.pi/4, d[0], n_s, n_p)
int_j_x_no_loss, int_j_y_no_loss = calc_intensity(wp_no_loss)

int_test = single_wp_intensity(np.pi/4, d[0], n_s, n_p)

#plt.plot(freqs, int_j_x_loss, label='int_x with loss')
#plt.plot(freqs, int_test, label='int_x with loss test')
plt.plot(freqs, int_j_x_no_loss, label='int_x no loss')
plt.legend()
plt.show()
"""

# optimal angle shift: angle x-axis, min. intensity y-axis
"""
n_s, n_p = 1.3*np.ones(f_pnts), 1.75*np.ones(f_pnts)
angles = np.deg2rad(np.linspace(30, 55, 1000))
minima_t_loss, minima_no_t_loss = [], []
for angle in angles:
    wp_t_loss = jones_wp_t_loss(angle, d[0], n_s, n_p)
    int_j_x_loss, int_j_y_loss = calc_intensity(wp_t_loss)
    minima_t_loss.append(min(int_j_x_loss))

    wp_no_loss = jones_wp_no_t_loss(angle, d[0], n_s, n_p)
    int_j_x_no_loss, int_j_y_no_loss = calc_intensity(wp_no_loss)
    minima_no_t_loss.append(min(int_j_x_no_loss))

print(180*np.abs(angles[np.argmin(minima_t_loss)]-angles[np.argmin(minima_no_t_loss)])/np.pi)

plt.plot(np.linspace(30, 55, 1000), minima_no_t_loss, label='int_x no loss')
plt.plot(np.linspace(30, 55, 1000), minima_t_loss, label='int_x with loss')
plt.legend()
plt.show()
"""

# angle dev from optimal dependent on bf: x-axis n_p + [0, 1], y-axis angle dev. from no t_loss
"""
n_s_values = n_p + np.linspace(0.3, 2, len(n_p))
angle_dev = []
cnt = 0
for n_s in n_s_values:
    cnt += 1
    angle_dev.append(np.mean(optimal_angle_dev2(n_s, n_p)))
    print(f'{cnt} / {len(n_p)}')

#np.save('angle_dev_bf_plot', np.array(angle_dev))

plt.plot(np.linspace(0, 1, len(n_p)), angle_dev)
plt.xlabel('bf')
plt.ylabel('angle_dev')
plt.show()
"""
