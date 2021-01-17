import numpy as np
from numpy import sqrt, power, sin, cos, exp, pi
import matplotlib.pyplot as plt
from sympy.solvers import nsolve
from sympy import Symbol, tan, Rational
eps = Symbol("eps")
from scipy.constants import c

# x0 : start value (te, tm). Returns RI in p + s
def rytov_two_layer_numerical(l1, l2, eps1, eps2, frequency, x0):
    k = 2 * pi * frequency / c

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

    return [eps_s**0.5, eps_p**0.5]


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

def real_ri(l_mat1, l_mat2, eps1, eps2, frequencies):
    n_s, n_p, k_s, k_p = calc_wp_deltas(l_mat1, l_mat2)
    n_s, n_p = n_s.real, n_p.real

    prev_sol = [n_s[0], n_p[0]]

    n_p_n, n_s_n = [], []
    for i, freq in enumerate(frequencies):
        print(i, len(frequencies))

        # try and find solution closest to previous one
        sols_s, sols_p = [], []
        for delta in np.linspace(-1.5, 1.5, 20):
            start_val = (prev_sol[0] + delta, prev_sol[1] + delta)
            try:
                num_res = rytov_two_layer_numerical(l_mat1, l_mat2, eps1, eps2, freq, x0=start_val)
                print(num_res[0], num_res[1])
                if all([i ** 2 > 0 for i in num_res]):
                    sols_s.append(num_res[0])
                    sols_p.append(num_res[1])

            except Exception as e:  # not all start values lead to convergence (secant method)
                print("\n" + str(start_val))
                print(str(e) + "\n")

        s_sol_index = int(np.argmin([((i - prev_sol[0]) ** 2) ** 0.5 for i in sols_s]))
        p_sol_index = int(np.argmin([((i - prev_sol[1]) ** 2) ** 0.5 for i in sols_p]))
        
        n_s_n.append(sols_s[s_sol_index])
        n_p_n.append(sols_p[p_sol_index])

        prev_sol = (n_p_n[-1], n_s_n[-1])

    return n_s_n, n_p_n

if __name__ == '__main__':
    """
    n_s, n_p, k_s, k_p = calc_wp_deltas(l_mat1, l_mat2)
    n_s, n_p = n_s.real, n_p.real

    prev_sol = [n_s[0], n_p[0]]

    n_p_n, n_s_n = [], []
    for i, freq in enumerate(freqs):
        print(i, len(freqs))

        # try and find solution closest to previous one
        sols_s, sols_p = [], []
        for delta in np.linspace(-1.5, 1.5, 20):


            start_val = (prev_sol[0] + delta, prev_sol[1] + delta)
            try:
                num_res = rytov_two_layer_numerical(l_mat1, l_mat2, eps_mat1[0].real, eps_mat2[0].real,
                                                    freq, x0=start_val)
                print(num_res[0], num_res[1])
                if all([i ** 2 > 0 for i in num_res]):
                    sols_s.append(num_res[0])
                    sols_p.append(num_res[1])

            except Exception as e:  # not all start values lead to convergence (secant method)
                print("\n" + str(start_val))
                print(str(e) + "\n")

        s_sol_index = int(np.argmin([((i - prev_sol[0]) ** 2) ** 0.5 for i in sols_s]))
        p_sol_index = int(np.argmin([((i - prev_sol[1]) ** 2) ** 0.5 for i in sols_p]))

        n_s_n.append(sols_s[s_sol_index])
        n_p_n.append(sols_p[p_sol_index])

        prev_sol = (n_p_n[-1], n_s_n[-1])
    """
    sqrt2 = sqrt(2)

    um = 10 ** -6
    THz = 10 ** 12
    GHz = 10 ** 9

    f_start = 1.0 * THz
    f_end = 1.5 * THz

    f_pnts = int((f_end - f_start) / (1 * GHz))
    freqs = np.linspace(f_start, f_end, f_pnts)
    wls = c / freqs

    l_mat1 = 80 * um  # mat1: nicht luft
    l_mat2 = 30 * um  # mat2: luft

    n1 = 2  # 1.89  # material
    n2 = 1  # luft

    eps_mat1 = ((n1 ** 2 + 0.000j) * np.ones(freqs.shape, dtype=np.float))
    eps_mat2 = ((n2 ** 2 + 0.000j) * np.ones(freqs.shape, dtype=np.float))

    n_s, n_p, k_s, k_p = calc_wp_deltas(l_mat1, l_mat2)
    n_s_n, n_p_n = real_ri(l_mat1, l_mat2, eps_mat1[0].real, eps_mat2[0].real, freqs)

    plt.plot(freqs, n_s, label="s-second order", color="blue")
    plt.plot(freqs, n_p, label="p-second order", color="red")
    plt.plot(freqs, n_s_n, 'b*', label="s-numerical")
    plt.plot(freqs, n_p_n, 'r+', label="p-numerical")
    plt.xlabel("f")
    plt.ylabel("n")
    plt.tick_params(direction='in')
    plt.legend()
    plt.show()