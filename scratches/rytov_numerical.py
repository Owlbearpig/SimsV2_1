import numpy as np
from numpy import pi as pi
import matplotlib.pyplot as plt
from sympy.solvers import nsolve
from sympy import Symbol, tan, Rational
eps = Symbol("eps")
from scipy.constants import c
um = 10**-6
THz = 10**12
GHz = 10**9


# tan x ~ x + 1/3 x ** 3
def rytov_two_layer_second_order(width_layer_1, width_layer_2, frequency, ref_index_1, ref_index_2):
    a = width_layer_1
    b = width_layer_2
    f = frequency
    n_1 = ref_index_1
    n_2 = ref_index_2

    k = 2*np.pi*f/c

    d = a + b
    f_1 = a/d
    f_2 = b/d

    # Parallel (first return value) (p-pol.)
    n_te_0 = (n_1**2 * f_1 + n_2**2 * f_2)**(1/2)

    n_te = (n_te_0**2 + (1/3) * (d * k/2 * f_1 * f_2 * (n_1**2 - n_2**2))**2)**(1/2)

    # Perpendicular (second return value) (s-pol.)
    n_tm_0 = ((n_1**2 * n_2**2)/(f_1 * n_2**2 + f_2 * n_1**2))**(1/2)

    n_tm = (n_tm_0**2 + (1/3)*((d * k/2 * f_1 * f_2 * ((1/n_1)**2 - (1/n_2)**2) * n_te_0 * (n_tm_0**3))**2))**(1/2)

    return [n_te, n_tm]


# x0 : start value (te, tm). Returns RI in p + s
def rytov_two_layer_numerical(width_layer_1, width_layer_2, frequency, ref_index_1, ref_index_2, x0):
    a = width_layer_1
    b = width_layer_2
    f = frequency
    eps_1 = ref_index_1**2
    eps_2 = ref_index_2**2

    k = 2 * np.pi * f / (3 * 10 ** 8)

    r = Rational(1, 2)

    # parallel (first return value) (p-pol.)
    eps_te = nsolve(
        k * (eps_2 - eps) ** r * tan(k * (eps_2 - eps) ** r * b * r) + k * (
                    eps_1 - eps) ** r * tan(k * (eps_1 - eps) ** r * a * r)
        , eps, x0[0])

    # perpendicular (second return value) (s-pol.)
    eps_tm = nsolve(
        (1/eps_2) * k * (eps_2 - eps) ** r * tan(k * (eps_2 - eps) ** r * b * r) + (1/eps_1) * k * (
                    eps_1 - eps) ** r * tan(k * (eps_1 - eps) ** r * a * r)
        , eps, x0[1])

    return [eps_te**0.5, eps_tm**0.5]


if __name__ == '__main__':

    # Material 1: PP, Material 2: PP compound
    THz_FB_values = {
        "width_layer_1": 250 * um,  # m
        "width_layer_2": 150 * um,  # m
        #"frequency": 300 * GHz,  # Hz
        "ref_index_1": complex(1.5, 0),  # DQ
        "ref_index_2": complex(2.4, 0)  # DQ
    }

    # Material 1: silica glass, Material 2: air
    HQ_THz_GWP_values = {
        "width_layer_1": 30 * um,  # m,
        "width_layer_2": 70 * um,  # m
        #"frequency": 400 * GHz,  # Hz
        "ref_index_1": complex(4**(1/2), 0),  # DQ
        "ref_index_2": complex(1, 0)  # DQ
    }

    mat_params = HQ_THz_GWP_values

    nopp = 50  # number of plot points; resolution
    start_freq, end_freq = 400, 2000
    frequency_list = [(start_freq + end_freq * (i/nopp)) * GHz for i in range(nopp)]
    a = mat_params["width_layer_1"]
    b = mat_params["width_layer_2"]
    eps_1 = mat_params["ref_index_1"]**2
    eps_2 = mat_params["ref_index_2"]**2

    n_tm_list_n, n_te_list_n, n_tm_list_a, n_te_list_a = [], [], [], []
    for freq in frequency_list:
        mat_params["frequency"] = freq
        second_order_sol = rytov_two_layer_second_order(**mat_params)
        for i in range(len(second_order_sol)):
            if type(second_order_sol[i]) == complex:
                second_order_sol[i] = second_order_sol[i].real  # only using real part of ref. index

        n_te_list_a.append(second_order_sol[0])
        n_tm_list_a.append(second_order_sol[1])
        if freq == frequency_list[0]:
            prev_sol = second_order_sol
        else:
            prev_sol = (n_te_list_n[-1], n_tm_list_n[-1])

        print("\n" + str(freq) + "\n")

        # try and find solution closest to previous one
        solutions_te, solutions_tm = [], []
        for delta in np.linspace(-1.5, 1.5, 10):
            start_val = (prev_sol[0] + delta, prev_sol[1] + delta)
            try:
                num_res = rytov_two_layer_numerical(x0=start_val, **mat_params)
                print(num_res[0], num_res[1])
                if all([i ** 2 > 0 for i in num_res]):
                    solutions_te.append(num_res[0])
                    solutions_tm.append(num_res[1])

            except Exception as e:  # not all start values lead to convergence (secant method)
                print("\n" + str(start_val))
                print(str(e) + "\n")

        te_sol_index = int(np.argmin([((i - prev_sol[0]) ** 2) ** 0.5 for i in solutions_te]))
        tm_sol_index = int(np.argmin([((i - prev_sol[0]) ** 2) ** 0.5 for i in solutions_tm]))

        n_te_list_n.append(solutions_te[te_sol_index])
        n_tm_list_n.append(solutions_tm[tm_sol_index])

    plt.plot(frequency_list, n_te_list_a, label="p-second order", color="red")
    plt.plot(frequency_list, n_tm_list_a, label="s-second order", color="blue")
    plt.plot(frequency_list, n_te_list_n, 'r*', label="p-numerical")
    plt.plot(frequency_list, n_tm_list_n, 'b+', label="s-numerical")
    plt.xlabel("f")
    plt.ylabel("n")
    plt.tick_params(direction='in')
    plt.legend()
    plt.show()

