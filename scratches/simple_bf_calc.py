import numpy as np
from scipy.constants import c
from numpy import power, outer, sqrt, abs, pi, sin, exp, cos
import matplotlib.pyplot as plt

sqrt2 = sqrt(2)
GHz = 10**9
THz = 10**12
um = 10**-6

# TODO Look at jones_interface_transmission.py. Same purpose but that works. This doesn't... freq array shape is wrong.

class QuickCalc:
    def __init__(self, freq_range=(), stripes=()):
        self.wp_cnt = 1

        if freq_range:
            self.freqs = np.linspace(freq_range[0], freq_range[1], 50) * THz
        else:
            self.freqs = np.linspace(50, 450, 50)*GHz
        self.wls = (c / self.freqs).reshape(len(self.freqs), 1)

        self.eps_mat1 = (np.ones_like(self.freqs)*4).reshape(len(self.freqs), 1)
        self.eps_mat2 = (np.ones_like(self.freqs)*1).reshape(len(self.freqs), 1)  # air

        if stripes:
            self.bf = self.form_birefringence(stripes)
        else:
            self.bf = None

        self.x_linear_j = np.array([1, 0])
        self.x_pol_j = np.array([[1, 0], [0, 0]])
        self.y_pol_j = np.array([[0, 0], [0, 1]])

    def form_birefringence(self, stripes):
        """
        :return: array with length of frequency, frequency resolved [ns, np, ks, kp]
        """
        if len(stripes) == 2:
            l_mat1, l_mat2 = stripes
        else:
            l_mat1, l_mat2 = stripes[0:self.wp_cnt], stripes[self.wp_cnt:]

        wls = self.wls
        eps_mat1, eps_mat2 = self.eps_mat1, self.eps_mat2

        a = (1 / 3) * power(outer(1 / wls, (l_mat1 * l_mat2 * pi) / (l_mat1 + l_mat2)), 2)

        # first order s and p
        wp_eps_s_1 = outer((eps_mat2 * eps_mat1), (l_mat2 + l_mat1)) / (
                outer(eps_mat2, l_mat1) + outer(eps_mat1, l_mat2))

        wp_eps_p_1 = outer(eps_mat1, l_mat1 / (l_mat2 + l_mat1)) + outer(eps_mat2, l_mat2 / (l_mat2 + l_mat1))

        # 2nd order
        wp_eps_s_2 = wp_eps_s_1 + (a * power(wp_eps_s_1, 3) * wp_eps_p_1 * power((1 / eps_mat1 - 1 / eps_mat2), 2))
        wp_eps_p_2 = wp_eps_p_1 + (a * power((eps_mat1 - eps_mat2), 2))

        # returns
        n_p, n_s = (
            sqrt(abs(wp_eps_p_2) + wp_eps_p_2.real) / sqrt2,
            sqrt(abs(wp_eps_s_2) + wp_eps_s_2.real) / sqrt2
        )
        k_p, k_s = (
            sqrt(abs(wp_eps_p_2) - wp_eps_p_2.real) / sqrt2,
            sqrt(abs(wp_eps_s_2) - wp_eps_s_2.real) / sqrt2
        )

        return np.array([n_s, n_p, k_s, k_p])

    def jones_wp_no_t_loss(self, theta, d, n_s, n_p):
        phi_s, phi_p = 2 * d * n_s * pi / self.wls, 2 * d * n_p * pi / self.wls
        alpha_s, alpha_p = -2 * pi * k_s * d / self.wls, -2 * pi * k_p * d / self.wls
        x, y = 1j * phi_s + alpha_s, 1j * phi_p + alpha_p

        j00 = exp(x) * cos(theta) ** 2 + exp(y) * sin(theta) ** 2
        j01 = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
        j10 = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
        j11 = exp(x) * sin(theta) ** 2 + exp(y) * cos(theta) ** 2
        print(j00.shape)
        return np.array([[j00, j01],
                         [j10, j11]])

    def intensities(self, wp_matrix):
        print(self.x_pol_j.shape, wp_matrix.shape, self.x_linear_j.shape)
        final_j_x = np.einsum('ab,bce,c->ae', self.x_pol_j, wp_matrix, self.x_linear_j)
        final_j_y = np.einsum('ab,bce,c->ae', self.y_pol_j, wp_matrix, self.x_linear_j)

        int_j_x = final_j_x[0] * np.conjugate(final_j_x[0]) + final_j_x[1] * np.conjugate(final_j_x[1])
        int_j_y = final_j_y[0] * np.conjugate(final_j_y[0]) + final_j_y[1] * np.conjugate(final_j_y[1])

        return 10 * np.log10(int_j_x), 10 * np.log10(int_j_y)


qc = QuickCalc(freq_range=(1, 1.5), stripes=(100 * um, 100 * um))

freqs = qc.freqs

n_s, n_p, k_s, k_p = qc.bf
wp = qc.jones_wp_no_t_loss(theta=pi/4, d=500*um, n_p=n_p, n_s=n_s)
int_x, int_y = qc.intensities(wp_matrix=wp)

plt.plot(freqs, int_x, label='int_x')
plt.plot(freqs, int_y, label='int_y')
plt.legend()
plt.show()
