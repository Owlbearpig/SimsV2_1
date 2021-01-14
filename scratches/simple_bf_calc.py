import numpy as np
from scipy.constants import c
from numpy import power, outer, sqrt, abs, pi
sqrt2 = sqrt(2)
GHz = 10**9

class BF:
    def __init__(self):

        self.wp_cnt = 5

        self.freqs = np.linspace(50, 450, 50)*GHz
        self.wls = c/self.freqs
        self.eps_mat1 = np.ones_like(self.freqs)*4
        self.eps_mat2 = np.ones_like(self.freqs)*1 # air


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

