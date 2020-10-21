from numpy.random import uniform as rnd
from numpy import cos, sin, arctan, sqrt, sum, power, pi, outer, abs, exp
import string
import pandas
from pathlib import PureWindowsPath
from modules.identifiers.dict_keys import DictKeys
from modules.utils.constants import *
from functools import reduce


# setup erf, init value and optimization bounds
class ErfSetup(DictKeys):

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.const_wp_dimensions = settings[self.const_wp_dim_key]
        self.stripe_widths = np.array(settings[self.initial_stripe_widths_key]) * um  # ["material_width", "air_width"]
        self.const_angles = np.array(settings[self.const_angles_key])
        self.const_widths = np.array(settings[self.const_widths_key]) * um
        self.width_pattern = np.array(settings[self.width_pattern_key])
        self.selected_material_data_path = settings[self.selected_material_data_path_key]
        self.frequency_range = np.array([settings[self.min_freq_key], settings[self.max_freq_key]]) * THz
        self.frequency_resolution_multiplier = np.array(settings[self.frequency_resolution_multiplier_key])
        self.randomizer_seed = settings[self.randomizer_seed_key]
        self.log_of_res = settings[self.log_of_res_key]

        self.anis_p = settings[self.anisotropy_p_key]
        self.anis_s = settings[self.anisotropy_s_key]

        self.const_n_s = settings[self.const_refr_index_input_n_s_key]
        self.const_k_s = settings[self.const_refr_index_input_k_s_key]
        self.const_n_p = settings[self.const_refr_index_input_n_p_key]
        self.const_k_p = settings[self.const_refr_index_input_k_p_key]

        self.wp_type = settings[self.wp_type_key]

        self.set_seed()

        self.x_slices = self.settings[self.x_slicing_key]

        self.x0 = np.concatenate((np.array(self.settings[self.x0_angle_input_key]),
                                  np.array(self.settings[self.x0_widths_input_key]) * um,
                                  np.array(self.settings[self.x0_stripes_input_key]) * um))

        # bounds, passed on to optimization
        self.min_angle = settings[self.min_angle_key]
        self.max_angle = settings[self.max_angle_key]
        self.min_width = settings[self.min_width_key]
        self.max_width = settings[self.max_width_key]
        self.min_stripe_width = settings[self.min_stripe_width_key]
        self.max_stripe_width = settings[self.max_stripe_width_key]

        self.bounds = self.set_bounds()

        # ------------------------- constants ------------------------- #
        self.frequencies = None
        self.wls = None
        self.eps_mat1 = None
        self.eps_mat2 = None
        self.wp_cnt = None
        self.freq_cnt = None
        self.n0 = None
        self.einsum_str = None
        self.einsum_path = None

        self.set_constants()
        # ------------------------- constants ------------------------- #

        # always call last
        self.erf = self.setup_erf()

    def set_seed(self):
        np.random.seed(seed=self.randomizer_seed)

    def set_bounds(self):
        """
        sets bounds for optimization, needed for starting value x0

        :return: dict,
        "min_angles", "max_angles", in deg
        "min_widths", "max_widths", in um
        "min_stripes", "max_stripes" in um
        """

        angles2vary = self.x_slices[0][1] - self.x_slices[0][0]
        widths2vary = self.x_slices[1][1] - self.x_slices[1][0]
        stripes2vary = self.x_slices[2][1] - self.x_slices[2][0]

        min_angle, max_angle = self.min_angle, self.max_angle
        min_width, max_width = self.min_width, self.max_width
        min_stripe, max_stripe = self.min_stripe_width, self.max_stripe_width

        new_bounds = {"min_angles": [min_angle] * angles2vary,
                      "max_angles": [max_angle] * angles2vary,
                      "min_widths": [min_width * um] * widths2vary,
                      "max_widths": [max_width * um] * widths2vary,
                      "min_stripes": [min_stripe * um] * stripes2vary,
                      "max_stripes": [max_stripe * um] * stripes2vary}

        return new_bounds

    def read_data_file(self, file_path):
        data_filepath = Path(PureWindowsPath(file_path)).absolute()
        df = pandas.read_csv(data_filepath)

        freq_dict_key = [key for key in df.keys() if "freq" in key][0]
        eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
        eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

        frequencies = np.array(df[freq_dict_key])

        data_slice = np.where((frequencies > self.frequency_range[0]) &
                              (frequencies < self.frequency_range[1]))
        data_slice = data_slice[0][::int(1 // self.frequency_resolution_multiplier)]

        eps_mat_r = np.array(df[eps_mat_r_key])[data_slice]
        eps_mat_i = np.array(df[eps_mat_i_key])[data_slice]

        eps_mat1 = (eps_mat_r + eps_mat_i * 1j).reshape(len(data_slice), 1)

        return eps_mat1, frequencies[data_slice].reshape(len(data_slice), 1)

    def set_constants(self):
        """
        load eps data from self.selected_material_data_path
        """
        if self.settings[self.birefringence_type_dropdown_key] in 'Form':
            self.eps_mat1, self.frequencies = self.read_data_file(self.selected_material_data_path)
            # second material is air, ri=1
            self.eps_mat2 = np.ones(self.eps_mat1.shape, dtype=self.eps_mat1.dtype).reshape(len(self.frequencies), 1)

        else:
            fast_material_data_path = self.settings[self.selected_fast_material_data_path_key]
            slow_material_data_path = self.settings[self.selected_slow_material_data_path_key]

            self.eps_mat1, self.frequencies = self.read_data_file(fast_material_data_path)
            self.eps_mat2, _ = self.read_data_file(slow_material_data_path)

        self.freq_cnt = len(self.frequencies)
        self.wls = (c / self.frequencies).reshape(self.freq_cnt, 1)

        self.wp_cnt = len(self.const_angles)

        # n
        self.n0 = self.form_birefringence(self.stripe_widths)

        # setup einsum_str
        s0 = string.ascii_lowercase + string.ascii_uppercase
        self.einsum_str = ''
        for i in range(self.wp_cnt):
            self.einsum_str += s0[self.wp_cnt + 2] + s0[i] + s0[i + 1] + ','

        # remove last comma
        self.einsum_str = self.einsum_str[:-1]
        self.einsum_str += '->' + s0[self.wp_cnt + 2] + s0[0] + s0[self.wp_cnt]

        # einsum path (not sure if necessary)
        test_array = np.zeros((self.wp_cnt, self.freq_cnt, 4, 4))
        self.einsum_path = np.einsum_path(self.einsum_str, *test_array, optimize='greedy')


    def form_birefringence(self, stripes):
        """
        :return: array with length of frequency, frequency resolved delta n, delta kappa
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

        return np.array([n_s, n_p, self.anis_s * k_s, self.anis_p * k_p])

    def j_matrix_input(self, angles, d, n_s, n_p, k_s, k_p):
        wls = self.wls
        phi_s, phi_p = (2 * n_s * pi / wls) * d.T, (2 * n_p * pi / wls) * d.T

        # anisotropic part only
        if self.settings[self.weak_absorption_checkbox_key]:
            alpha_s, alpha_p = np.zeros_like(wls), -(2 * pi * (k_p - k_s) / wls) * d.T
        # original absorption term
        else:
            alpha_s, alpha_p = -(2 * pi * k_s / wls) * d.T, -(2 * pi * k_p / wls) * d.T

        theta = np.tile(angles, (self.freq_cnt, 1))

        return theta, 1j * phi_s + alpha_s, 1j * phi_p + alpha_p

    def j_absorption_factor(self, d, k_s):
        return np.prod(exp(-(4 * pi * k_s / self.wls) * d.T), axis=1)

    def build_j_matrix_stack(self, theta, x, y):
        J = np.zeros((self.freq_cnt, self.wp_cnt, 2, 2), dtype=np.complex)
        J[:, :, 0, 0] = exp(y) * sin(theta) ** 2 + exp(x) * cos(theta) ** 2
        J[:, :, 0, 1] = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
        J[:, :, 1, 0] = J[:, :, 0, 1]
        J[:, :, 1, 1] = exp(x) * sin(theta) ** 2 + exp(y) * cos(theta) ** 2

        np.einsum(self.einsum_str, *J.transpose((1, 0, 2, 3)), out=J[:, 0], optimize=self.einsum_path[0])

        return J[:, 0]

    def m_matrix_input(self, angles, d, n_s, n_p, k_s, k_p):
        wls = self.wls
        bf = n_s - n_p

        # tan(gamma) = p_p/p_s (p_y/p_x) (p_x = e^(-k_s))
        gamma = arctan(exp((2 * pi * (k_s - k_p) / wls) * d.T))

        p_squared = exp((-4 * pi * k_s / wls) * d.T) + exp((-4 * pi * k_p / wls) * d.T)

        ret = 2 * np.pi * bf * d.T / wls

        theta = np.tile(angles, (self.freq_cnt, 1))

        return theta, gamma, ret, p_squared

    def build_m_matrix_stack(self, theta, gamma, ret, p_squared):
        M = np.zeros((self.freq_cnt, self.wp_cnt, 4, 4), dtype=np.float)

        M[:, :, 0, 0] = 1
        M[:, :, 0, 1] = cos(2 * gamma) * cos(2 * theta)
        M[:, :, 0, 2] = -sin(2 * theta) * cos(2 * gamma)
        M[:, :, 1, 1] = cos(2 * theta) ** 2 + sin(2 * theta) ** 2 * cos(ret) * sin(2 * gamma)
        M[:, :, 1, 2] = cos(2 * theta) * sin(2 * theta) * (sin(2 * gamma) * cos(ret) - 1)
        M[:, :, 1, 3] = sin(2 * theta) * sin(2 * gamma) * sin(ret)
        M[:, :, 2, 2] = sin(2 * theta) ** 2 + cos(2 * theta) ** 2 * sin(2 * gamma) * cos(ret)
        M[:, :, 2, 3] = cos(2 * theta) * sin(2 * gamma) * sin(ret)
        M[:, :, 3, 3] = sin(2 * gamma) * cos(ret)

        M[:, :] += M[:, :].transpose(0, 1, 3, 2)
        M[:, :, np.arange(4), np.arange(4)] *= 0.5

        M[:, :, 3, 1] *= -1
        M[:, :, 3, 2] *= -1

        np.einsum(self.einsum_str, *M.transpose((1, 0, 2, 3)), out=M[:, 0], optimize=self.einsum_path[0])

        M[:, 0] = (M[:, 0].transpose(1, 2, 0) * ((0.5 ** self.wp_cnt) * np.prod(p_squared, axis=1))).transpose(2, 0, 1)

        return M[:, 0]

    def setup_input_vectors(self, x):
        angle_x_indices = self.x_slices[0]
        width_x_indices = self.x_slices[1]
        stripe_x_indices = self.x_slices[2]

        const_angles = self.const_angles
        angles = const_angles.astype(np.float)

        const_widths = self.const_widths
        width_pattern = self.width_pattern

        relevant_identifiers = width_pattern[np.where(const_widths == 0)]

        unique_relevant_identifiers = np.unique(relevant_identifiers)

        # setup width array from locked_widths, linked_widths and x
        d = const_widths.astype(np.float)

        open_spots = d[d == 0]

        for i in range(len(unique_relevant_identifiers)):
            open_spots[relevant_identifiers == unique_relevant_identifiers[i]] = \
                x[width_x_indices[0]:width_x_indices[1]][i]

        d[d == 0] = open_spots

        d = d.reshape(self.wp_cnt, 1)

        angles[const_angles == 0] = x[angle_x_indices[0]:angle_x_indices[1]]

        stripes = x[stripe_x_indices[0]:stripe_x_indices[1]]

        return angles, d, stripes

    def natural_birefringence(self):
        eps_s = self.eps_mat1
        eps_p = self.eps_mat2

        n_p, n_s = (sqrt(abs(eps_p) + eps_p.real) / sqrt2, sqrt(abs(eps_s) + eps_s.real) / sqrt2)
        k_p, k_s = (sqrt(abs(eps_p) - eps_p.real) / sqrt2, sqrt(abs(eps_s) - eps_s.real) / sqrt2)

        return np.array([n_s, n_p, self.anis_s * k_s, self.anis_p * k_p])

    def setup_ri(self, stripes):
        if self.settings[self.birefringence_type_dropdown_key] in 'Form':
            if self.const_wp_dimensions:
                # if stripe width doesn't change -> birefringence doesn't change
                n = self.n0
            else:
                n = self.form_birefringence(stripes)
        else:
            n = self.natural_birefringence()

        # let stripes have no effect if we overwrite ri anyways
        if self.settings[self.set_ri_real_part_checkbox_key] or self.settings[self.set_ri_img_part_checkbox_key]:
            self.const_wp_dimensions = True

        if self.settings[self.set_ri_real_part_checkbox_key]:
            n[0], n[1] = self.const_n_s*np.ones_like(n[0]), self.const_n_p*np.ones_like(n[1])
        if self.settings[self.set_ri_img_part_checkbox_key]:
            n[2], n[3] = self.const_k_s*np.ones_like(n[2]), self.const_k_p*np.ones_like(n[3])

        n[2], n[3] = self.anis_s * n[2], self.anis_p * n[3]

        return n

    def setup_erf(self):
        """
        for setting up the error function

        :return: error function to be minimized
        """

        def j_stack_err(x):
            """
            calculates error given angles, widths and stripe widths using jones calc

            :param x: array of angles, widths, stripe widths
            :return: value of error function
            """
            angles, d, stripes = self.setup_input_vectors(x)

            n_s, n_p, k_s, k_p = self.setup_ri(stripes)

            theta, x, y = self.j_matrix_input(angles, d, n_s, n_p, k_s, k_p)

            J = self.build_j_matrix_stack(theta, x, y)

            self.int_x, self.int_y = 10*np.log10(J[:, 0, 0] * np.conjugate(J[:, 0, 0])), \
                                     10*np.log10(J[:, 1, 0] * np.conjugate(J[:, 1, 0]))

            if self.wp_type == 'λ/2':
                res = sum((1 - J[:, 1, 0] * np.conjugate(J[:, 1, 0]) + J[:, 0, 0] * np.conjugate(J[:, 0, 0])) ** 2)
                # Jan loss function. No I_x
                #res = sum((1 - J[:, 1, 0] * np.conjugate(J[:, 1, 0])) ** 2)
            else:
                # OG intensity loss
                # res = sum((J[:, 1, 0] * np.conjugate(J[:, 1, 0]) - J[:, 0, 0] * np.conjugate(J[:, 0, 0])) ** 2)
                q = J[:, 0, 0] / J[:, 1, 0]
                res = sum(q.real**2 + (q.imag-1)**2)

            # for testing
            abs_factor = self.j_absorption_factor(d, k_s)
            self.intensity_x = 10*np.log10(abs_factor * J[:, 0, 0] * np.conjugate(J[:, 0, 0]))
            self.intensity_y = 10*np.log10(abs_factor * J[:, 1, 0] * np.conjugate(J[:, 1, 0]))

            if self.log_of_res:
                return np.log10(res / self.freq_cnt)

            return res.real / self.freq_cnt

        def m_stack_err(x):
            """
            calculates error given angles, widths and stripe widths using mueller-stokes calc

            :param x: array of angles, widths, stripe widths
            :return: value of error function
            """
            angles, d, stripes = self.setup_input_vectors(x)

            n_s, n_p, k_s, k_p = self.setup_ri(stripes)

            theta, gamma, ret, p_squared = self.m_matrix_input(angles, d, n_s, n_p, k_p, k_s)

            M = self.build_m_matrix_stack(theta, gamma, ret, p_squared)

            if self.wp_type == 'λ/2':
                res = sum((M[:, 0, 0] + M[:, 0, 1] - 1) ** 2) + sum((M[:, 1, 0] + M[:, 1, 1] + 1) ** 2)
            else:
                res = sum((M[:, 0, 0] - M[:, 3, 1]) ** 2) + sum((M[:, 3, 3]) ** 2)  # original lambda / 4

            if self.log_of_res:
                return np.log10(res / self.freq_cnt)

            return res / self.freq_cnt

        if self.settings[self.calculation_method_key] in 'Stokes':
            return m_stack_err
        else:
            return j_stack_err

if __name__ == '__main__':
    from modules.settings.settings import Settings
    import matplotlib.pyplot as plt

    path = '/home/alex/Desktop/Projects/SimsV2_1/modules/results/saved_results/fp_results_quartz/fp_quartz_11-44-59_Thread-2/settings.json'
    settings_dict = Settings().load_settings(path)
    erf_setup = ErfSetup(settings_dict)
    erf = erf_setup.erf

    angles = np.deg2rad([95.68, 290.49, 134.65, 332.32, 348.36])
    d = np.array([590.0, 600.0, 570.0, 400.0, 600.0])*um

    x0 = np.concatenate((angles, d))
    erf(x0)

    int_x, int_y = erf_setup.intensity_x, erf_setup.intensity_y
    freqs = erf_setup.frequencies

    plt.plot(freqs, int_x, label='after x-pol')
    plt.plot(freqs, int_y, label='after y-pol')
    plt.show()
