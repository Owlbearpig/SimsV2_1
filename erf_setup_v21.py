from numpy.random import uniform as rnd
from numpy import cos, sin, sqrt, sum, power, pi, outer, abs, exp, conj
import string
import pandas
from pathlib import PureWindowsPath
from modules.identifiers.dict_keys import DictKeys
from modules.material_manager.materials import ProjectMaterials
from modules.utils.constants import *
from modules.utils.calculations import calc_final_jones_intensities, retardance, eig
from py_pol.jones_matrix import Jones_matrix

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
        self.material_name = settings[self.material_drop_down_key]
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

    def load_material_data(self, material_name):
        material = self.all_materials.get_material(material_name)

        data_filepath = Path(PureWindowsPath(material.path))
        df = pandas.read_csv(project_dir / data_filepath)

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
        self.all_materials = ProjectMaterials()

        if self.settings[self.birefringence_type_dropdown_key] in 'Form':
            self.eps_mat1, self.frequencies = self.load_material_data(self.material_name)
            # second material is air, ri=1
            self.eps_mat2 = np.ones(self.eps_mat1.shape, dtype=self.eps_mat1.dtype).reshape(len(self.frequencies), 1)

        else:
            fast_material_name = self.settings[self.fast_material_dropdown_key]
            slow_material_name = self.settings[self.slow_material_dropdown_key]

            self.eps_mat1, self.frequencies = self.load_material_data(fast_material_name)
            self.eps_mat2, _ = self.load_material_data(slow_material_name)

        self.freq_cnt = len(self.frequencies)
        self.wls = (c / self.frequencies).reshape(self.freq_cnt, 1)

        self.wp_cnt = self.settings[self.wp_cnt_key]

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

    def build_j_matrix_stack(self, theta, x, y):

        j = np.zeros((self.freq_cnt, self.wp_cnt, 2, 2), dtype=np.complex)
        j[:, :, 0, 0] = exp(y) * sin(theta) ** 2 + exp(x) * cos(theta) ** 2
        j[:, :, 0, 1] = 0.5 * sin(2 * theta) * (exp(y) - exp(x))
        j[:, :, 1, 0] = j[:, :, 0, 1]
        j[:, :, 1, 1] = exp(x) * sin(theta) ** 2 + exp(y) * cos(theta) ** 2

        np.einsum(self.einsum_str, *j.transpose((1, 0, 2, 3)), out=j[:, 0], optimize=self.einsum_path[0])

        return j[:, 0]

    def m_matrix_input(self, angles, d, n_s, n_p, k_s, k_p):
        wls = self.wls
        bf = n_s - n_p

        p_s, p_p = exp(-(2 * pi * k_s / wls) * d.T), exp(-(2 * pi * k_p / wls) * d.T)

        ret = 2 * np.pi * bf * d.T / wls

        theta = np.tile(angles, (self.freq_cnt, 1))

        return theta, ret, p_s, p_p

    def build_m_matrix_stack(self, theta, ret, p_s, p_p):
        M = np.zeros((self.freq_cnt, self.wp_cnt, 4, 4), dtype=np.float)
        if self.settings[self.weak_absorption_checkbox_key]:
            a, b, c1 = 0.5*(1 + (p_p/p_s)**2), 0.5*(1 - (p_p/p_s)**2), p_p/p_s
        else:
            a, b, c1 = 0.5 * (p_s ** 2 + p_p ** 2), 0.5 * (p_s ** 2 - p_p ** 2), p_s * p_p

        M[:, :, 0, 0] = a
        M[:, :, 0, 1] = b * cos(2 * theta)
        M[:, :, 0, 2] = -sin(2 * theta) * b
        M[:, :, 1, 1] = a*(cos(2 * theta) ** 2) + (sin(2 * theta) ** 2) * cos(ret) * c1
        M[:, :, 1, 2] = cos(2 * theta) * sin(2 * theta) * (c1 * cos(ret) - a)
        M[:, :, 1, 3] = sin(2 * theta) * c1 * sin(ret)
        M[:, :, 2, 2] = a*(sin(2 * theta) ** 2) + (cos(2 * theta) ** 2) * c1 * cos(ret)
        M[:, :, 2, 3] = cos(2 * theta) * c1 * sin(ret)
        M[:, :, 3, 3] = c1 * cos(ret)

        M[:, :] += M[:, :].transpose(0, 1, 3, 2)
        M[:, :, np.arange(4), np.arange(4)] *= 0.5

        np.negative(M[:, :, 3, 1], out=M[:, :, 3, 1])
        np.negative(M[:, :, 3, 2], out=M[:, :, 3, 2])

        np.einsum(self.einsum_str, *M.transpose((1, 0, 2, 3)), out=M[:, 0], optimize=self.einsum_path[0])

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
                #n = self.n0
                n = self.form_birefringence(self.stripe_widths)
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

        const_bf_offset = self.settings[self.add_const_birefringence_input_key]

        n[1] += const_bf_offset # add bf offset to TE mode

        return n

    def absorption_factor(self, d, k_s):
        return np.prod(exp((-4 * pi * k_s / self.wls) * d.T), axis=1)

    def setup_erf(self):
        """
        for setting up the error function

        :return: error function to be minimized
        """
        # split frequency range in two, lower half for l4, upper half for l2. 2 sets of angles but equal widths
        def j_stack_err_mixed_wp_type(x):
            angles, d, stripes = self.setup_input_vectors(x)

            n_s, n_p, k_s, k_p = self.setup_ri(stripes)

            theta, x, y = self.j_matrix_input(angles, d, n_s, n_p, k_s, k_p)

            m = self.freq_cnt
            # p_shift ∝ freq. p_shift_l2 = 2 * p_shift_l4
            j_l2 = self.build_j_matrix_stack(theta[m // 2:, 0:self.wp_cnt], x[m // 2:], y[m // 2:])
            j_l4 = self.build_j_matrix_stack(theta[:m // 2, self.wp_cnt:], x[:m // 2], y[:m // 2])

            res_l2 = sum((1 - j_l2[:, 1, 0] * conj(j_l2[:, 1, 0]) + j_l2[:, 0, 0] * conj(j_l2[:, 0, 0])) ** 2)

            q = j_l4[:, 0, 0] / j_l4[:, 1, 0]
            res_l4 = sum(q.real ** 2 + (q.imag - 1) ** 2)

            return (res_l4.real + res_l2.real) / self.freq_cnt
        
        def j_stack_err(x):
            """
            calculates error given angles, widths and stripe widths using jones calc

            :param x: array of angles, widths, stripe widths
            :return: value of error function
            """

            angles, d, stripes = self.setup_input_vectors(x)

            n_s, n_p, k_s, k_p = self.setup_ri(stripes)

            theta, x, y = self.j_matrix_input(angles, d, n_s, n_p, k_s, k_p)

            j = self.build_j_matrix_stack(theta, x, y)

            # adds the full absorption in case of enabled weak absorption for testing
            #"""
            #j = np.einsum('mij,m->mij', j, sqrt(self.absorption_factor(d, k_s)))
            #self.intensity_x, self.intensity_y = calc_final_jones_intensities(j)
            #"""

            if self.wp_type == 'λ/2':
                #res_int = sum((1 - j[:, 1, 0] * conj(j[:, 1, 0])) ** 2 + (j[:, 0, 0] * conj(j[:, 0, 0])) ** 2) # OG loss
                # jan loss function : No I_x
                #res_no_x = sum((1 - j[:, 1, 0] * conj(j[:, 1, 0])) ** 2)
                #res = sum(j[:, 0, 0].real ** 2 + j[:, 0, 0].imag ** 2 + (1-j[:, 1, 0].real) ** 2 + j[:, 1, 0].imag ** 2)
                #v1, v2, E1, E2 = eig(j)
                #e2x, e2y = (E2[0, :], E2[1, :])
                #res_shift = sum(2*((pi/2 - retardance(j))**2)/pi**2)
                #res = sum(e2y.real) + res_shift
                #res = res_int #+ res_shift
                #trace = sum((j[:, 0, 0] + j[:, 1, 1])**2)
                #res = res_shift + trace
                #res = res_shift
                #res = res_int
                res = sum((j[:, 0, 0].real)**2)
            else:
                # OG intensity loss
                # res = sum((j[:, 1, 0] * conj(j[:, 1, 0]) - j[:, 0, 0] * conj(j[:, 0, 0])) ** 2)
                #int_tot = sum((1-(conj(j[:, 1, 0]) * j[:, 1, 0] + conj(j[:, 0, 0]) * j[:, 0, 0]))**2)
                int_tot = 0
                q = j[:, 0, 0] / j[:, 1, 0]
                res_int = sum(q.real**2 + (q.imag+1)**2) + 0.05*int_tot
                res_shift = 0#sum(2 * ((pi/2 - retardance(j)) ** 2) / pi ** 2)

                res = res_int + res_shift

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

            theta, ret, p_s, p_p = self.m_matrix_input(angles, d, n_s, n_p, k_s, k_p)

            M = self.build_m_matrix_stack(theta, ret, p_s, p_p)

            # adds the full absorption in case of enabled weak absorption for testing
            #"""
            #M = np.einsum('mij,m->mij', M, self.absorption_factor(d, k_s))
            #from modules.utils.calculations import calc_final_stokes_intensities
            #self.intensity_x, self.intensity_y = calc_final_stokes_intensities(M)
            #"""

            if self.wp_type == 'λ/2':
                res = sum((M[:, 0, 0] + M[:, 0, 1] - 1) ** 2) + sum((M[:, 1, 0] + M[:, 1, 1] + 1) ** 2)
            else:
                res = sum((M[:, 0, 0] - M[:, 3, 1]) ** 2) + sum((M[:, 3, 3]) ** 2)  # original lambda / 4

            if self.log_of_res:
                return np.log10(res / self.freq_cnt)

            return res / self.freq_cnt
        
        if self.wp_type == 'mixed':
            return j_stack_err_mixed_wp_type
        if self.settings[self.calculation_method_key] in 'Stokes':
            return m_stack_err
        else:
            return j_stack_err

    def get_j_stack(self, x):
        angles, d, stripes = self.setup_input_vectors(x)

        n_s, n_p, k_s, k_p = self.setup_ri(stripes)

        theta, x, y = self.j_matrix_input(angles, d, n_s, n_p, k_s, k_p)

        return self.build_j_matrix_stack(theta, x, y)

if __name__ == '__main__':
    from modules.settings.settings import Settings
    import matplotlib.pyplot as plt
    from pathlib import Path
    from modules.identifiers.dict_keys import DictKeys
    from modules.utils.calculations import calc_intensity
    keys = DictKeys()

    #dir_1 = Path(r'SLE_l2_const_widths_3/5wp_0.35-1.9THz_shift+int_16-10-30_OptimizationProcess-1')
    dir_1 = Path(r'SLE_l2_const_widths_dbo5/5wp_0.35-1.9THz_best_of_112-12-14_OptimizationProcess-3')
    dir_path_1 = saved_results_dir / dir_1

    dir_2 = Path(r'SLE_l2_longrun_restarts/5wp_0.65-2.2THz_250-850um_22-32-10_OptimizationProcess-1')
    dir_path_2 = saved_results_dir / dir_2

    dir_3_fp = Path(r'fp_results/fp_l2')
    dir_3_fp = saved_results_dir / dir_3_fp

    dir_4_ghz = Path(r'MUT_1H1_QWP_IntOpt_/GHz_QWP_MUT1_12-44-09_Thread-3')
    dir_path_ghz = saved_results_dir / dir_4_ghz

    dir_path = dir_path_ghz

    settings_dict = Settings().load_settings(dir_path / 'settings.json')

    settings_dict[keys.frequency_resolution_multiplier_key] = 1
    settings_dict[keys.min_freq_key] = 0.05
    settings_dict[keys.max_freq_key] = 0.15
    settings_dict[keys.weak_absorption_checkbox_key] = False
    settings_dict[keys.calculation_method_key] = 'Jones'
    settings_dict[keys.anisotropy_p_key] = 1
    settings_dict[keys.anisotropy_s_key] = 1
    settings_dict[keys.const_widths_key] = [0] * int(settings_dict[keys.wp_cnt_key])
    #settings_dict[keys.x_slicing_key] = [[0,4], [4,8], [8,16]]

    erf_setup = ErfSetup(settings_dict)

    angles_ = np.load(dir_path / 'angles.npy')
    d_ = np.load(dir_path / 'widths.npy')
    stripes_ = np.load(dir_path / 'stripes.npy')

    x_res = np.concatenate((angles_, d_, stripes_))

    freqs = erf_setup.frequencies

    j_stack = erf_setup.get_j_stack(x_res)
    intensity_x, intensity_y = calc_final_jones_intensities(j_stack)

    angles_test = np.deg2rad([79.3, 58.6, 37.9, 15.6, 9.1])
    d_test = np.array([635, 495, 495, 495, 310])*um
    stripes_test = np.array([75, 67, 65, 74, 77, 29, 31, 28, 26, 28])*um

    """
    [79.33, 58.55, 37.88, 15.56, 9.07]
    [635.0, 495.0, 495.0, 495.0, 310.0]
    [74.6, 66.5, 64.5, 73.5, 77.0, 29.4, 30.6, 27.9, 26.3, 28.2]
    """

    # errors
    #d_test += 1*np.random.random(d_test.shape) * um
    #stripes_test += 1*np.random.random(stripes_test.shape) * um
    #stripes_test[0:5] += 10*np.ones(5)*um
    #angles_test += np.deg2rad(np.random.random(angles_test.shape))

    #x_res_test = np.concatenate((angles_test, d_test, stripes_test))
    #j_stack_test = erf_setup.get_j_stack(x_res_test)
    #intensity_x_test, intensity_y_test = calc_final_jones_intensities(j_stack_test)

    from py_pol.jones_matrix import Jones_matrix
    from py_pol.jones_vector import Jones_vector

    j_res = Jones_matrix('j_res')
    j_res.from_matrix(j_stack)
    j_res.parameters.field_transmissions = j_res.parameters.transmissions
    j_res.parameters.get_all()

    #plt.plot(freqs, np.abs(j_res.parameters.dict_params['retardance']-pi), label='ret og')
    #plt.plot(freqs, np.abs(j_res_test.parameters.dict_params['retardance']-pi), label='ret test')
    #plt.plot(freqs, np.abs(j_res_fp.parameters.dict_params['retardance']-pi), label='ret fp')
    #plt.legend()
    #plt.show()

    v1, v2, E1, E2 = eig(j_stack)
    e_state = (E1 + E2).transpose(1, 0)

    #eigen_state = Jones_vector('eigen_state_sum')
    #eigen_state.from_matrix(e_state)

    #E1_pypol = Jones_vector('E1')
    #E1_pypol.from_matrix(E1)

    #E2_pypol = Jones_vector('E2')
    #E2_pypol.from_matrix(E2)

    #E1_pypol.parameters.get_all()
    #E2_pypol.parameters.get_all()
    #eigen_state.parameters.get_all()

    #print(E1_pypol.parameters)
    #print(E2_pypol.parameters)
    #print(eigen_state.parameters)

    #print(j_stack.shape)
    #print(e_state.shape)

    j_final = np.einsum('vab,vb->va', j_stack, e_state)

    j_final_pypol = Jones_vector('j_final')
    j_final_pypol.from_matrix(j_final)

    polarizer_y = np.zeros((len(freqs), 2, 2))
    polarizer_x = np.zeros((len(freqs), 2, 2))
    orientations = j_final_pypol.parameters.azimuth()
    for i, angle in enumerate(orientations):
        r, r_inv = r_z_j(angle), r_z_j(-angle)
        polarizer_y[i] = np.dot(r_inv, np.dot(y_pol_j, r))
        r, r_inv = r_z_j(angle+pi/2), r_z_j(-angle+pi/2)
        polarizer_x[i] = np.dot(r_inv, np.dot(x_pol_j, r))

    x_pol_state = np.einsum('vab,vb->va', polarizer_x, j_final)
    y_pol_state = np.einsum('vab,vb->va', polarizer_y, j_final)

    #x_pol_state = np.einsum('ab,vb->va', x_pol_j, j_final)
    #y_pol_state = np.einsum('ab,vb->va', y_pol_j, j_final)

    int_x_e, int_y_e = calc_intensity(x_pol_state), calc_intensity(y_pol_state)


    #print(j_res.parameters)
    # print(j_res.checks)
    #plt.plot(freqs, int_x_e, label='e after x-pol')
    #plt.plot(freqs, int_y_e, label='e after y-pol')
    plt.plot(freqs, intensity_x, label='x-pol')
    plt.plot(freqs, intensity_y, label='y-pol')
    #plt.plot(freqs, intensity_x_fp, label='fp x-pol')
    #plt.plot(freqs, intensity_y_fp, label='fp y-pol')
    #plt.plot(freqs, intensity_x, label='og x-pol')
    #plt.plot(freqs, intensity_y, label='og y-pol')
    plt.legend()
    plt.show()
