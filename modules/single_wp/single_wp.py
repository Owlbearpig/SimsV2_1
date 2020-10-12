from erf_setup_v21 import ErfSetup
from modules.identifiers.dict_keys import DictKeys
from modules.utils.constants import *
from modules.utils.calculations import (make_m_matrix_stack, make_j_matrix_stack,
                                        calc_final_stokes_intensities, calc_final_jones_intensities)
from modules.results.results import Result
from modules.utils.plotting import Plot


class SingleWaveplate(DictKeys):
    def __init__(self, ui_values):
        super().__init__()
        self.ui_values = ui_values
        self.single_wp_settings = self.set_single_wp_settings()
        self.erf_setup = ErfSetup(self.single_wp_settings)
        self.single_wp_result = self.make_single_wp_result()
        self.single_wp_plotter = self.setup_plotter()

    def set_single_wp_settings(self):
        single_wp_settings = self.ui_values.copy()
        single_wp_settings[self.wp_cnt_key] = 1
        single_wp_settings[self.const_angles_key] = [0]
        single_wp_settings[self.const_widths_key] = [0]
        single_wp_settings[self.width_pattern_key] = [1]
        single_wp_settings[self.x_slicing_key] = [[0, 1], [1, 2], [2, 4]]

        return single_wp_settings

    def make_single_wp_result(self):
        single_wp_result = Result()
        single_wp_result.erf_setup = self.erf_setup

        return single_wp_result

    def setup_plotter(self):
        single_wp_plotter = Plot()
        single_wp_plotter.result = self.single_wp_result
        single_wp_plotter.result.name = ''

        return single_wp_plotter

    def calculate_refractive_indices(self):
        stripe_width_mat1 = np.array(self.single_wp_settings[self.single_wp_stripe1_width_input_key]) * um
        stripe_width_mat2 = np.array(self.single_wp_settings[self.single_wp_stripe2_width_input_key]) * um
        refractive_indices = self.erf_setup.setup_ri((stripe_width_mat1, stripe_width_mat2))

        return refractive_indices

    def calculate_intensities(self):
        refractive_indices = self.calculate_refractive_indices()
        angle = np.deg2rad(self.single_wp_settings[self.single_wp_angle_input_key])
        width = np.array(self.single_wp_settings[self.single_wp_width_input_key]) * um
        values = (angle, width, None)
        if self.single_wp_settings[self.calculation_method_key] in 'Stokes':
            m_matrix_stack = make_m_matrix_stack(self.erf_setup, refractive_indices, values)
            int_x, int_y = calc_final_stokes_intensities(m_matrix_stack)
        else:
            j_matrix_stack = make_j_matrix_stack(self.erf_setup, refractive_indices, values)
            int_x, int_y = calc_final_jones_intensities(j_matrix_stack)

        return int_x, int_y

    def calculate_zeroth_order_width(self, freq):
        refractive_indices = self.calculate_refractive_indices()
        f_index = np.argmin(np.abs(self.erf_setup.frequencies - freq))
        bf = np.float(np.abs(refractive_indices[0] - refractive_indices[1])[f_index])

        d_l2, d_l4 = c / (2 * freq * bf), c / (4 * freq * bf)
        return d_l2, d_l4

    def single_wp_refractive_indices_plot(self):
        n_p, n_s, k_p, k_s = self.calculate_refractive_indices()
        if self.ui_values[self.single_wp_plot_ns_checkbox_key]:
            self.single_wp_plotter.simple_plot(n_s, legend_label='n_s', fig_title='Single wp refractive indices')
        if self.ui_values[self.single_wp_plot_np_checkbox_key]:
            self.single_wp_plotter.simple_plot(n_p, legend_label='n_p', fig_title='Single wp refractive indices')
        if self.ui_values[self.single_wp_plot_ks_checkbox_key]:
            self.single_wp_plotter.simple_plot(k_s, legend_label='k_s', fig_title='Single wp refractive indices')
        if self.ui_values[self.single_wp_plot_kp_checkbox_key]:
            self.single_wp_plotter.simple_plot(k_p, legend_label='k_p', fig_title='Single wp refractive indices')

    def single_wp_intensity_plot(self):
        int_x, int_y = self.calculate_intensities()
        self.single_wp_plotter.simple_plot(int_x, legend_label='Int. after x polarizer',
                                           fig_title='Single wp intensities', y_label='Intensity (dB)')
        self.single_wp_plotter.simple_plot(int_y, legend_label='Int. after y polarizer',
                                           fig_title='Single wp intensities', y_label='Intensity (dB)')
