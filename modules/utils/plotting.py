import matplotlib.pyplot as plt
import pandas as pd
from modules.utils.constants import *
from modules.identifiers.dict_keys import DictKeys
from modules.cst.read_cst_data import CSTData


class Plot(DictKeys):
    def __init__(self, ui_window=None):
        super().__init__()
        self.ui_window = ui_window
        self.result = None
        self.cst_result = None

    def set_plotted_values(self, values):
        angles, widths, stripes = values
        np.set_printoptions(suppress=True)
        self.ui_window[self.plotted_error_angles_key].update(str(np.round(np.rad2deg(angles), 2)))
        self.ui_window[self.plotted_error_widths_key].update(str(np.round(widths*m_to_um, 1)))
        self.ui_window[self.plotted_error_stripes_key].update(str(np.round(stripes*m_to_um, 1)))

    def get_plotted_values(self):
        return self.result.get_values()

    # almost every plot has frequency on x-axis
    def simple_plot(self, y, legend_label='', fig_title=None, title=None, y_label=None):
        plt.figure(fig_title)
        plt.title(title)
        x = self.result.erf_setup.frequencies * Hz_to_THz
        plt.plot(x, y, label=legend_label + ' ' + self.result.name.split("_")[0])
        plt.xlabel('Frequency (THz)')
        plt.ylabel(y_label)
        if legend_label:
            plt.legend()
        plt.show(block=False)

    def result_plot(self, target_figure='Result plot'):
        legend_x_polarizer_label = 'X-Polarizer'
        legend_y_polarizer_label = 'Y-Polarizer'
        y_axis_label = 'Intensity (dB)'
        fig_title = target_figure
        int_x, int_y = self.result.calculated_values['intensities']
        self.simple_plot(int_x, legend_label=legend_x_polarizer_label, fig_title=fig_title, y_label=y_axis_label)
        self.simple_plot(int_y, legend_label=legend_y_polarizer_label, fig_title=fig_title, y_label=y_axis_label)

        self.set_plotted_values(self.result.get_values())

    def refractive_index_plot(self, ui_values):
        error_values = self.result.add_errors(ui_values)
        n_s, n_p, k_s, k_p = error_values['refractive_indices']
        fig_title = 'Refractive index plot'
        if ui_values[self.plot_np_key]:
            self.simple_plot(n_p, legend_label='n_p', fig_title=fig_title)
        if ui_values[self.plot_ns_key]:
            self.simple_plot(n_s, legend_label='n_s', fig_title=fig_title)
        if ui_values[self.plot_kp_key]:
            self.simple_plot(k_p, legend_label='k_p', fig_title=fig_title)
        if ui_values[self.plot_ks_key]:
            self.simple_plot(k_s, legend_label='k_s', fig_title=fig_title)

    def birefringence_plot(self, ui_values):
        error_values = self.result.add_errors(ui_values)
        n_s, n_p, k_s, k_p = error_values['refractive_indices']
        bf_real, bf_imag = n_s - n_p, k_s - k_p
        fig_title = 'Birefringence plot'
        if ui_values[self.plot_bf_real_key]:
            self.simple_plot(bf_real, legend_label='Real part', fig_title=fig_title)
        if ui_values[self.plot_bf_imag_key]:
            self.simple_plot(bf_imag, legend_label='Imaginary part', fig_title=fig_title)

    def polarization_plot(self, pol_comps):
        lin_comp, cir_comp = pol_comps
        fig_title = 'Polarization plot'
        y_label = 'Polarization degree'
        legend_lin_comp_label, legend_circular_comp_label = 'Linear component', 'Circular component'
        title = 'Polarization plot'
        self.simple_plot(lin_comp, legend_label=legend_lin_comp_label,
                         fig_title=fig_title, y_label=y_label, title=title)
        self.simple_plot(cir_comp, legend_label=legend_circular_comp_label,
                         fig_title=fig_title, y_label=y_label, title=title)

    def error_plot(self, ui_values):
        error_values = self.result.add_errors(ui_values)

        int_x, int_y = error_values['intensities']

        if ui_values[self.difference_plot_key]:
            self.simple_plot(int_x - int_y, y_label='x-y-difference', fig_title='Difference plot')
        if ui_values[self.polarization_plot_key]:
            self.polarization_plot(error_values['pol_comps'])
        if not ui_values[self.polarization_plot_key] and not ui_values[self.difference_plot_key]:
            fig_title = 'Error plot'
            y_axis_label = 'Intensity (dB)'
            legend_x_polarizer_label, legend_y_polarizer_label = 'X-Polarizer(err)', 'Y-Polarizer(err)'

            self.simple_plot(int_x, legend_label=legend_x_polarizer_label, fig_title=fig_title, y_label=y_axis_label)
            self.simple_plot(int_y, legend_label=legend_y_polarizer_label, fig_title=fig_title, y_label=y_axis_label)

        self.set_plotted_values(error_values['new_values'])

    @staticmethod
    def cst_plot_base(x, y, legend_label):
        plt.figure('CST')
        plt.plot(x, y, label=legend_label)
        plt.xlabel('Frequencies (GHz)')
        plt.ylabel('Intensity (dB)')
        if legend_label:
            plt.legend()
        plt.show(block=False)

    def cst_plot(self, ui_values):
        folder_path = self.cst_result.cst_result_folder_path
        for file_name in self.cst_result.result_file_names:
            file_path = cst_results_dir / folder_path / file_name

            cst_data_loader = CSTData(file_path)
            frequency_axis = cst_data_loader.f_s_parameters
            if ui_values['cst_plot_x']:
                xp_s, xp_e = ui_values[self.plot_port_spinbox_start_x_key], ui_values[self.plot_port_spinbox_end_x_key]
                z24 = cst_data_loader.get_s_parameters(xp_s, xp_e)
                int_x = 20 * np.log10(z24)
                label = f'{file_name} x-pol. ({xp_s}, {xp_e})'.replace('.txt', '')
                self.cst_plot_base(frequency_axis, int_x, legend_label=label)

            if ui_values['cst_plot_y']:
                yp_s, yp_e = ui_values[self.plot_port_spinbox_start_y_key], ui_values[self.plot_port_spinbox_end_y_key]
                z14 = cst_data_loader.get_s_parameters(yp_s, yp_e)
                int_y = 20 * np.log10(z14)
                label = f'{file_name} y-pol. ({yp_s}, {yp_e})'.replace('.txt', '')
                self.cst_plot_base(frequency_axis, int_y, legend_label=label)

    @staticmethod
    def export_data(x, y, file_path):
        df = pd.DataFrame({'x': x,
                           'y': y})
        df.to_csv(plot_data_dir / (file_path.stem + '.csv'), index_label='n')

    def polar_plot(self, ui_values):
        f_min, f_max = self.result.add_angle_resolved_intensities(ui_values)
        self.ui_window[self.actual_min_frequency_key].update(f'({round(f_min, 2)} THz)')
        self.ui_window[self.actual_max_frequency_key].update(f'({round(f_max, 2)} THz)')

        polar_intensities = 20 * np.log10(self.result.calculated_values['polar_intensities'])
        plt.polar(self.result.calculated_values['polar_angles'], polar_intensities)


        data_dir = Path('/home/alex/Desktop/MDrive/AG/BFWaveplates/Data/PLAWP_table2')
        # data_dir = Path('E:\MEGA\AG\BFWaveplates\Data\PLAWP_table1')
        files = list(data_dir.iterdir())
        from modules.utils.calculations import fft
        refs = [file for file in files if 'reference' in str(file)]

        data = np.loadtxt(data_dir / refs[0])  # 180 deg @ index 0

        fft_ref = fft(data[:, 0], data[:, 1])

        freqs = fft_ref[0]
        freq_index_min, freq_index_max = np.argmin(np.abs(freqs - f_min)), np.argmin(np.abs(freqs - f_max))
        res = []
        for file in files:
            file = str(file)
            if 'reference' in file:
                continue

            deg = float(file.split('-')[-1].split(' ')[0])
            data = np.loadtxt(data_dir / file)
            data_ft = np.abs(fft(data[:, 0], data[:, 1])[1][freq_index_min:freq_index_max])
            ref_ft = np.abs(fft_ref[1][freq_index_min:freq_index_max])
            ft = sum(data_ft) / sum(ref_ft)
            res.append([deg, ft])

        key = lambda x: x[0]
        res.sort(key=key)

        theta = [t[0] - 7 for t in res]
        r = [20 * np.log10(t[1]) for t in res]
        mes_data = [[t[0]-7, 20 * np.log10(t[1])] for t in res]
        mes_data.sort(key=lambda x: x[1])
        print(mes_data[0])
        data = list(zip(self.result.calculated_values['polar_angles'], polar_intensities))
        data.sort(key=lambda x: x[1])
        print(np.rad2deg(data[0]))
        plt.polar(np.deg2rad(theta), r, marker=".")

        plt.xlabel('Angle (Deg)')

        plt.show(block=False)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.plot([1,2,3], [4,5,6])
    plt.show()
