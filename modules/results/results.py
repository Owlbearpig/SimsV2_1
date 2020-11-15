from erf_setup_v21 import ErfSetup
from modules.identifiers.dict_keys import DictKeys
from modules.utils.calculations import (make_m_matrix_stack, make_j_matrix_stack,
                                        calc_final_stokes_intensities, calc_final_jones_intensities,
                                        calculate_final_vectors, calc_polarization_degrees_m,
                                        calc_polarization_degrees_j, rotate_matrix)
from modules.utils.helpers import search_dir
from modules.utils.constants import *
from modules.settings.settings import Settings
from datetime import datetime


class Result(DictKeys):
    def __init__(self, ui_values=None):
        super().__init__()
        if ui_values:
            self.result_date_dir_name = ui_values[self.folder_list_key][0]
            self.name = ui_values[self.result_list_key][0]
            self.result_full_path = self.set_result_path()
            self.result_settings, self.original_result_settings = self.set_result_settings(ui_values)
            self.f, self.angles, self.widths, self.stripes = self.set_values()
        self.err_plot_title = ''
        self.x = None
        self.actual_selected_frequency = None
        self.erf_setup = None
        self.calculated_values = {}
        self.error_values = {}

    def __repr__(self):
        return str(self.name)

    def set_result_path(self):
        return saved_results_dir / self.result_date_dir_name / self.name

    def set_result_settings(self, ui_values):
        result_settings_path = self.result_full_path / 'settings.json'
        settings_dict = Settings().load_settings(result_settings_path)
        original_result_settings = settings_dict.copy()

        settings_dict[self.min_freq_key] = ui_values[self.min_freq_key]
        settings_dict[self.max_freq_key] = ui_values[self.max_freq_key]
        settings_dict[self.frequency_resolution_multiplier_key] = ui_values[self.frequency_resolution_multiplier_key]
        settings_dict[self.weak_absorption_checkbox_key] = ui_values[self.weak_absorption_checkbox_key]

        return settings_dict, original_result_settings

    def set_values(self):
        f = np.load(self.result_full_path / 'f.npy', allow_pickle=True)
        angles = np.load(self.result_full_path / 'angles.npy', allow_pickle=True)
        widths = np.load(self.result_full_path / 'widths.npy', allow_pickle=True)
        stripes = np.load(self.result_full_path / 'stripes.npy', allow_pickle=True)

        return f, angles, widths, stripes

    def get_values(self):
        return self.angles, self.widths, self.stripes

    # should be used for displaying in ui only
    # values rounded. Deg, um, um
    def get_rounded_values(self):
        angles, widths, stripes = self.get_values()
        f = np.round(self.f, 7)
        rounded_angles = np.round(np.rad2deg(angles), 2)
        rounded_widths = np.round(widths * m_to_um, 1)
        rounded_stripes = np.round(stripes * m_to_um, 1)

        return f, rounded_angles, rounded_widths, rounded_stripes

    def replace_stripes(self, ui_values, stripes):
        if ui_values[self.overwrite_stripes_key]:
            l0 = float(ui_values[self.stripe0_input_key]) * um
            l1 = float(ui_values[self.stripe1_input_key]) * um
            replaced_stripes = np.array([l0 * np.ones(len(stripes) // 2),
                                         l1 * np.ones(len(stripes) // 2)]).flatten()
            return replaced_stripes
        else:
            return stripes

    # add random errors (and probably my own errors :) to the input
    def add_errors(self, ui_values):
        angles, widths, stripes = self.get_values()
        new_angles, new_widths, new_stripes = angles.copy(), widths.copy(), stripes.copy()
        angle_err_enabled = ui_values[self.angle_err_checkbox_key]
        width_err_enabled = ui_values[self.width_err_checkbox_key]
        stripe_err_enabled = ui_values[self.stripe_err_checkbox_key]

        new_stripes = self.replace_stripes(ui_values, new_stripes)

        # add random errors
        title = ''

        if angle_err_enabled:
            multiplier = ui_values[self.angle_err_slider_key]
            angle_error = 2 * multiplier * np.random.random(angles.shape) - multiplier
            new_angles += np.deg2rad(angle_error)
            title += 'Angle err: ' + str(np.round(angle_error, 2)) + ' (deg)' + '\n'
        if width_err_enabled:
            multiplier = ui_values[self.width_err_slider_key]
            width_error = 2 * multiplier * np.random.random(widths.shape) - multiplier
            new_widths += width_error * um
            title += 'Width err: ' + str(np.round(width_error, 2)) + ' (μm)' + '\n'
        if stripe_err_enabled:
            multiplier = ui_values[self.stripe_err_slider_key]
            stripe_error = 2 * multiplier * np.random.random(stripes.shape) - multiplier
            new_stripes += stripe_error * um
            title += 'Stripe err: ' + str(np.round(stripe_error, 2)) + ' (μm)'

        self.error_values['title'] = title
        self.error_values['new_values'] = np.abs(new_angles), np.abs(new_widths), np.abs(new_stripes)
        self.error_values['refractive_indices'] = self.erf_setup.setup_ri(new_stripes)
        self.error_values['m_matrix_stack'] = make_m_matrix_stack(self.erf_setup,
                                                                  self.error_values['refractive_indices'],
                                                                  (new_angles, new_widths, new_stripes))
        self.error_values['j_matrix_stack'] = make_j_matrix_stack(self.erf_setup,
                                                                  self.error_values['refractive_indices'],
                                                                  (new_angles, new_widths, new_stripes))
        if ui_values[self.calculation_method_key] in 'Stokes':
            self.error_values['intensities'] = calc_final_stokes_intensities(self.error_values['m_matrix_stack'])
            self.error_values['pol_comps'] = calc_polarization_degrees_m(self.error_values['m_matrix_stack'])
        else:
            self.error_values['intensities'] = calc_final_jones_intensities(self.error_values['j_matrix_stack'])
            self.error_values['pol_comps'] = calc_polarization_degrees_j(self.error_values['j_matrix_stack'])

        return self.error_values

    def add_angle_resolved_intensities(self, ui_values):
        f_min, f_max = ui_values[self.polar_plot_min_freq_input_key], ui_values[self.polar_plot_max_freq_input_key]
        freqs = self.erf_setup.frequencies * Hz_to_THz
        f_min_index, f_max_index = np.argmin(np.abs(f_min-freqs)), np.argmin(np.abs(f_max-freqs))
        angles = np.linspace(0, 2*pi, 720)
        j = self.calculated_values['j_matrix_stack'][f_min_index:f_max_index]

        polar_intensities = np.zeros(len(angles))
        for i, theta in enumerate(angles):
            rotated_stack = rotate_matrix(j, theta)
            int_x = rotated_stack[:, 0, 0] * np.conjugate(rotated_stack[:, 0, 0])
            polar_intensities[i] = np.sum(int_x).real / (f_max_index-f_min_index)

        self.calculated_values['polar_intensities'] = polar_intensities
        self.calculated_values['polar_angles'] = angles

        return freqs[f_min_index, 0], freqs[f_max_index, 0]


class CSTResult(DictKeys):
    def __init__(self, ui_values):
        super().__init__()
        self.cst_result_folder_path = Path(ui_values[self.cst_folders_key][0])
        self.result_file_names = ui_values[self.cst_file_list_key]
        self.name = ui_values[self.cst_file_list_key][0]


class Results(DictKeys):
    def __init__(self):
        super().__init__()
        self.cst_results_folder_path = None
        self.selected_result = None
        self.selected_cst_result = None
        self.result_selected = False

    # on result_list selection
    def set_selected_result(self, ui_values_dict):
        self.selected_result = Result(ui_values_dict)
        self.selected_result.erf_setup = ErfSetup(self.selected_result.result_settings)

        # calculate stuff that's required everytime a result is selected
        refractive_indices = self.selected_result.erf_setup.setup_ri(self.selected_result.stripes)
        self.selected_result.calculated_values['refractive_indices'] = refractive_indices

        m_matrix_stack = make_m_matrix_stack(self.selected_result.erf_setup, refractive_indices,
                                             self.selected_result.get_values())
        self.selected_result.calculated_values['m_matrix_stack'] = m_matrix_stack

        j_matrix_stack = make_j_matrix_stack(self.selected_result.erf_setup, refractive_indices,
                                             self.selected_result.get_values())
        self.selected_result.calculated_values['j_matrix_stack'] = j_matrix_stack

        if ui_values_dict[self.calculation_method_key] in 'Stokes':
            self.selected_result.calculated_values['intensities'] = calc_final_stokes_intensities(m_matrix_stack)
        else:
            self.selected_result.calculated_values['intensities'] = calc_final_jones_intensities(j_matrix_stack)

        self.result_selected = True

    # on cst_file_list selection
    def set_selected_cst_result(self, ui_values):
        self.selected_cst_result = CSTResult(ui_values)

    @staticmethod
    def result_names(selected_dir):
        result_dirs = search_dir(saved_results_dir / selected_dir, object_type='dir')
        # check if the result folder has a f.npy file in it
        valid_results = []
        for result_dir in result_dirs:
            f_file_path = saved_results_dir / selected_dir / result_dir / 'f.npy'
            if f_file_path.absolute().exists():
                valid_results.append(result_dir)
        return sorted(valid_results, reverse=True)

    @staticmethod
    def dir_names():
        saved_results_list = search_dir(saved_results_dir, object_type='dir', iterative_search=False)

        def is_date_dir(dir_name):
            try:
                datetime.strptime(dir_name, "%d-%m-%Y")
                return True
            except ValueError:
                return False

        date_dirs, other_dirs = [], []
        for dir_ in saved_results_list:
            if is_date_dir(dir_):
                date_dirs.append(dir_)
            else:
                other_dirs.append(dir_)

        # in place : )
        date_dirs.sort(key=lambda date: datetime.strptime(date, "%d-%m-%Y"), reverse=True)
        if other_dirs:
            other_dirs.sort()
            date_dirs.extend(other_dirs)
        return date_dirs

    def result_info_frame_values(self):
        f, angles, widths, stripes = self.selected_result.get_rounded_values()
        f_str = "F: " + str(f)
        angles_str = str(list(angles))
        widths_str = str(list(widths))
        stripes_str = str(list(stripes))
        total_width = str(np.round(np.sum(self.selected_result.widths) * um_to_mm, 2))

        return {"f": f_str, "angles": angles_str, "widths": widths_str, 'total_width': total_width,
                "stripes": stripes_str}

    def key_group_to_string(self, label):
        ret_str = ''
        for key in DictKeys().__dict__.values():
            if key.label == label:
                try:
                    ret_str += str(key) + ': ' + str(self.selected_result.original_result_settings[key]) + '\n'
                except KeyError:
                    continue
        return ret_str

    def get_final_state_info(self, selected_frequency):
        m_matrix_stack = self.selected_result.calculated_values['m_matrix_stack']
        j_matrix_stack = self.selected_result.calculated_values['j_matrix_stack']
        erf_setup = self.selected_result.erf_setup

        sf, jf, actual_selected_frequency = calculate_final_vectors(m_matrix_stack, j_matrix_stack,
                                                                    erf_setup, selected_frequency)
        numpy_settings = np.seterr(all="ignore")
        stokes_parameters = sf.parameters.get_all()
        self.selected_result.actual_selected_frequency = actual_selected_frequency

        parameter_str1, parameter_str2 = '', ''
        for counter, key in enumerate(stokes_parameters):
            value = stokes_parameters[key]
            if key not in ['S_p', 'S_u']:
                value = np.round(value, 3)
            # split the text in two columns, one side has 0.8 of the elements 0.2 for the rest
            if counter < 0.8 * len(stokes_parameters):
                parameter_str1 += str(key) + ': ' + str(value) + '\n'
            else:
                parameter_str2 += str(key) + ': ' + str(value) + '\n'
        np.seterr(**numpy_settings)

        stokes_final_str = str(np.round(sf.parameters.components(), 3))
        jones_final_str = str(np.round(jf.parameters.components(), 3))

        return stokes_final_str, jones_final_str, parameter_str1, parameter_str2

    @staticmethod
    def get_cst_result_file_names(selected_folder):
        cst_result_folder_path = cst_results_dir / selected_folder

        cst_results = search_dir(dir_path=cst_result_folder_path, object_type='file', file_extension='txt')

        return cst_results

    @staticmethod
    def get_cst_folder_names():
        cst_folders = search_dir(dir_path=cst_results_dir, object_type='dirs')
        return cst_folders
