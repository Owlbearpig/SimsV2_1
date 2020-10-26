from json import (load as jsonload, dump as jsondump)
from modules.settings.defaults import default_settings_dict
from modules.utils.helpers import search_dir
from modules.identifiers.dict_keys import DictKeys
from modules.utils.initial_setup import InitPrep
from modules.utils.constants import *


class Settings(DictKeys):
    def __init__(self):
        super().__init__()
        self.default_settings_file_name = settings_module_dir / Path('default_settings.json')
        self.previous_settings_file_name = settings_module_dir / Path('settings.json')

    def load_settings(self, json_file_path=None):
        if json_file_path:
            try:
                with open(json_file_path, 'r') as f:
                    return jsonload(f)
            except FileNotFoundError as e:
                print(f'{e} \nLoading previous settings')
        try:
            with open(self.previous_settings_file_name, 'r') as f:
                return jsonload(f)
        except FileNotFoundError:
            print('No previous settings found, loading default')
            try:
                with open(self.default_settings_file_name, 'r') as f:
                    return jsonload(f)
            except FileNotFoundError:
                print('No default settings found, making some new ones :)')
                self.make_default_settings()
                with open(self.default_settings_file_name, 'r') as f:
                    return jsonload(f)

    # save ui settings
    def save_settings(self, ui_values_dict, json_file_path=None):
        # don't save every ui element value
        saved_values = {}
        for key in default_settings_dict:
            try:
                saved_values[key] = ui_values_dict[key]
            except KeyError:
                continue
        if json_file_path:
            self.make_settings_save_file(saved_values, json_file_path)
        else:
            self.make_settings_save_file(saved_values, self.previous_settings_file_name)

    @staticmethod
    def make_settings_save_file(dict_, settings_file_name):
        with open(settings_file_name, 'w') as f:
            jsondump(dict_, f, indent=4)

    def make_default_settings(self):
        self.make_settings_save_file(default_settings_dict, self.default_settings_file_name)

    def single_run_settings(self, ui_values_dict):
        wp_cnt = len(ui_values_dict[self.angle_input_key])
        settings_for_single_run = ui_values_dict.copy()
        settings_for_single_run[self.save_name_key] = ui_values_dict[self.run_once_label_input_key] + '_'
        settings_for_single_run[self.wp_cnt_key] = wp_cnt
        settings_for_single_run[self.const_angles_key] = [0] * wp_cnt
        settings_for_single_run[self.const_widths_key] = [0] * wp_cnt
        settings_for_single_run[self.width_pattern_key] = list(range(1, wp_cnt + 1))
        settings_for_single_run[self.x_slicing_key] = list(InitPrep(settings_for_single_run).x_slices)

        return settings_for_single_run

    def dbo_settings(self, ui_values_dict):
        if ui_values_dict[self.dbo_continue_job_checkbox_key]:
            previous_job_path = Path(ui_values_dict[self.dbo_continue_job_input_key])
            settings_path = previous_job_path.parent / (str(previous_job_path.stem) + '_settings.json')
            settings = self.load_settings(settings_path)
        else:
            save_name = Path(ui_values_dict[self.dbo_save_name_input_key])
            if save_name.suffix:
                save_name = save_name.stem
            settings_path = dbo_results_dir / (str(save_name) + '_settings.json')
            self.save_settings(ui_values_dict, settings_path)
            settings = ui_values_dict

        # we don't want to load this from old ui values
        settings[self.dbo_continue_job_checkbox_key] = ui_values_dict[self.dbo_continue_job_checkbox_key]

        return settings

    def fix_old_settings(self):
        settings_paths = search_dir(saved_results_dir, file_extension='json', return_path=True)
        for settings_file_path in settings_paths:
            settings_dict = self.load_settings(settings_file_path)
            # if settings dict is missing key add default value
            for key in default_settings_dict:
                try:
                    settings_dict[key]
                except KeyError:
                    settings_dict[key] = default_settings_dict[key]
            self.make_settings_save_file(settings_dict, settings_file_path)


if __name__ == '__main__':
    import numpy as np
    path = r'E:\CURPROJECT\SimsV2_1\modules\results\saved_results\Ceramic_New_Absorption_Matrix\4wp_thin_mid_f_range_19-29-30_OptimizationProcess-5\angles.npy'
    np.load(path, allow_pickle=True)

