from json import (load as jsonload, dump as jsondump)
from modules.settings.defaults import default_settings_dict
from pathlib import Path
from modules.utils.constants import settings_module_folder
from modules.identifiers.dict_keys import DictKeys
from modules.utils.initial_setup import InitPrep


class Settings(DictKeys):
    def __init__(self):
        super().__init__()
        self.default_settings_file_name = settings_module_folder / Path('default_settings.json')
        self.previous_settings_file_name = settings_module_folder / Path('settings.json')

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

if __name__ == '__main__':
    new_settings = Settings()
    new_settings.make_default_settings()
    new_settings_dict = new_settings.load_settings()

