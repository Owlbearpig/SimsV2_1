from modules.identifiers.dict_keys import DictKeys
import pathlib
from datetime import datetime
import json
from modules.utils.constants import *
from modules.settings.settings import Settings


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pathlib.Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class Saver(DictKeys):
    def __init__(self, settings, new_process):
        super().__init__()
        self.settings = settings
        self.process = new_process

        self.save_name = settings[self.save_name_key]
        self.save_folder_name = settings[self.save_folder_name_key]
        self.process_name = self.process.name

        self.save_folder = self.make_dirs()

        if self.settings[self.save_settings_key]:
            self.save_settings()

    def make_dirs(self):
        save_name = self.settings[self.save_name_key]
        if save_name:
            save_name += ''
        process_name = self.process_name

        if self.save_folder_name:
            folder_name = self.save_folder_name
        else:
            folder_name = datetime.today().strftime('%d-%m-%Y')
        now = datetime.today().strftime('%H-%M-%S')

        save_folder = saved_results_dir / str(folder_name) / (save_name + str(now) + '_' + str(process_name))
        save_folder.mkdir(exist_ok=True, parents=True, mode=0o755)

        return save_folder

    def save_settings(self):
        Settings().save_settings(self.settings, self.save_folder / 'settings.json')

    def save_x(self, f, angles, widths, stripes):
        np.save(self.save_folder / 'f', f)
        np.save(self.save_folder / 'angles', angles)
        np.save(self.save_folder / 'widths', widths)
        np.save(self.save_folder / 'stripes', stripes)


class OutputHandler(Saver):

    def __init__(self, settings, new_process):
        super().__init__(settings, new_process)
        self.max_iterations = settings[self.iterations_key]
        self.save_all_results = self.settings[self.save_all_results_key]
        self.print_interval = self.settings[self.print_interval_key]
        self.save_array = None

    def print_output(self, output):
        if output.new_best:
            print('New best minimum:\n')
            print(output)
        elif not output.iter_cnt % self.print_interval:
            print(': ) : ) Status: : ) : )\n')
            print(output)
        else:
            pass

    def main(self, output):
        self.print_output(output)

        if output.new_best:
            self.process.best_result = output.f
            self.save_x(output.f, output.angles, output.widths, output.stripes)

        if self.save_all_results:
            iteration = output.iter_cnt
            save_interval = min(self.max_iterations, 1000)

            if isinstance(self.save_array, np.ndarray):
                concatenated_values = np.concatenate((np.array([output.f], float),
                                                      output.angles, output.widths, output.stripes))
                self.save_array[(iteration - 1) % save_interval] = concatenated_values
            else:
                len_x = len(output.angles) + len(output.widths) + len(output.stripes)
                self.save_array = np.zeros((save_interval, len_x + 1))
            if not output.iter_cnt % save_interval or output.iter_cnt == self.max_iterations:
                np.save(self.save_folder / str(output.iter_cnt), self.save_array)
                self.save_array = np.zeros(self.save_array.shape)
