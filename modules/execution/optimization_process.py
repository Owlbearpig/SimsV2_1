from multiprocessing import Process
import multiprocessing as mp
from pathlib import Path
from erf_setup_v21 import ErfSetup
from optimizer_setup_v2 import OptimizerSetup
from modules.identifiers.dict_keys import DictKeys
from optimizer_setup_v2 import Output
import ast
from modules.settings.settings import Settings
from modules.utils.constants import *
import numpy as np
from itertools import combinations_with_replacement
import time


def make_optimizer(**kwargs):
    settings = kwargs['settings']
    queue = kwargs['queue']

    erf_setup = ErfSetup(settings)
    settings[DictKeys().process_name_key] = mp.current_process().name
    new_optimizer = OptimizerSetup(erf_setup, settings, queue)
    new_optimizer.start_optimization()


class DBOOutput(dict):
    def __init__(self, output_identifier=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_identifier = output_identifier

    def __str__(self):
        return str(self.output_identifier) if self.output_identifier else super(DBOOutput, self).__str__()


class DBO(DictKeys):
    def __init__(self, settings, queue):
        super().__init__()
        self.settings = self.set_settings(settings)
        self.save_name = self.settings[self.dbo_save_name_input_key]
        self.result_file_path = self.make_save_file()
        self.iter_cnt = 0
        self.best_f = np.inf
        self.cur_f = np.inf
        self.queue = queue

    def set_settings(self, settings):
        settings[self.process_name_key] = mp.current_process().name

        return settings

    def make_save_file(self):
        result_file_path = dbo_results_dir / self.save_name
        if not result_file_path.suffix:
            result_file_path = str(result_file_path) + '.txt'
        f = open(result_file_path, 'a')
        f.close()
        return result_file_path

    def read_previous_job_output(self):
        previous_job_path = self.settings[self.dbo_continue_job_input_key]
        combinations = []
        with open(previous_job_path, 'r') as f:
            for line in f:
                split_line = line.split(',_,')
                if float(split_line[0]) < self.best_f:
                    self.best_f = float(split_line[0])
                combinations.append(list(ast.literal_eval(split_line[1])))

        return combinations

    def get_combinations(self):
        wp_cnt = self.settings[self.wp_cnt_key]
        d_lst = self.settings[self.dbo_widths_input_key]
        combinations = [list(combination) for combination in combinations_with_replacement(d_lst, wp_cnt)]

        if self.settings[self.dbo_continue_job_checkbox_key]:
            previous_combinations = self.read_previous_job_output()
            for combination in previous_combinations:
                try:
                    combinations.remove(combination)
                except ValueError:
                    continue

        return combinations

    def callback(self, x, f, accept):
        self.iter_cnt += 1
        if f < self.best_f:
            self.best_f = f
        if f < self.cur_f:
            self.cur_f = f
        output = {'iter_cnt': self.iter_cnt, 'f': np.round(self.cur_f, 5)}
        new_output = DBOOutput('dbo_output', output)
        self.queue.put(new_output)

    def on_task_completion(self, combination, opt_res):
        s = str(self.cur_f) + ',_,' + str(combination) + ',_,' + str(list(opt_res.x)) + '\n'
        with open(self.result_file_path, 'a') as f:
            f.write(s)
        f.close()
        self.iter_cnt = 0
        self.cur_f = np.inf

    def job_progress_output(self, progress):
        new_output = DBOOutput('dbo_job_progress', progress)
        self.queue.put(new_output)


if __name__ == '__main__':
    keys = DictKeys()
    settings = {keys.dbo_continue_job_input_key: '/home/alex/Desktop/Projects/SimsV2_1/modules/results/dbo/test2.txt'}
    settings[keys.dbo_save_name_input_key] = 'test2'
    settings[keys.wp_cnt_key] = 5
    settings[keys.dbo_continue_job_checkbox_key] = True
    settings[keys.dbo_widths_input_key] = [520, 420, 320, 560]
    new_dbo = DBO(settings, None)
    print(new_dbo.get_combinations())
    print(new_dbo.read_previous_job_output())


def make_discrete_bruteforce_optimizer(**kwargs):
    settings = kwargs['settings']
    queue = kwargs['queue']

    new_dbo = DBO(settings, queue)
    combinations = new_dbo.get_combinations()
    # return if job is complete
    if not combinations:
        return

    for i, combination in enumerate(combinations):
        progress = {'task_cnt': i + 1, 'total_task_cnt': len(combinations),
                    'cur_combination': combination, 'best_f': new_dbo.best_f}
        new_dbo.job_progress_output(progress)
        new_dbo.settings[new_dbo.const_widths_key] = combination
        erf_setup = ErfSetup(new_dbo.settings)
        new_optimizer = OptimizerSetup(erf_setup, settings, queue)
        new_optimizer.custom_callback = new_dbo.callback
        opt_res = new_optimizer.start_optimization()

        new_dbo.on_task_completion(combination, opt_res)


def no_optimization(settings, new_values, queue):
    angles = np.deg2rad(np.array(new_values[DictKeys().angle_input_key]))
    widths = np.array(new_values[DictKeys().widths_input_key])*um
    stripes = np.array(new_values[DictKeys().stripes_input_key])*um

    new_erf_setup = ErfSetup(settings)

    x0 = np.array([*angles, *widths, *stripes]).flatten()
    f = new_erf_setup.erf(x0)
    output = Output(f, angles, widths, stripes, 'Single run', 1, 1)
    output.new_best = True
    queue.put(output)


class OptimizationProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings = kwargs['kwargs']['settings']
        self.queue = kwargs['kwargs']['queue']
        self.best_result = np.inf
        self.start_time = None
        self.task_info = None
        self.list_index = None
        self.label = ''

    def __str__(self):
        return self.label if self.label else "Process " + str(self._identity[0])

    def terminate(self):
        # don't think it matters if terminate or kill is used
        super().kill()
        self.queue.put(str(self) + ' terminated')

    def start(self):
        super().start()
        self.start_time = time.time()
        self.add_task_info()
        self.queue.put(str(self) + ' started')

    def add_task_info(self):
        erf_setup = ErfSetup(self.settings)
        n, m = erf_setup.wp_cnt, erf_setup.freq_cnt

        self.task_info = {
            'freq_cnt': m,
            'wp_cnt': n,
            'opt_params': len(erf_setup.x0)
        }


class DBOProcess(OptimizationProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'DBO'
