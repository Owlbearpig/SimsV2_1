from multiprocessing import Process
import multiprocessing as mp
from erf_setup_v21 import ErfSetup
from optimizer_setup_v2 import OptimizerSetup
from modules.identifiers.dict_keys import DictKeys
from optimizer_setup_v2 import Output
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

    def get_combinations(self):
        wp_cnt = self.settings[self.wp_cnt_key]
        d_lst = self.settings[self.dbo_widths_input_key]
        combinations = combinations_with_replacement(d_lst, wp_cnt)
        # add check loadfile stuff here so we  just replace full combinations

        return combinations

    def callback(self, x, f, accept):
        self.iter_cnt += 1
        new_output = DBOOutput(output_identifier='dbo_output')
        new_output['iter_cnt'], new_output['f'] = self.iter_cnt, np.round(f, 5)
        self.queue.put(new_output)

    def on_task_completion(self, combination, opt_res, progress):
        s = str(combination) + ', ' + str(list(opt_res.x)) + '\n'
        with open(self.result_file_path, 'a') as f:
            f.write(s)
        f.close()
        self.iter_cnt = 0
        self.job_progress_output(progress)

    def job_progress_output(self, progress):
        new_output = DBOOutput('dbo_job_progress', progress)
        self.queue.put(new_output)


def make_discrete_bruteforce_optimizer(**kwargs):
    settings = kwargs['settings']
    queue = kwargs['queue']

    new_dbo = DBO(settings, queue)
    combinations = list(new_dbo.get_combinations())
    for i, combination in enumerate(combinations):
        new_dbo.settings[new_dbo.const_widths_key] = list(combination)
        erf_setup = ErfSetup(new_dbo.settings)
        new_optimizer = OptimizerSetup(erf_setup, settings, queue)
        new_optimizer.custom_callback = new_dbo.callback
        opt_res = new_optimizer.start_optimization()

        progress = {'task_cnt': i+1, 'total_task_cnt': len(combinations)}
        new_dbo.on_task_completion(combination, opt_res, progress)


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
        self.name = ''

    def __str__(self):
        return self.name if self.name else "Process " + str(self._identity[0])

    def terminate(self, pname=None):
        # don't think it matters if terminate or kill is used
        super().kill()
        name = pname if pname else str(self)
        self.queue.put(name + ' terminated')

    def start(self, pname=None):
        super().start()
        self.start_time = time.time()
        self.add_task_info()
        name = pname if pname else str(self)
        self.queue.put(name + ' started')

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
        self.name = 'DBO'

    def start(self, pname=None):
        super().start(pname='DBO process')

    def terminate(self, pname=None):
        super().terminate(pname='DBO process')