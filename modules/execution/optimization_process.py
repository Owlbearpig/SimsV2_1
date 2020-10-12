from multiprocessing import Process
import multiprocessing as mp
from erf_setup_v21 import ErfSetup
from optimizer_setup_v2 import OptimizerSetup
from modules.identifiers.dict_keys import DictKeys
from optimizer_setup_v2 import Output
from modules.utils.constants import *
import numpy as np
import time


def make_optimizer(**kwargs):
    settings = kwargs['settings']
    queue = kwargs['queue']

    erf_setup = ErfSetup(settings)
    settings[DictKeys().process_name_key] = mp.current_process().name
    new_optimizer = OptimizerSetup(erf_setup, settings, queue)
    new_optimizer.start_optimization()


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

    def __str__(self):
        return "Process " + str(self._identity[0])

    def terminate(self):
        super().terminate()
        self.queue.put(str(self) + ' terminated')

    def start(self):
        super().start()
        self.start_time = time.time()
        self.queue.put(str(self) + ' started')

    def add_task_info(self):
        erf_setup = ErfSetup(self.settings)
        n, m = erf_setup.wp_cnt, erf_setup.freq_cnt

        self.task_info = {
            'freq_cnt': m,
            'wp_cnt': n,
            'opt_params': len(erf_setup.x0)
        }
