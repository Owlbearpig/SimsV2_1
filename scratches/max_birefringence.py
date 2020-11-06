from modules.settings.settings import Settings
import matplotlib.pyplot as plt
from modules.utils.constants import *
from pathlib import Path
import numpy as np
from erf_setup_v21 import ErfSetup
from modules.identifiers.dict_keys import DictKeys
keys = DictKeys()

dir_path = Path(r'/home/alex/Desktop/Projects/SimsV2_1/modules/results/saved_results/SLE_l2_longrun_restarts/l2_200GHz_14-05-27_Thread-2')

settings_dict = Settings().load_settings(dir_path / 'settings.json')
print(settings_dict[keys.selected_material_data_path_key])
settings_dict[keys.weak_absorption_checkbox_key] = False
settings_dict[keys.calculation_method_key] = 'Jones'
settings_dict[keys.anisotropy_p_key] = 1
settings_dict[keys.anisotropy_s_key] = 1

erf_setup = ErfSetup(settings_dict)
erf = erf_setup.erf

angles_ = np.load(dir_path / 'angles.npy')
d_ = np.load(dir_path / 'widths.npy')
stripes_ = np.load(dir_path / 'stripes.npy')
stripes_ = np.array([0.000055, 0.0000205])

best_s0, best_s1 = 0, 0
best_bf = 0
for s0 in np.linspace(10, 80, 100):
    print(s0)
    for s1 in np.linspace(10, 80, 100):
        max_f = c / (2*(s1 + s0)*um)
        max_f_range = np.where(max_f > erf_setup.frequencies)
        stripes_ = np.array([s0, s1])*um
        n = erf_setup.setup_ri(stripes_)
        max_bf = (n[1, max_f_range] - n[0, max_f_range]).max()
        if max_bf > best_bf:
            best_bf = max_bf
            best_s0, best_s1 = s0, s1

print(best_bf)
print(best_s0, best_s1)

"""
plt.plot(erf_setup.frequencies*Hz_to_THz, n[0], label='n_s')
plt.plot(erf_setup.frequencies*Hz_to_THz, n[1], label='n_p')
plt.plot(erf_setup.frequencies*Hz_to_THz, n[1]-n[0])
plt.show()
"""

