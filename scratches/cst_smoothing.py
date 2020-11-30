import numpy as np
import matplotlib.pyplot as plt
from modules.cst.read_cst_data import CSTData
from erf_setup_v21 import ErfSetup
from pathlib import Path
from modules.settings.settings import Settings
from modules.identifiers.dict_keys import DictKeys
from modules.utils.calculations import calc_final_jones_intensities
keys = DictKeys()

# r'/home/alex/Desktop/Projects/SimsV2_1/modules/results/saved_results/SLE_l2_const_widths_dbo5/5wp_0.35-1.9THz_best_of_112-12-14_OptimizationProcess-3'
# r'E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_dbo5\5wp_0.35-1.9THz_best_of_112-12-14_OptimizationProcess-3'
base_path = Path(r'/home/alex/Desktop/Projects/SimsV2_1/modules/results/saved_results/SLE_l2_const_widths_dbo5/5wp_0.35-1.9THz_best_of_112-12-14_OptimizationProcess-3')

# r'E:\CURPROJECT\SimsV2_1\modules\cst\cst_results\SLE_l2_sent2hermans\SLE_l2_sent2hermans_4eck.txt'
# r'/home/alex/Desktop/Projects/SimsV2_1/modules/cst/cst_results/SLE_l2_sent2hermans/SLE_l2_sent2hermans_4eck.txt'
cst_path = r'/home/alex/Desktop/Projects/SimsV2_1/modules/cst/cst_results/SLE_l2_sent2hermans/SLE_l2_sent2hermans_4eck.txt'
cst_path = r'/home/alex/Desktop/Projects/SimsV2_1/modules/cst/cst_results/SLE_l2_sent2hermans/SLE_l2_sent2hermans_4eck_lowres.txt'
settings_module = Settings()
erf_settings = settings_module.load_settings(base_path / 'settings.json')

erf_settings[keys.frequency_resolution_multiplier_key] = 1
erf_settings[keys.min_freq_key] = 0.4
erf_settings[keys.max_freq_key] = 2.0
erf_settings[keys.weak_absorption_checkbox_key] = False
erf_settings[keys.calculation_method_key] = 'Jones'
erf_settings[keys.anisotropy_p_key] = 1
erf_settings[keys.anisotropy_s_key] = 1
erf_settings[keys.const_widths_key] = [0] * int(erf_settings[keys.wp_cnt_key])
erf_settings[keys.x_slicing_key] = [[0, 5], [5, 10], [10, 20]]

erf_setup = ErfSetup(erf_settings)
angles = np.load(base_path / 'angles.npy')
widths = np.load(base_path / 'widths.npy')
stripes = np.load(base_path / 'stripes.npy')

x = np.concatenate((angles, widths, stripes))

j_stack = erf_setup.get_j_stack(x)

intensity_x = calc_final_jones_intensities(j_stack)[0]
intensity_y = calc_final_jones_intensities(j_stack)[1]

erf_freqs = erf_setup.frequencies

cst_data_reader = CSTData(cst_path)
cst_freqs = cst_data_reader.f_s_parameters*10**12
xp1, xp2 = 2, 4
yp1, yp2 = 2, 3

x_cst_data = cst_data_reader.get_s_parameters(xp1, xp2)
y_cst_data = cst_data_reader.get_s_parameters(yp1, yp2)

int_x_cst = 20 * np.log10(x_cst_data)
int_y_cst = 20 * np.log10(y_cst_data)

def smooth(y, y2):
    step = int(len(y)/len(y2))

    smooth_y = np.zeros_like(y2)
    for i in range(len(y2)):

        smooth_y[i] = np.mean(y[step*i:step*(i+1)])

    return smooth_y

"""
cst_smooth_x = smooth(int_x_cst, intensity_x)
cst_smooth_y = smooth(int_y_cst, intensity_y)
plt.plot(cst_freqs, int_x_cst)
plt.plot(cst_freqs[::10][:98], cst_smooth_x)
plt.plot(cst_freqs[::10][:98], cst_smooth_y)
"""

#"""
plt.plot(erf_freqs, intensity_x, label='int x')
plt.plot(erf_freqs, intensity_y, label='int y')
plt.plot(cst_freqs, int_x_cst, label='int x cst')
plt.plot(cst_freqs, int_y_cst, label='int y cst')
#"""
"""
plt.plot(erf_freqs, intensity_y-intensity_x, label='erf int y - int x')
plt.plot(cst_freqs, int_y_cst-int_x_cst, label='cst int y - int x')
"""
plt.legend()
plt.show()
