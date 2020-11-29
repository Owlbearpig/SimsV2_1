import numpy as np
import matplotlib.pyplot as plt
from modules.cst.read_cst_data import CSTData
from erf_setup_v21 import ErfSetup
from pathlib import Path
from modules.settings.settings import Settings
base_path = Path(r'E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_dbo5\5wp_0.35-1.9THz_best_of_112-12-14_OptimizationProcess-3')

cst_path = r'E:\CURPROJECT\SimsV2_1\modules\cst\cst_results\SLE_l2_sent2hermans\SLE_l2_sent2hermans_4eck.txt'
erf_settings = base_path / 'settings.json'
settings_module = Settings()
cst_data_reader = CSTData(cst_path)
calc_settings = settings_module.load_settings(cst_path)
xp1, xp2 = 2, 3
yp1, yp2 = 2, 4

x_cst_data = cst_data_reader.get_s_parameters(xp1, xp2)
y_cst_data = cst_data_reader.get_s_parameters(yp1, yp2)

erf_setup = ErfSetup(calc_settings)
angles = np.load(base_path / 'angles.npy')
widths = np.load(base_path / 'widths.npy')
stripes = np.load(base_path / 'stripes.npy')

x0 = np.concatenate((angles, widths, stripes))
erf_setup.erf(x0)



