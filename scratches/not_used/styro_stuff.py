from pathlib import Path
import numpy as np
from scipy.constants import c as c0
import pandas as pd
import matplotlib.pyplot as plt


base_path = Path(r'/home/alex/Desktop/MDrive/AG/BFWaveplates/Data/Styro')

res_path = base_path / 'res'

files = [file for file in res_path.iterdir() if '.csv' in file.name]

linestyles = {'F-10-P_dick_D=8890.csv': '-', 'F-10-P_mittel_D=5580.csv': '-', 'F-C_dick_D=48550.csv': '--', 'F-C_dünn_D=26820.csv': '--',
              'F-Old_dick_D=19710.csv': '-.', 'F_old_dünn_D=8570.csv': '-.', 'F-Sh_dick_D=18440.csv': ':', 'F-Sh_mittel_dick_D=11610.csv': ':'}

d = {'F-10-P_dick_D=8890.csv': 8890, 'F-10-P_mittel_D=5580.csv': 5580, 'F-C_dick_D=48550.csv': 48550, 'F-C_dünn_D=26820.csv': 26820,
    'F-Old_dick_D=19710.csv': 19710, 'F_old_dünn_D=8570.csv': 8570, 'F-Sh_dick_D=18440.csv': 18440, 'F-Sh_mittel_dick_D=11610.csv': 11610}

delta_d = {'F-10-P_dick_D=8890.csv': 130, 'F-10-P_mittel_D=5580.csv': 140, 'F-C_dick_D=48550.csv': 140, 'F-C_dünn_D=26820.csv': 50,
    'F-Old_dick_D=19710.csv': 100, 'F_old_dünn_D=8570.csv': 70, 'F-Sh_dick_D=18440.csv': 30, 'F-Sh_mittel_dick_D=11610.csv': 140}




def fix_keys_ffs(_dict):
    fixed_dict = {}
    for key, value in _dict.items():
        fixed_dict[key.replace(' ','')] = value
    return fixed_dict

for file in files:
    data = fix_keys_ffs(pd.read_csv(file))
    freqs = data['freq'] * 10**-12
    ref_ind = data['ref_ind']
    tl = data['epsilon_i']/data['epsilon_r']

    eps_i, eps_r = data['epsilon_i'], data['epsilon_r']
    delta_n = (ref_ind - 1) * delta_d[file.name] / d[file.name]
    delta_eps_r = 2*ref_ind*delta_n
    delta_eps_i = data['delta_Eps_2']
    dtl = np.sqrt((delta_eps_i/eps_r)**2 + delta_eps_r**2 * (eps_i/eps_r**2)**2)

    abs_ = data['alpha']
    delta_abs = data['delta_A']

    #plt.plot(freqs, ref_ind, label=file.stem, linestyle=linestyles[file.name])
    #plt.plot(freqs, tl, label=file.stem, linestyle=linestyles[file.name])
    plt.plot(freqs, abs_, label=file.stem, linestyle=linestyles[file.name])
    #plt.errorbar(freqs, ref_ind, delta_n, np.zeros_like(freqs))
    #plt.fill_between(freqs, ref_ind - delta_n, ref_ind + delta_n, alpha=0.25)
    #plt.fill_between(freqs, tl - dtl, tl + dtl, alpha=0.25)
    plt.fill_between(freqs, abs_ - delta_abs, abs_ + delta_abs, alpha=0.25)


plt.xlabel('Frequency (THz)')
plt.ylabel(r'$\alpha$ $(cm^{-1})$')
plt.legend()
plt.show()

