from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


base_path = Path(r'/home/alex/Desktop/MDrive/AG/BFWaveplates/Data/Styro')

res_path = base_path / 'res'

files = [file for file in res_path.iterdir() if '.csv' in file.name]

linestyles = {'F-10-P_dick_D=8890.csv': '-', 'F-10-P_mittel_D=5580.csv': '-', 'F-C_dick_D=48550.csv': '--', 'F-C_dünn_D=26820.csv': '--',
              'F-Old_dick_D=19710.csv': '-.', 'F_old_dünn_D=8570.csv': '-.', 'F-Sh_dick_D=18440.csv': ':', 'F-Sh_mittel_dick_D=11610.csv': ':'}

d = {'F-10-P_dick_D=8890.csv': 8890, 'F-10-P_mittel_D=5580.csv': 5580, 'F-C_dick_D=48550.csv': 48550, 'F-C_dünn_D=26820.csv': 26820,
    'F-Old_dick_D=19710.csv': 19710, 'F_old_dünn_D=8570.csv': 8570, 'F-Sh_dick_D=18440.csv': 18440, 'F-Sh_mittel_dick_D=11610.csv': 11610}

dd = {'F-10-P_dick_D=8890.csv': 130, 'F-10-P_mittel_D=5580.csv': 140, 'F-C_dick_D=48550.csv': 140, 'F-C_dünn_D=26820.csv': 50,
    'F-Old_dick_D=19710.csv': 100, 'F_old_dünn_D=8570.csv': 70, 'F-Sh_dick_D=18440.csv': 30, 'F-Sh_mittel_dick_D=11610.csv': 140}

for file in files:
    data = pd.read_csv(file)
    freqs = data['freq            '] * 10**-12
    ref_ind = data['ref_ind         ']
    delta_n = (ref_ind-1)*dd[file.name]/d[file.name]
    #delta_n = data['delta_N         ']

    plt.plot(freqs, ref_ind, label=file.stem, linestyle=linestyles[file.name])
    #plt.errorbar(freqs, ref_ind, delta_n, np.zeros_like(freqs))
    plt.fill_between(freqs, ref_ind - delta_n, ref_ind + delta_n, alpha=0.25)

plt.xlabel('Frequency (THz)')
plt.ylabel('RI')
plt.legend()
plt.show()

