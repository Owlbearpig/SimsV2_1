import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.constants import c

fs_path = r'/home/alex/Desktop/Projects/SimsV2_1/modules/material_manager/data_folders/FusedSilica_4Eck/4Eck_D=2042.csv'
ceramic_fast = r'/home/alex/Desktop/Projects/SimsV2_1/modules/material_manager/data_folders/CeramicFast/Sample1_090deg_1825ps_0µm88Grad_D=3000.csv'
ceramic_slow = r'/home/alex/Desktop/Projects/SimsV2_1/modules/material_manager/data_folders/CeramicSlow/Sample1_000deg_1825ps_0µm-2Grad_D=3000.csv'

paths = [fs_path, ceramic_fast, ceramic_slow]

for file_path in paths:
    path = Path(file_path).absolute()
    df = pd.read_csv(path)

    freq_dict_key = [key for key in df.keys() if "freq" in key][0]
    eps_mat_r_key = [key for key in df.keys() if "epsilon_r" in key][0]
    eps_mat_i_key = [key for key in df.keys() if "epsilon_i" in key][0]

    frequencies = np.array(df[freq_dict_key])
    eps_mat_r = np.array(df[eps_mat_r_key])
    eps_mat_i = np.array(df[eps_mat_i_key])

    plt.plot(frequencies, eps_mat_i, label=path.stem)

plt.legend()
plt.show()


