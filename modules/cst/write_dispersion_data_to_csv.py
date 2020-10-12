import pandas as pd
import numpy as np
from pathlib import Path

THz = 10**12
f_pnts = 180

freqs = np.linspace(0.05, 2, f_pnts)*THz
eps_r = np.array([3.5721]*f_pnts)
eps_i = np.array([0.053]*f_pnts)


def cst_format():
    data_table = {
        'freq': freqs,
        'eps_r': eps_r,
        'eps_i': eps_i
    }

    df = pd.DataFrame(data=data_table)

    df.to_csv('PLA_const_refInd.csv', index=False)


def python_app_format():
    data_table = {
        'freq': freqs,
        'ref_ind': np.sqrt(eps_r),
        'epsilon_r': eps_r,
        'epsilon_i': eps_i
    }

    df = pd.DataFrame(data=data_table)

    file_name = Path('../../materials/PLA/PLA_const_refInd.csv')

    df.to_csv(file_name, index=False)


python_app_format()
