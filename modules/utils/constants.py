from pathlib import Path
from numpy import sqrt
from scipy.constants import c
import numpy as np

# constants
sqrt2 = sqrt(2)
c0 = c

# Units
um = 10 ** -6
m_to_um = 10 ** 6
ps = 10 ** -12
THz = 10 ** 12
Hz_to_THz = 10 ** -12
GHz = 10 ** 9

# Mueller Matrices


# Locations
project_folder = Path(__file__).parents[2]

module_folder = project_folder / Path('modules')

settings_module_folder = module_folder / Path('settings')

results_module_folder = module_folder / Path('results')
saved_results_folder = results_module_folder / Path('saved_results')

cst_module_folder = module_folder / Path('cst')
cst_results_folder = cst_module_folder / Path('cst_results')

material_manager_folder = module_folder / Path('material_manager')
data_folders = material_manager_folder / Path('data_folders')


# constant matrices
x_pol_m = 0.5 * np.array([[1, 1, 0, 0],
                          [1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

y_pol_m = 0.5 * np.array([[1, -1, 0, 0],
                          [-1, 1, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])

x_pol_j = np.array([[1, 0],
                    [0, 0]])

y_pol_j = np.array([[0, 0],
                    [0, 1]])

# constant vectors
stokes_initial = np.array([1, 1, 0, 0])
jones_initial = np.array([1, 0])

if __name__ == '__main__':
    print(saved_results_folder)
