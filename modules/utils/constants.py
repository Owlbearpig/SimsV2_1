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
project_dir = Path(__file__).parents[2]

module_dir = project_dir / Path('modules')

settings_module_dir = module_dir / Path('settings')

results_module_dir = module_dir / Path('results')
saved_results_dir = results_module_dir / Path('saved_results')
archived_results_dir = results_module_dir / Path('archived')
dbo_results_dir = results_module_dir / Path('dbo')

cst_module_dir = module_dir / Path('cst')
cst_results_dir = cst_module_dir / Path('cst_results')

material_manager_dir = module_dir / Path('material_manager')
data_dir = material_manager_dir / Path('data_folders')

plot_data_dir = project_dir / Path('plot_data')

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
    print(saved_results_dir)
