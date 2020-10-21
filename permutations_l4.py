import itertools
from erf_setup_v21 import ErfSetup
from modules.utils.constants import *
import matplotlib.pyplot as plt

d_lst = np.array([520, 515, 495, 330, 310, 830, 635, 320])

from modules.settings.settings import Settings

path = '/home/alex/Desktop/Projects/SimsV2_1/modules/results/saved_results/fp_results_quartz_l4/fp_quartz_l4_12-27-02_Thread-3/settings.json'
settings_dict = Settings().load_settings(path)
erf_setup = ErfSetup(settings_dict)
erf = erf_setup.erf

angles = np.deg2rad([116.1, 236.6, 215.4, 195.0, 187.2])
d = np.array([290.0, 390.0, 430.0, 470.0, 500.0]) * um
stripes = np.array([49.0, 42.0]) * um

x0 = np.concatenate((angles, d, stripes))

frequencies = erf_setup.frequencies * Hz_to_THz
res0 = erf(x0)
int_x0, int_y0 = erf_setup.int_x, erf_setup.int_y
print(d)
print(res0)
print()
best_val = np.inf
best_combo = None
index_of_best = None
combinations = list(itertools.combinations(d_lst, 5))
for i, combination in enumerate(combinations):
    d = np.array(combination) * um - 10 * np.ones_like(combination) * um
    x = np.concatenate((angles, d, stripes))
    d += np.random.random(d.shape) * um
    new_val = erf(x)
    if new_val < best_val:
        best_val = new_val
        best_combo = d
        best_int_x, best_int_y = erf_setup.int_x, erf_setup.int_y
        index_of_best = i

print(best_val)
print(best_combo)
print(f'OG: {combinations[index_of_best]}')

plt.title(f'{combinations[index_of_best]} μm')
plt.plot(frequencies, int_x0, label='Int. x-Pol. res0')
plt.plot(frequencies, int_y0, label='Int. y-Pol. res0')
plt.plot(frequencies, best_int_x, label='Int. x-Pol. best permutation')
plt.plot(frequencies, best_int_y, label='Int. y-Pol. best permutation')
plt.xlabel('Frequency (Thz)')
plt.ylabel('Intensity (dB)')

ax1 = plt.gca()

plt.rcParams.update({'font.size': 22})
ax1.xaxis.set_tick_params(width=7)
ax1.yaxis.set_tick_params(width=7)
ax1.tick_params(direction='in', length=10, width=3, labelsize=18)
ax1.yaxis.label.set_size(24)
ax1.xaxis.label.set_size(24)
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(3)

plt.title(f'{combinations[index_of_best]} μm')
plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity (dB)')
plt.legend(loc=(0.30, 0.1), prop={'size': 22})
plt.show()


