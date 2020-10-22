import itertools
from erf_setup_v21 import ErfSetup
from modules.utils.constants import *
import matplotlib.pyplot as plt

d_lst = np.array([520, 515, 495, 330, 310, 830, 635, 320])

from modules.settings.settings import Settings

path = '/home/alex/Desktop/Projects/SimsV2_1/modules/results/saved_results/New4Hermanns_l2/5wp_300-850_700-2000_12-50-04_OptimizationProcess-1/settings.json'
settings_dict = Settings().load_settings(path)
erf_setup = ErfSetup(settings_dict)
erf = erf_setup.erf

angles = np.deg2rad([36.66, 332.72, 179.39, 17.83, 8.31])
d = np.array([518.8, 804.9, 384.5, 636.5, 464.4]) * um
stripes = np.array([50.0, 47.6, 47.1, 32.8, 44.8, 13.8, 19.4, 16.6, 36.1, 14.7]) * um

x0 = np.concatenate((angles, d, stripes))

frequencies = erf_setup.frequencies * Hz_to_THz
res0 = erf(x0)
int_x0, int_y0 = erf_setup.int_x, erf_setup.int_y
print(d)
print(res0)
for t in list(itertools.combinations_with_replacement(d_lst, 5)):
    if t == (830, 830, 830, 830, 830):
        print(t)
print()
best_val = np.inf
best_combo = None
index_of_best = None
combinations = list(itertools.combinations_with_replacement(d_lst, 5))
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


plt.plot(frequencies, int_x0, label='Int. x-Pol. res0')
plt.plot(frequencies, int_y0, label='Int. y-Pol. res0')
plt.plot(frequencies, best_int_x, label='Int. x-Pol. best permutation')
plt.plot(frequencies, best_int_y, label='Int. y-Pol. best permutation')

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
plt.legend(loc=(0.6, 0.6), prop={'size': 22})
plt.show()


