import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from modules.utils.constants import plot_data_dir
import pandas


data_dir = Path('/home/alex/Desktop/MDrive/AG/BFWaveplates/Data/PLAWP_table1')
#data_dir = Path('E:\MEGA\AG\BFWaveplates\Data\PLAWP_table1')
files = list(data_dir.iterdir())

def fft(data):
    delta = np.float(np.mean(np.diff(data[:, 0])))
    Y = np.fft.fft(data[:, 1], axis=0)
    freqs = np.fft.fftfreq(len(data[:, 0]), delta)
    idx = freqs > 0

    return freqs[idx], Y[idx]

fft_ref = fft(np.loadtxt(data_dir / files[0]))

freqs = fft_ref[0]


"""
degs = [22.0, 40.998, 48.998, 56.0, 69.998, 89.0, 111.999, 133.998, 158.998, 176.999]
for file in files:
    file = str(file)
    if 'reference' in file:
        continue
    deg = float(file.split('-')[-1].split(' ')[0])
    if deg not in degs:
        continue
    data = np.loadtxt(data_dir / file)
    fft_file = fft(data)[1]

    plt.plot(freqs, 20*np.log10(np.abs(fft_file)/np.abs(fft_ref[1])), label=str(deg))

plt.xlim((0, 0.7))
plt.ylim((-20, 10))
plt.legend()
plt.show()
"""

freq_min, freq_max = 0.2, 0.25
freq_index_min = np.argmin(np.abs(freqs-freq_min))
freq_index_max = np.argmin(np.abs(freqs-freq_max))
print(freqs[freq_index_min], freqs[freq_index_max])
ft_list, degress = [], []
res = []
for file in files:
    file = str(file)
    if 'reference' in file:
        continue

    deg = float(file.split('-')[-1].split(' ')[0])
    data = np.loadtxt(data_dir / file)
    ft = sum(np.abs(fft(data)[1][freq_index_min:freq_index_max]))/sum(np.abs(fft_ref[1][freq_index_min:freq_index_max]))
    res.append([deg, ft])

    #plt.plot(ft[0], 20*np.log10(np.abs(ft[1])/np.abs(ref0)), label=file)

    #plt.plot(ft[0], ft[1], label=file)
    #plt.xlim([0.0, 0.6])
key = lambda x: x[0]
res.sort(key=key)

theta = [t[0] for t in res]
r = [t[1] for t in res]

plt.polar(np.deg2rad(theta), r)

plt.xlabel('Angle (Deg)')
#plt.ylabel('Transmission')
#plt.legend()
plt.show()
