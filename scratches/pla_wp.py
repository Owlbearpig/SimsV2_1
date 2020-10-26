import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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
    #dBdata = 10 * np.log10(np.abs(Y))

    return freqs[idx], Y[idx]

fft_ref = fft(np.loadtxt(data_dir / files[0]))

freqs = fft_ref[0]
#print(freqs)
# 6

for file in files:
    file = str(file)
    if 'reference' in file:
        continue
    data = np.loadtxt(data_dir / file)
    fft_file = fft(data)[1]

    plt.plot(freqs, 20*np.log10(np.abs(fft_file)/np.abs(fft_ref[1])))

plt.xlim((0, 0.7))
plt.ylim((-20, 10))
plt.show()


freq = 0.4
freq_index = np.argmin(np.abs(freqs-freq))
print(freqs[freq_index], freq_index)

ft_list, degress = [], []
for file in files:
    file = str(file)
    if 'reference' in file:
        continue

    deg = float(file.split('-')[-1].split(' ')[0])
    data = np.loadtxt(data_dir / file)
    ft = np.abs(fft(data)[1][freq_index])/np.abs(fft_ref[1][freq_index])
    ft_list.append(ft)
    degress.append(deg)
    #plt.plot(ft[0], 20*np.log10(np.abs(ft[1])/np.abs(ref0)), label=file)

    #plt.plot(ft[0], ft[1], label=file)
    #plt.xlim([0.0, 0.6])

plt.plot(degress, ft_list)
plt.xlabel('Angle (Deg)')
plt.ylabel('Transmission')
#plt.legend()
plt.show()
