import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from modules.utils.constants import plot_data_dir
import pandas


data_dir = Path('E:\MEGA\AG\BFWaveplates\Data\PLAWP')
files = list(data_dir.iterdir())

def fft(data):
    delta = np.float(np.mean(np.diff(data[:, 0])))
    Y = np.fft.rfft(data[:, 1], axis=0)
    freqs = np.fft.rfftfreq(len(data[:, 0]), delta)
    dBdata = 10 * np.log10(np.abs(Y))

    return freqs, dBdata

ref0 = fft(np.loadtxt(data_dir / files[1]))[1]

for file in files:
    data = np.loadtxt(data_dir / file)
    ft = fft(data)
    #plt.plot(ft[0], np.abs(ft[1] - ref0), label=file)
    """
    df = pandas.DataFrame({'x': data[:, 0],
                           'y': data[:, 1]})
    df.to_csv(plot_data_dir / 'fft' / (file.stem + '.csv'), index_label='n')
    """
    plt.plot(ft[0], ft[1], label=file)
    plt.xlim([-0.1, 1.5])

plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity (dB)')
plt.legend()
plt.show()