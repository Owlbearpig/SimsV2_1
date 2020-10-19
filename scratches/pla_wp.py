import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from modules.utils.constants import plot_data_dir
import pandas


data_dir = Path('/home/alex/Desktop/MDrive/AG/BFWaveplates/Data/PLAWP')
files = list(data_dir.iterdir())

def fft(data):
    delta = np.float(np.mean(np.diff(data[:, 0])))
    Y = np.fft.fft(data[:, 1], axis=0)
    freqs = np.fft.fftfreq(len(data[:, 0]), delta)
    idx = freqs > 0
    #dBdata = 10 * np.log10(np.abs(Y))

    return freqs[idx], Y[idx]

ref0 = fft(np.loadtxt(data_dir / files[0]))[1]

for file in files:
    data = np.loadtxt(data_dir / file)
    ft = fft(data)
    plt.plot(ft[0], 20*np.log10(np.abs(ft[1])/np.abs(ref0)), label=file)
    """
    df = pandas.DataFrame({'x': data[:, 0],
                           'y': data[:, 1]})
    df.to_csv(plot_data_dir / 'fft' / (file.stem + '.csv'), index_label='n')
    """
    #plt.plot(ft[0], ft[1], label=file)
    plt.xlim([0.0, 0.6])

plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity (dB)')
plt.legend()
plt.show()