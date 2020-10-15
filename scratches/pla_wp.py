import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

base = Path('E:\MEGA\AG\BFWaveplates\Data\PLAWP')
files = [
    r'2020-10-15T15-22-54.690036-vertical-122.000 mm.txt',
    r'2020-10-15T15-27-03.918716-horizontal--122.000 mm.txt',
    r'2020-10-15T15-33-53.530458-45deg_acw--124.000 mm.txt',
    r'2020-10-15T15-41-19.510101-45deg_cw--124.000 mm.txt',
    r'2020-10-15T15-45-02.246547-small_angle_cw-124.000 mm.txt',
    r'2020-10-15T15-52-59.276203-vertical_focus-122.000 mm.txt'
]


def load(path):
    data = np.loadtxt(path)
    return data

def fft(data):
    delta = np.float(np.mean(np.diff(data[:, 0])))
    Y = np.fft.rfft(data[:, 1], axis=0)
    freqs = np.fft.rfftfreq(len(data[:, 0]), delta)
    dBdata = 10 * np.log10(np.abs(Y))

    return freqs, dBdata

ref0 = fft(load(base / files[1]))[1]

for file in files:
    data = load(base / file)
    ft = fft(data)
    #plt.plot(ft[0], np.abs(ft[1] - ref0), label=file)
    plt.plot(ft[0], ft[1], label=file)
    plt.xlim([-0.1, 1.5])

plt.xlabel('Frequency (THz)')
plt.ylabel('Intensity (dB)')
plt.legend()
plt.show()