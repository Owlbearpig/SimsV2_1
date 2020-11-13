import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def fft(t, y):
    delta = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(t), delta)
    idx = freqs > 0

    return freqs[idx], Y[idx]

base_dir = Path('/home/alex/Desktop/MDrive/AG/BFWaveplates/Data/Styro/18mm_f_sh/1c')

refs = [base_dir / file for file in base_dir.iterdir() if 'ref' in file.stem]
samples = [base_dir / file for file in base_dir.iterdir() if 'sample' in file.stem]

def post_process(data_files):
    data0 = np.loadtxt(data_files[0])
    data_sets = np.zeros((len(data_files), *data0.shape))

    for i, file_path in enumerate(data_files):
        data = np.loadtxt(file_path)
        data[:, 1] -= np.mean(data[:, 1])
        data_sets[i] = data

    return np.mean(data_sets, axis=0)


d = 18.44 #mm
ref = post_process(refs)
sample = post_process(samples)

freqs, fft_ref = fft(ref[:, 0], ref[:, 1])
freqs_sample, fft_sample = fft(sample[:, 0], sample[:, 1])

T = fft_sample/fft_ref
idx = (freqs > 0.3) & (freqs < 1.0)
f = freqs[idx]#*10**(-12)
phase = np.unwrap(np.angle(T[idx]))

p = np.polyfit(f, phase, 1)
phase -= p[1]

ri = 1 + 0.3*phase/(2*np.pi*f*d)
plt.figure()
plt.plot(f, ri)
plt.ylim(0.9,1.1)
plt.show()