import numpy as np
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt

top_dir = Path(r'E:\CURPROJECT\SimsV2_1\modules\results\saved_results\DeleteMe')
sub_dirs_1wp = [
'1wp_WA_14-34-52_OptimizationProcess-2',
'1wp_WA_14-50-17_OptimizationProcess-3',
'1wp_WA_14-53-35_OptimizationProcess-4',
'1wp_WA_14-59-02_OptimizationProcess-5',
]
sub_dirs_2wp = [
'2wp_WA_0.0218-08-48_OptimizationProcess-1',
    '2wp_WA_0.0218-08-57_OptimizationProcess-2',
    '2wp_WA_0.0218-08-58_OptimizationProcess-3',
    '2wp_WA_0.0218-08-58_OptimizationProcess-4',
    '2wp_WA_0.0218-08-59_OptimizationProcess-5'
]
sub_dirs_2wp_2 = [
    '2wp_WA_0.0418-54-29_OptimizationProcess-11',
    '2wp_WA_0.0418-55-10_OptimizationProcess-12',
    '2wp_WA_0.0418-55-47_OptimizationProcess-13'
]


for sub_dir in sub_dirs_2wp_2:
    dir = top_dir / sub_dir

    cnt = 0
    for file in dir.iterdir():
        cnt += 1

    files = [dir / (str(i*10**3) + '.npy') for i in range(1, cnt-4)]

    data = np.zeros((1000*len(files), 7))

    for i, file in enumerate(files):
        data[1000*i:1000*(i+1)] = np.load(file)

    plt.plot(data[100:3000,0], label=str(sub_dir))
    #plt.plot(data[0:100,0])
    plt.ylim((-0.01,0.2))
    print(min(data[1:,0]), np.argmin(data[1:,0]))


plt.legend()
plt.show()
