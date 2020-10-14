import numpy as np
from functools import reduce
import time

wp_cnt = 5
freq_cnt = 55
iterations = 1000
sub_m_shape = (4, 4)

m, n = freq_cnt, wp_cnt
# reduce
M = np.random.random((m, n, *sub_m_shape)) + 1j*np.random.random((m, n, *sub_m_shape))
res_r = np.zeros((m, *sub_m_shape)) + 1j*np.zeros((m, *sub_m_shape))
start_time_r = time.time()
for _ in range(iterations):

    for i in range(m):
        res_r[i] = reduce(np.dot, M[i])

total_r = time.time() - start_time_r
print("reduce", iterations/total_r)

# this can be done on initialization : )
def make_ein_str(wp_cnt):
    import string
    s0 = string.ascii_lowercase + string.ascii_uppercase
    res = ''
    for i in range(wp_cnt):
        res += s0[wp_cnt+2] + s0[i] + s0[i+1] + ','

    # remove last comma
    res = res[:-1]
    res += '->' + s0[wp_cnt+2] + s0[0] + s0[wp_cnt]

    return res


ein_str = make_ein_str(wp_cnt)
path_info = np.einsum_path(ein_str, *(M.transpose((1, 0, 2, 3))), optimize='greedy')

# einsum
res_e = np.zeros((m, *sub_m_shape)) + 1j*np.zeros((m, *sub_m_shape))
start_time_ein = time.time()
for _ in range(iterations):
    np.einsum(ein_str, *(M.transpose((1, 0, 2, 3))), out=res_e, optimize=path_info[0])

total_ein = time.time() - start_time_ein
print("einsum", iterations/total_ein)
print((np.isclose(res_e-res_r, 0)).all())
print()
print(path_info[0])
print(path_info[1])
