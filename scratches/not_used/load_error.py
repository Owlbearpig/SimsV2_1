import numpy as np


broken_file = r"E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_4\5wp_0.35-1.9THz_close_to_new_best_l215-01-29_OptimizationProcess-1\f2.npy"
broken_file2 = r"E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_4\5wp_0.35-1.9THz_int_best11-25-18_OptimizationProcess-2\f.npy"
working_file = r"E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_3\5wp_0.35-1.9THz_2xshift+int18-53-45_OptimizationProcess-1\f.npy"
working_file2 = r"E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_3\5wp_0.35-1.9THz_big_start_width_int_23-38-18_OptimizationProcess-3\f.npy"
file = r"E:\CURPROJECT\SimsV2_1\modules\results\saved_results\SLE_l2_const_widths_4\5wp_0.35-1.9THz_close_to_new_best_l215-01-29_OptimizationProcess-1\stripes.npy"

np.load(file, allow_pickle=True)

