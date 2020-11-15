from modules.utils.calculations import all_combinations
from itertools import combinations_with_replacement


d_lst = [1, 520, 515, 495, 330, 310, 830, 635, 320]
wp_cnt = 5

combinations = all_combinations(d_lst, wp_cnt)  # all combis
combinations = combinations_with_replacement(d_lst, wp_cnt)
# combinations = [list(combination) for combination in combinations_with_replacement(d_lst, wp_cnt)] # bin combi

combinations = [combi for combi in combinations if sum(combi) < 2500]

print(len(combinations))
