from pathlib import Path
path = Path(r'E:\CURPROJECT\SimsV2_1\modules\results\DBO\try3_full_abs_.txt')

best_f = [1]
widths = None
with open(path) as file:
    for line in file:
        f = float(line.split(',_,')[0])
        if f < best_f[0]:
            widths = line.split(',_,')[1]
            best_f.insert(0, f)

print(best_f)
print(widths)