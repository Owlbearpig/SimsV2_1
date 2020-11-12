from pathlib import Path
path = Path(r'E:\CURPROJECT\SimsV2_1\modules\results\DBO\try2_with_0.txt')

best_f = 1
widths = None
with open(path) as file:
    for line in file:
        f = float(line.split(',_,')[0])
        if f < best_f:
            widths = line.split(',_,')[1]
            best_f = f

print(best_f)
print(widths)