import pandas as pd

csv_material_data_file = r'E:\MEGA\AG\BFWaveplates\Data\fused silica parameters\4Eck_D=2042.csv'

df = pd.read_csv(csv_material_data_file)

def fix_keys_ffs(_dict):
    fixed_dict = {}
    for key, value in _dict.items():
        fixed_dict[key.replace(' ','')] = value
    return fixed_dict

df = fix_keys_ffs(df)

"""
Hz_to_GHZ = 10**-9
with open('4Eck_D=2042.txt', 'a') as file:
    for freq, e_r, e_i in zip(df['freq'], df['epsilon_r'], df['epsilon_i']):
        line = f'{freq*Hz_to_GHZ}    {e_r}    {e_i}\n'
        file.write(line)
"""
Hz_to_GHZ = 10**-9
with open('4Eck_D=2042.txt', 'a') as file:
    for freq, e_r, e_i in zip(df['freq'], df['epsilon_r'], df['epsilon_i']):
        line = f'{freq*Hz_to_GHZ}    {e_r}    {0*e_i}\n'
        file.write(line)
