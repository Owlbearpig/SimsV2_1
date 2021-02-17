import numpy as np
import PySimpleGUI as sg
import pathlib
import os
import pandas as pd

GHz = 10**9
Hz_to_GHZ = 10 ** -9

new_keys = {'frequency/GHz': 'freq', 'epsilon_r\'': 'epsilon_r', 'epsilon_r\'\'': 'epsilon_i'}

def fix_df(df):
    # new header
    df.columns = ['freq', 'epsilon_r', 'epsilon_i']

    # fix freq. unit
    df['freq'] = df['freq'].apply(lambda x: x*GHz)

    return df

# attempt to convert any csv file in given folder
def convert_data_folder(dir_path):
    dir_path = pathlib.Path(dir_path)

    filenames = os.listdir(dir_path)
    csv_files = [file for file in filenames if file[-4:] == '.csv']

    save_dir = dir_path / 'csv_formatted'

    if not os.path.exists(save_dir):
       os.makedirs(save_dir)

    for csv_file in csv_files:
        print(csv_file)
        df_original = pd.read_csv(dir_path / csv_file, sep=";")
        df_new_keys = fix_df(df_original)
        df_new_keys.to_csv(save_dir / csv_file)

layout = [
    [sg.Input(), sg.FolderBrowse('FolderBrowse')],
    [sg.Submit(), sg.Cancel()],
]

window = sg.Window('Test', layout)

while True:
    event, values = window.read()

    if event is None or event == 'Cancel':
        break

    if event == 'Submit':
        # if folder was not selected then use current folder `.`
        foldername = values['FolderBrowse'] or '.'
        convert_data_folder(foldername)

window.close()
