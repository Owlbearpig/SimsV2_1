from py_pol.mueller import Mueller
from py_pol.stokes import Stokes
from py_pol.jones_vector import Jones_vector
from py_pol.utils import degrees
import numpy as np
from py_pol.jones_vector import Jones_vector

import PySimpleGUI as sg

sg.theme('BluePurple')

layout = [[sg.Text('Your typed chars appear here:'), sg.Text(size=(15,1), key='-OUTPUT-')],
          [sg.Input(key='-IN-')],
          [sg.Button('Show'), sg.Button('Exit')],
          [sg.Checkbox("abe", disabled=False, key="abe")]]

window = sg.Window('Pattern 2B', layout)

while True:  # Event Loop
    event, values = window.read()
    print(event, values)
    window["abe"].Update(disabled=True)
    print(window["abe"].Disabled)
    if event == sg.WIN_CLOSED or event == 'Exit':
        break
    if event == 'Show':
        # Update the "output" text element to be the value of "input" element
        window['-OUTPUT-'].update(values['-IN-'])

window.close()

