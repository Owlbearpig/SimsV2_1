import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from pathlib import Path

import PySimpleGUI as sg
# import PySimpleGUIQt as sg
# import PySimpleGUIWeb as sg

"""
    Demonstration of using Multiline elements to "print", uses both redefining "print" and calling Multiline.print
    Demonstration of how to work with multiple colors when outputting text to a multiline element
"""


def main():

    MLINE_KEY = '-MLINE-'+sg.WRITE_ONLY_KEY

    layout = [  [sg.Text('Demonstration of Multiline Element\'s ability to show multiple colors ')],
                [sg.Multiline(size=(60,20), disabled=True, autoscroll=False, key=MLINE_KEY)],
                [sg.B('Plain'), sg.Button('Text Blue Line'), sg.Button('Text Green Line')],
                [sg.Button('Background Blue Line'),sg.Button('Background Green Line'), sg.B('White on Green')]  ]

    window = sg.Window('Demonstration of Multicolored Multline Text', layout)

    # Make all prints output to the multline in Red text
    print = lambda *args, **kwargs: window[MLINE_KEY].print(*args, **kwargs, text_color='red')

    while True:
        event, values = window.read()       # type: (str, dict)
        if event in (None, 'Exit'):
            break
        print(event, values)
        if 'Text Blue' in event:
            window[MLINE_KEY].print('This is blue text', text_color='blue', end='')
        if 'Text Green' in event:
            window[MLINE_KEY].print('This is green text', text_color='green')
        if 'Background Blue' in event:
            window[MLINE_KEY].print('This is Blue Background', background_color='blue')
        if 'Background Green' in event:
            window[MLINE_KEY].print('This is Green Background', background_color='green')
        if 'White on Green' in event:
            window[MLINE_KEY].print('This is white text on a green background', text_color='white', background_color='green')
        if event == 'Plain':
            window[MLINE_KEY].print('This is plain text with no extra coloring')
    window.close()


if __name__ == '__main__':
    main()