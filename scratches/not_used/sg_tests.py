import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from pathlib import Path

"""
Demonstrates one way of embedding PyLab figures into a PySimpleGUI window.
"""

spinbox = sg.Spin(values=('25.0 C', '35.0 C', '78.0 C'), initial_value='25.0 C', font=('Helvetica 35'), size=(20, 1))

layout = [
    [spinbox]
]

window = sg.Window('spinbox', layout)

event, values = window.read()

window.close()