import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from pathlib import Path

"""
Demonstrates one way of embedding PyLab figures into a PySimpleGUI window.
"""

x = np.linspace(0, 10, 100)
y = np.sin(x) ** 2 * np.random.random(x.shape)

fig, ax = plt.subplots()
ax.plot(x, y)


# ------------------------------- Beginning of Matplotlib helper code -----------------------
class Toolbar(NavigationToolbar2Tk):
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)


def onclick(event):
    event_type = 'double' if event.dblclick else 'single'
    print(f'{event_type} click: button={event.button}, x={event.x}, '
          f'y={event.y}, xdata={event.xdata}, ydata={event.ydata}')


def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    # uncomment for toolbar
    #Toolbar(figure_canvas_agg, window['controls_cv'].TKCanvas)
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    figure_canvas_agg.mpl_connect('button_press_event', onclick)
    return figure_canvas_agg


# ------------------------------- Beginning of GUI CODE -------------------------------

def on_folder_browser():
    p = Path(values["folder"]).glob('**/*')
    files = [x.name for x in p if x.is_file()] #and (".csv" in str(x))]
    window['file_listbox'].update(files)


# define the window layout
layout = [
    [sg.Canvas(key='canvas'), sg.Column(
        layout=[
            [sg.In(size=(35, 1), key="folder", enable_events=True), sg.FolderBrowse(key='folder_browse')],
            [sg.Listbox(values=[], key='file_listbox', size=(40, 25))]
        ])],
    [sg.Canvas(key='controls_cv')],
]

# create the form and show it without the plot
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, finalize=True)

# add the plot to the window
fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)

while True:
    event, values = window.read()

    if event == 'folder':
        on_folder_browser()
    if event == sg.WIN_CLOSED:
        break

window.close()
