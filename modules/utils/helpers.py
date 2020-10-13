import pathlib
import ast
from modules.identifiers.dict_keys import DictKeys
from modules.settings.defaults import default_settings_dict
from modules.utils.constants import *
import traceback
import PySimpleGUI as sg
base_folder = pathlib.Path(__file__).parents[1].absolute()


def error_popup(e):
    tb = traceback.format_exc()
    page_len = 2000
    for cnt, i in enumerate(range(0, len(tb), page_len)):
        sg.popup_error(tb[i:i+page_len],
                       title=f'Traceback (Good luck) ({1 + cnt}/{1 + len(tb)//page_len})',
                       non_blocking=True)
    e = str(e)
    if len(e) < 50:
        e = e + ' '*(50-len(e)) + '\(.;.;.)/'
    sg.popup_error(e, title='Error broke it again ;(', non_blocking=True)


def search_dir(dir_path, object_type='files', name='', file_extension='', iterative_search=True, return_path=False):
    # iterative search is deep search, all sub dirs
    if iterative_search:
        search_space = list(dir_path.rglob("*"))
    else:
        search_space = list(dir_path.iterdir())
    if return_path:
        files = [x for x in search_space if x.is_file() and (f'{name}.{file_extension}' in str(x))]
        dirs = [x for x in search_space if x.is_dir() and (f'{name}' in str(x))]
    else:
        files = [x.name for x in search_space if x.is_file() and (f'{name}.{file_extension}' in str(x))]
        dirs = [x.name for x in search_space if x.is_dir() and (f'{name}' in str(x))]
    if object_type.lower() in 'files':
        ret = files
    else:
        ret = dirs
    return ret


def check_x0_wp_cnt(ui_values):
    return len(ui_values[DictKeys().x0_angle_input_key]) != int(ui_values[DictKeys().wp_cnt_key])

def check_values(func):
    def wrapper(self, current_ui_values):
        if check_x0_wp_cnt(current_ui_values):
            sg.PopupNoWait('---Check x0---', title='oof')
            return
        if current_ui_values[DictKeys().birefringence_type_dropdown_key] in 'Form':
            if not current_ui_values[DictKeys().selected_material_data_path_key]:
                sg.PopupNoWait('---No material selected---', title=':(')
                return
        elif current_ui_values[DictKeys().birefringence_type_dropdown_key] in 'Natural':
            if not current_ui_values[DictKeys().selected_slow_material_data_path_key]:
                sg.PopupNoWait('---No slow material selected---', title=':(')
                return
            if not current_ui_values[DictKeys().selected_fast_material_data_path_key]:
                sg.PopupNoWait('---No fast material selected---', title=':(')
                return
        try:
            return func(self, current_ui_values)
        except Exception as e:
            error_popup(e)

    return wrapper


def fix_types(event, ui_values):
    if ui_values is None:
        return event, ui_values

    formatted_dict = {}
    for key in ui_values:
        try:
            # some ui outputs are str -> convert to correct primitive type
            value, default_value = ui_values[key], default_settings_dict[key]
            # just assign to output if type is correct
            if isinstance(value, type(default_value)):
                formatted_dict[key] = value
            else:
                formatted_dict[key] = cast_from_ui(value, default_value)
            # some ui_elements don't have default settings
        except KeyError:
            formatted_dict[key] = ui_values[key]

    return event, formatted_dict


def remove_brackets(s):
    return s.replace("[", "").replace("]", "")


# convert ui string to correct primitive type
def cast_from_ui(value, default_value=None):
    if not value:
        return
    if isinstance(default_value, type(int())):
        return int(value)
    try:
        return ast.literal_eval(value)
    except Exception as e:
        error_popup(e)


# lists look weird in the ui, therefore convert to str before setting ui element value
def cast_to_ui(value):
    if isinstance(value, type(list())):
        return str(value)
    else:
        return value


if __name__ == '__main__':
    test = Path("/home/alex/Desktop/MDrive/AG/BFWaveplates/PycharmProjects/SimsV2_1/modules/results/saved_results")
    print(search_dir(test, object_type='dir'))

