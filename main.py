from modules.utils.helpers import sg
from gui_tabs.tabs import Tabs
from modules.execution.optimization_process import (OptimizationProcess, DBOProcess, make_optimizer, no_optimization,
                                                    make_discrete_bruteforce_optimizer)
from multiprocessing import Queue, JoinableQueue
import time
from modules.execution.stopppable_thread import StoppableThread
from modules.utils.constants import *
from modules.utils.helpers import error_popup, cast_to_ui, fix_types, check_values
from modules.material_manager.materials import ProjectMaterials, Material
from modules.results.results import Results
from modules.results.queue_reader import QueueReader
from modules.results.handle_returns import OutputHandler
import _tkinter
from modules.identifiers.dict_keys import DictKeys
from modules.settings.settings import Settings
from modules.utils.plotting import Plot
from modules.utils.initial_setup import InitPrep
from modules.single_wp.single_wp import SingleWaveplate
from modules.execution.process_manager import ProcessManager
# print = sg.Print
# sg.theme('GreenMono')


class WaveplateApp(DictKeys):

    def __init__(self):
        super().__init__()
        self.new_optimizer = None
        self.new_erf_setup = None

        self.queue = Queue()

        self.output_handlers = {}

        self.process_manager = ProcessManager()

        self.material_manager = ProjectMaterials()

        self.result_manager = Results()

        self.settings_module = Settings()

        self.queue_reader = QueueReader(self)

        previous_settings = self.settings_module.load_settings()
        tabs = Tabs(previous_settings, self.material_manager)

        window_layout = [
            [sg.TabGroup([[sg.Tab('Erf settings', tabs.get_tab1_layout()),
                           sg.Tab('Material settings', tabs.get_tab2_layout()),
                           sg.Tab('Optimization settings', tabs.get_tab3_layout()),
                           sg.Tab('Saver settings', tabs.get_tab4_layout()),
                           sg.Tab('Run', tabs.get_tab5_layout()),
                           sg.Tab('Result', tabs.get_tab6_layout()),
                           sg.Tab('Plots', tabs.get_tab7_layout()),
                           sg.Tab('CST results', tabs.get_tab8_layout()),
                           sg.Tab('Single waveplate', tabs.get_tab9_layout()),
                           ]])
             ]
        ]

        self.window = sg.Window('Waveplates', window_layout, default_element_size=(120, 120))

        self.plotter = Plot(self.window)

    # ---------------------------------------Erf. settings------------------------------------------------------------ #
    def add_material_to_list(self, ui_values):
        material_name = ui_values[self.new_material_name_key]
        material_path = ui_values[self.new_material_path_key]
        material = Material(material_name, material_path)

        self.material_manager.material_list.append(material)

        material.add_material_from_file()
        updated_material_list = [str(material) for material in self.material_manager.material_list]
        self.window[self.material_drop_down_key].update(values=updated_material_list)
        self.window[self.fast_material_dropdown_key].update(values=updated_material_list)
        self.window[self.slow_material_dropdown_key].update(values=updated_material_list)

    def on_material_selection(self, ui_values):
        form_bf_drop_down_value = ui_values[self.material_drop_down_key]
        form_bf_material_object = self.material_manager.get_material(form_bf_drop_down_value)
        self.window[self.selected_material_data_path_key].update(form_bf_material_object.path)

        slow_material_drop_value = ui_values[self.slow_material_dropdown_key]
        slow_material_object = self.material_manager.get_material(slow_material_drop_value)
        self.window[self.selected_slow_material_data_path_key].update(slow_material_object.path)

        fast_material_drop_value = ui_values[self.fast_material_dropdown_key]
        fast_material_object = self.material_manager.get_material(fast_material_drop_value)
        self.window[self.selected_fast_material_data_path_key].update(fast_material_object.path)

    def new_initial_values(self, ui_values):
        initial_setup = InitPrep(ui_values)
        self.window[self.x0_angle_input_key].update(str(initial_setup.angles0))
        self.window[self.x0_widths_input_key].update(str(initial_setup.widths0))
        self.window[self.x0_stripes_input_key].update(str(initial_setup.stripes0))
        self.window[self.x_slicing_key].update(str(list(initial_setup.x_slices)))

    def load_default_settings(self, *args):
        default_settings_dict = self.settings_module.load_settings(self.settings_module.default_settings_file_name)
        self.set_window_element_values(default_settings_dict)

    def on_wp_cnt_change(self, ui_values):
        old_angle_lst = ui_values[self.const_angles_key]
        old_widths_lst = ui_values[self.const_widths_key]
        old_stripes_lst = ui_values[self.initial_stripe_widths_key]

        new_wp_cnt = ui_values[self.wp_cnt_key]
        old_wp_cnt = len(old_angle_lst)

        stripe_width_mat0 = ui_values[self.initial_stripe_widths_key][0][0]
        stripe_width_mat1 = ui_values[self.initial_stripe_widths_key][1][0]

        if new_wp_cnt >= old_wp_cnt:
            wp_cnt_diff = new_wp_cnt - old_wp_cnt
            new_angle_lst = old_angle_lst + [0] * wp_cnt_diff
            new_widths_lst = old_widths_lst + [0] * wp_cnt_diff
            new_stripes_lst_mat0 = old_stripes_lst[0] + [stripe_width_mat0] * wp_cnt_diff
            new_stripes_lst_mat1 = old_stripes_lst[1] + [stripe_width_mat1] * wp_cnt_diff
        else:
            new_angle_lst = old_angle_lst[:new_wp_cnt]
            new_widths_lst = old_widths_lst[:new_wp_cnt]
            new_stripes_lst_mat0 = old_stripes_lst[0][:new_wp_cnt]
            new_stripes_lst_mat1 = old_stripes_lst[1][:new_wp_cnt]

        if ui_values[self.wp_type_key] == 'mixed':
            new_angle_lst *= 2

        self.window[self.const_angles_key].update(cast_to_ui(new_angle_lst))
        self.window[self.const_widths_key].update(cast_to_ui(new_widths_lst))
        self.window[self.initial_stripe_widths_key].update(cast_to_ui([new_stripes_lst_mat0,
                                                                       new_stripes_lst_mat1]))
        self.window[self.width_pattern_key].update(cast_to_ui(list(range(1, new_wp_cnt + 1))))

    # ---------------------------------------Run tab------------------------------------------------------------------ #
    # uses current ui and 'Run once' frame values
    @check_values
    def run_once(self, ui_values):
        settings_for_single_run = self.settings_module.single_run_settings(ui_values)
        thread = StoppableThread(target=no_optimization, args=(settings_for_single_run, ui_values, self.queue))
        self.output_handlers['Single run'] = OutputHandler(settings_for_single_run, thread)

        thread.start()

    @check_values
    def discrete_bruteforce_optimization(self, ui_values):
        dbo_settings = self.settings_module.dbo_settings(ui_values)
        dbo_process = DBOProcess(target=make_discrete_bruteforce_optimizer,
                                 kwargs={'queue': self.queue, 'settings': dbo_settings})
        self.process_manager.running_processes.append(dbo_process)
        self.update_process_list()
        dbo_process.start()

    @check_values
    def new_optimization_process(self, ui_values):
        # save settings in case of crash (:
        self.settings_module.save_settings(ui_values)
        new_process = OptimizationProcess(target=make_optimizer, kwargs={'queue': self.queue, 'settings': ui_values})
        self.output_handlers[new_process.name] = OutputHandler(ui_values, new_process)

        self.process_manager.running_processes.append(new_process)
        self.update_process_list()
        new_process.start()

    def stop_selected_process(self, *args):
        self.process_manager.stop_selected_process()
        self.update_process_list()

    def stop_all_processes(self, *args):
        self.process_manager.stop_all_processes()
        self.update_process_list()

    def update_process_list(self, *args):
        process_names = [str(process) for process in self.process_manager.running_processes]
        self.window[self.process_list_key].update(process_names)

    def update_basic_info_frame(self):
        selected_process = self.process_manager.selected_process
        if not selected_process:
            return

        self.window[self.basic_info_tab_l0_key].update('Wp cnt: ' + str(selected_process.task_info['wp_cnt']))
        self.window[self.basic_info_tab_l1_key].update('Freq. pnts: ' + str(selected_process.task_info['freq_cnt']))
        self.window[self.basic_info_tab_l2_key].update('Variable cnt: ' + str(selected_process.task_info['opt_params']))

    def on_process_selection(self, ui_values):
        if ui_values[self.process_list_key]:
            self.process_manager.set_selected_process(ui_values[self.process_list_key][0])
        else:
            return
        self.update_basic_info_frame()

    def update_info_frame(self, output):
        selected_process = self.process_manager.selected_process
        if not selected_process:
            return
        self.window[self.info_tab_l0_key].update(selected_process.name)

        if output.process_name == selected_process.name:
            self.window[self.info_tab_l1_key].update('Iter cnt: ' + str(output.iter_cnt))
            if not (output.iter_cnt % 10):
                now = time.time()
                selected_process.iter_speed = 10 / (now - selected_process.previous_iteration_time)
                selected_process.previous_iteration_time = now
            self.window[self.info_tab_l2_key].update(f'{np.round(selected_process.iter_speed, 2)} iterations / s')
            self.window[self.info_tab_l3_key].update('Best min: ' + str(np.round(selected_process.best_result, 5)))

    def update_dbo_info_frame(self, output):
        self.window[self.dbo_task_info_tab_l0_key].update('Iter cnt: ' + str(output['iter_cnt']))
        self.window[self.dbo_task_info_tab_l1_key].update('F: ' + str(output['f']))

    def update_job_progress(self, output):
        task_cnt, total_tasks, best_f = output['task_cnt'], output['total_task_cnt'], output['best_f']

        cur_combination = output['cur_combination']
        self.window[self.dbo_current_combination_key].update(f'Current combination: {cur_combination}')

        self.window[self.dbo_job_info_tab_l0_key].update(f'Best F: {round(best_f, 5)}')

        self.window[self.dbo_progressbar_key].update(task_cnt, total_tasks)
        s = f'Task: {task_cnt}/{total_tasks}, ({round(100*task_cnt/total_tasks, 1)} %)'
        self.window[self.dbo_job_progress_text_field_key].update(s)

    @check_values
    def optimizer_test(self, ui_values):
        test_settings = ui_values.copy()
        test_settings[self.iterations_key] = 5
        make_optimizer(queue=self.queue, settings=test_settings)

    def clear_outputwindow(self, *args):
        self.window['_output_'].update('')

    # ---------------------------------------Result tab--------------------------------------------------------------- #
    def set_selected_result_settings(self, *args):
        if self.result_manager.result_selected:
            selected_result_settings_dict = self.result_manager.selected_result.result_settings
            self.set_window_element_values(selected_result_settings_dict)

    def update_folder_list(self, *args):
        folder_names = self.result_manager.dir_names()
        self.window[self.folder_list_key].update(folder_names)

    def update_result_list(self, ui_values):
        selected_folders = ui_values[self.folder_list_key]
        if selected_folders:
            result_folder_names = self.result_manager.result_names(selected_folders[0])
            self.window[self.result_list_key].update(result_folder_names)

    def on_folder_selection(self, ui_values):
        self.update_result_list(ui_values)

    def on_result_selection(self, ui_values):
        if ui_values[self.result_list_key]:
            self.result_manager.set_selected_result(ui_values)
            self.plotter.result = self.result_manager.selected_result
            self.update_result_frames(ui_values)

    def update_result_frames(self, ui_values):
        # two frames: Primary values and final stokes vector with parameters
        values = self.result_manager.result_info_frame_values()
        self.window[self.result_info_l0_key].update(values['f'])
        self.window[self.result_info_l1_key].update(values['angles'])
        self.window[self.result_info_l2_key].update(values['widths'])
        self.window[self.result_info_l3_key].update(values['stripes'])
        self.window[self.selected_result_total_width_key].update(values['total_width'])

        self.update_stokes_frame(ui_values)

    def update_stokes_frame(self, ui_values):
        selected_frequency_str = ui_values[self.selected_frequency_key]
        if selected_frequency_str and self.result_manager.result_selected:
            selected_frequency = np.float(selected_frequency_str) * THz
            s_final_str, j_final_str, str1, str2 = self.result_manager.get_final_state_info(selected_frequency)

            self.window[self.stokes_vector_key].update(s_final_str)
            self.window[self.jones_vector_key].update(j_final_str)
            self.window[self.stokes_parameters_key1].update(str1)
            self.window[self.stokes_parameters_key2].update(str2)

            # shows the closest frequency to the selected one
            actual_frequency = np.round(self.result_manager.selected_result.actual_selected_frequency * Hz_to_THz, 3)[0]
            self.window[self.actual_frequency_key].update(actual_frequency)

    def show_erf_settings(self, *args):
        if self.result_manager.result_selected:
            dict_str = self.result_manager.key_group_to_string('erf_settings')
            sg.PopupNoWait(dict_str, title='Erf. settings')
        else:
            sg.PopupNoWait('No result selected')

    def show_optimizer_settings(self, *args):
        if self.result_manager.result_selected:
            dict_str = self.result_manager.key_group_to_string('optimizer_settings')
            sg.PopupNoWait(dict_str, title='Optimizer settings')
        else:
            sg.PopupNoWait('No result selected')

    def show_saver_settings(self, *args):
        if self.result_manager.result_selected:
            dict_str = self.result_manager.key_group_to_string('saver_settings')
            sg.PopupNoWait(dict_str, title='Saver settings')
        else:
            sg.PopupNoWait('No result selected')

    def show_material_settings(self, *args):
        if self.result_manager.result_selected:
            dict_str = self.result_manager.key_group_to_string('material')
            sg.PopupNoWait(dict_str, title='Material settings')
        else:
            sg.PopupNoWait('No result selected')

    def plot_selected_result(self, ui_values):
        if self.result_manager.result_selected:
            if ui_values[self.plot_in_cst_key]:
                target_figure = 'CST'
            else:
                target_figure = 'Result plot'
            self.plotter.result_plot(target_figure=target_figure)

    def fix_old_settings(self, *args):
        self.settings_module.fix_old_settings()

    # ---------------------------------------Plot tab----------------------------------------------------------------- #
    def plot_birefringence(self, ui_values):
        if self.result_manager.result_selected:
            self.plotter.birefringence_plot(ui_values)

    def plot_refractive_indices(self, ui_values):
        if self.result_manager.result_selected:
            self.plotter.refractive_index_plot(ui_values)

    def plot_x_with_error(self, ui_values):
        if self.result_manager.result_selected:
            self.plotter.error_plot(ui_values)

    def plot_x_without_error(self, *args):
        if self.result_manager.result_selected:
            self.plotter.result_plot(target_figure='Error plot')

    def update_error_sliders(self, ui_values):
        self.window[self.angle_err_slider_key].update(range=(ui_values['min_angle_error_input'],
                                                             ui_values['max_angle_error_input']))
        self.window[self.width_err_slider_key].update(range=(ui_values['min_width_error_input'],
                                                             ui_values['max_width_error_input']))
        self.window[self.stripe_err_slider_key].update(range=(ui_values['min_stripe_error_input'],
                                                              ui_values['max_stripe_error_input']))

    def polar_plot(self, ui_values):
        if self.result_manager.result_selected:
            self.plotter.polar_plot(ui_values)

    # ---------------------------------------CST tab------------------------------------------------------------------ #
    def update_cst_folder_listbox(self, *args):
        cst_folders = self.result_manager.get_cst_folder_names()
        self.window[self.cst_folders_key].update(cst_folders)

    def on_cst_folder_list_selection(self, ui_values):
        selected_folder = ui_values[self.cst_folders_key][0]
        cst_file_names = self.result_manager.get_cst_result_file_names(selected_folder)
        self.window[self.cst_file_list_key].update(cst_file_names)

    def on_cst_file_list_selection(self, ui_values):
        if ui_values[self.cst_file_list_key]:
            self.result_manager.set_selected_cst_result(ui_values)
            self.plotter.cst_result = self.result_manager.selected_cst_result

    def plot_cst_results(self, ui_values):
        if self.plotter.cst_result:
            self.plotter.cst_plot(ui_values)
    # ---------------------------------------Single WP---------------------------------------------------------------- #

    @check_values
    def single_wp_plot_intensities(self, ui_values):
        new_single_wp = SingleWaveplate(ui_values)
        new_single_wp.single_wp_intensity_plot()

    @check_values
    def single_wp_plot_refractive_indices(self, ui_values):
        new_single_wp = SingleWaveplate(ui_values)
        new_single_wp.single_wp_refractive_indices_plot()

    @check_values
    def calculate_zeroth_order_widths(self, ui_values):
        freq_input = ui_values[self.zeroth_order_freq_input_key]
        if not freq_input:
            return
        freq_input = np.float(freq_input) * GHz
        new_single_wp = SingleWaveplate(ui_values)
        l2_width, l4_width = new_single_wp.calculate_zeroth_order_width(freq_input)
        self.window[self.zeroth_order_width_result_l2_input_key].update(str(np.round(l2_width * m_to_um, 2)))
        self.window[self.zeroth_order_width_result_l4_input_key].update(str(np.round(l4_width * m_to_um, 2)))

    def set_window_element_values(self, settings_dict):
        for key in settings_dict:
            try:
                value = settings_dict[key]
                if isinstance(value, type(list())):
                    value = str(value)
                self.window[key].update(value)
            except KeyError:
                continue

    def exit(self, *args):
        if last_ui_values:
            self.settings_module.save_settings(last_ui_values)
        self.queue_reader.stop()
        self.window.close()

    # do things
    def event_map(self, event, ui_values):
        if event.label == 'error_sliders':
            try:
                self.update_error_sliders(ui_values)
            except _tkinter.TclError:
                return
        if event.label == 'material':
            self.on_material_selection(ui_values)

        binds = {
            # tab1
            self.wp_cnt_key: self.on_wp_cnt_change,
            self.wp_type_key: self.on_wp_cnt_change,
            self.new_x0_key: self.new_initial_values,
            self.set_default_settings_button_key: self.load_default_settings,
            # tab2
            self.add_material_button_key: self.add_material_to_list,
            # tab3
            # tab4
            # tab5
            self.start_process_button_key: self.new_optimization_process,
            self.stop_process_button_key: self.stop_selected_process,
            self.stop_all_processes_button_key: self.stop_all_processes,
            self.process_list_key: self.on_process_selection,
            self.run_once_button_key: self.run_once,
            self.optimizer_test_button_key: self.optimizer_test,
            self.dbo_start_job_button_key: self.discrete_bruteforce_optimization,
            self.clear_output_button_key: self.clear_outputwindow,

            # tab6
            self.update_result_list_button_key: self.update_result_list,
            self.update_folder_list_button_key: self.update_folder_list,
            self.folder_list_key: self.on_folder_selection,
            self.result_list_key: self.on_result_selection,
            self.plot_selected_result_button_key: self.plot_selected_result,
            self.erf_settings_key: self.show_erf_settings,
            self.optimizer_settings_key: self.show_optimizer_settings,
            self.saver_settings_key: self.show_saver_settings,
            self.material_settings_key: self.show_material_settings,
            self.set_selected_result_settings_button_key: self.set_selected_result_settings,
            self.selected_frequency_key: self.update_stokes_frame,
            self.fix_old_settings_button_key: self.fix_old_settings,
            # tab7
            self.plot_error_button_key: self.plot_x_with_error,
            self.plot_birefringence_button_key: self.plot_birefringence,
            self.plot_refractive_indices_button_key: self.plot_refractive_indices,
            self.plot_without_error_key: self.plot_x_without_error,
            self.polar_plot_button_key: self.polar_plot,
            # tab8
            self.update_cst_list_button_key: self.update_cst_folder_listbox,
            self.plot_cst_selections_button_key: self.plot_cst_results,
            self.cst_folders_key: self.on_cst_folder_list_selection,
            self.cst_file_list_key: self.on_cst_file_list_selection,
            # tab9
            self.single_wp_intensity_plot_button_key: self.single_wp_plot_intensities,
            self.single_wp_refractive_indices_plot_button_key: self.single_wp_plot_refractive_indices,
            self.calculate_zeroth_order_width_button_key: self.calculate_zeroth_order_widths,
        }

        try:
            try:
                binds[event](ui_values)
            # all elements trigger event but some are just ui settings (no direct function)
            except KeyError:
                pass
        except Exception as e:
            error_popup(e)


if __name__ == '__main__':

    main_app = WaveplateApp()

    main_app.queue_reader.start()

    _, last_ui_values = fix_types(*main_app.window.read(timeout=0))
    while True:
        event, current_ui_values = fix_types(*main_app.window.read())
        # for saving after closing window. (read returns None when pressing 'x')
        if current_ui_values and any(current_ui_values.values()):
            last_ui_values = current_ui_values
        if event == sg.WIN_CLOSED:
            break
        main_app.event_map(event, current_ui_values)

    main_app.exit()

    s = "".join(map(chr, [78, 105, 110, 106, 97]))
    print(s)