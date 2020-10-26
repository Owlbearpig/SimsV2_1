import PySimpleGUI as sg
from modules.identifiers.dict_keys import DictKeys
from modules.utils.helpers import cast_to_ui


# dunno how to implement this lol (metaclasses??)
def new_default_values(kwargs):
    try:
        kwargs['enable_events']
    except KeyError:
        kwargs['enable_events'] = True
    return kwargs


class Checkbox(sg.Checkbox):
    def __init__(self, *args, **kwargs):
        kwargs = new_default_values(kwargs)
        super().__init__(*args, **kwargs)


class Slider(sg.Slider):
    def __init__(self, *args, **kwargs):
        kwargs = new_default_values(kwargs)
        super().__init__(*args, **kwargs)


class Input(sg.Input):
    def __init__(self, *args, **kwargs):
        kwargs = new_default_values(kwargs)
        super().__init__(*args, **kwargs)


class Drop(sg.Drop):
    def __init__(self, *args, **kwargs):
        kwargs = new_default_values(kwargs)
        super().__init__(*args, **kwargs)


class Listbox(sg.Listbox):
    def __init__(self, *args, **kwargs):
        kwargs = new_default_values(kwargs)
        super().__init__(*args, **kwargs)


class Tabs(DictKeys):
    def __init__(self, settings, material_manager):
        super().__init__()
        self.material_manager = material_manager
        self.material_list = [str(material) for material in self.material_manager.material_list]
        self.settings = settings

    def get_tab1_layout(self):
        tab1_desc = sg.Text('Error function settings and stuff')
        const_wp_dim_checkbox = Checkbox('Const wp dimensions',
                                         size=(20, 1),
                                         default=self.settings[self.const_wp_dim_key],
                                         key=self.const_wp_dim_key)
        equal_stripes_checkbox = Checkbox('Equal stripes',
                                          default=self.settings[self.equal_stripes_key],
                                          key=self.equal_stripes_key)
        log_of_res_checkbox = Checkbox('Log of res',
                                       size=(20, 1),
                                       default=self.settings[self.log_of_res_key],
                                       key=self.log_of_res_key)
        wp_cnt_slider = Slider(range=(1, 20),
                               orientation='h',
                               size=(34, 20),
                               default_value=self.settings[self.wp_cnt_key],
                               key=self.wp_cnt_key)
        const_angles_input = Input(default_text=cast_to_ui(self.settings[self.const_angles_key]),
                                   key=self.const_angles_key)
        const_widths_input = Input(default_text=cast_to_ui(self.settings[self.const_widths_key]),
                                   key=self.const_widths_key)
        width_pattern_input = Input(default_text=cast_to_ui(self.settings[self.width_pattern_key]),
                                    key=self.width_pattern_key)
        freq_limit_low_input = Input(default_text=self.settings[self.min_freq_key], size=(15, 1),
                                     key=self.min_freq_key)
        freq_limit_high_input = Input(default_text=self.settings[self.max_freq_key], size=(15, 1),
                                      key=self.max_freq_key)
        stripe_widths_input = Input(default_text=cast_to_ui(self.settings[self.initial_stripe_widths_key]),
                                    key=self.initial_stripe_widths_key)
        randomizer_seed_input = Input(default_text=self.settings[self.randomizer_seed_key],
                                      size=(10, 1),
                                      key=self.randomizer_seed_key)
        low_width_bound_input = Input(default_text=self.settings[self.min_width_key],
                                      size=(10, 1),
                                      key=self.min_width_key)
        high_width_bound_input = Input(default_text=self.settings[self.max_width_key],
                                       size=(10, 1),
                                       key=self.max_width_key)
        low_angles_bound_input = Input(default_text=self.settings[self.min_angle_key],
                                       size=(10, 1),
                                       key=self.min_angle_key)
        high_angles_bound_input = Input(default_text=self.settings[self.max_angle_key],
                                        size=(10, 1), key=self.max_angle_key)
        low_stripes_bound_input = Input(default_text=self.settings[self.min_stripe_width_key],
                                        size=(10, 1),
                                        key=self.min_stripe_width_key)
        high_stripes_bound_input = Input(default_text=self.settings[self.max_stripe_width_key],
                                         size=(10, 1),
                                         key=self.max_stripe_width_key)

        low_anisotropy_input = Input(default_text=self.settings[self.anisotropy_s_key],
                                     size=(10, 1),
                                     key=self.anisotropy_s_key)
        high_anisotropy_input = Input(default_text=self.settings[self.anisotropy_p_key],
                                      size=(10, 1),
                                      key=self.anisotropy_p_key)
        wp_type_dropdown = Drop(values=['λ/2', 'λ/4', 'mixed'],
                                default_value=self.settings[self.wp_type_key],
                                auto_size_text=True,
                                key=self.wp_type_key)
        weak_absorption_checkbox = Checkbox('Enable weak absorption',
                                            default=self.settings[self.weak_absorption_checkbox_key],
                                            key=self.weak_absorption_checkbox_key,
                                            tooltip='Enable to use anisotropic part of absorption only')
        x0_angles_input = Input(default_text=cast_to_ui(self.settings[self.x0_angle_input_key]),
                                size=(70, 1),
                                key=self.x0_angle_input_key)
        x0_widths_input = Input(default_text=cast_to_ui(self.settings[self.x0_widths_input_key]),
                                size=(70, 1),
                                key=self.x0_widths_input_key)
        x0_stripes_input = Input(default_text=cast_to_ui(self.settings[self.x0_stripes_input_key]),
                                 size=(70, 1),
                                 key=self.x0_stripes_input_key)
        new_x0_button = sg.Button('New x0',
                                  key=self.new_x0_key)
        set_default_settings_button = sg.Button('Load default settings',
                                                key=self.set_default_settings_button_key)
        frequency_multiplier_input = Input(default_text=self.settings[self.frequency_resolution_multiplier_key],
                                           size=(5, 1),
                                           key=self.frequency_resolution_multiplier_key)
        x_slicing_input = Input(cast_to_ui(self.settings[self.x_slicing_key]),
                                key=self.x_slicing_key,
                                size=(20, 1),
                                disabled=True)
        calculation_method = Drop(['Jones', 'Stokes'],
                                  default_value=self.settings[self.calculation_method_key],
                                  key=self.calculation_method_key)

        tab1_layout = [
            [sg.Frame('Number of waveplates', [[wp_cnt_slider]]),
             sg.Frame('Frequency range',
                      [[sg.Text('Lower freq. (THz)', size=(15, 1)),
                        sg.Text('Upper freq. (THz)', size=(15, 1))],
                       [freq_limit_low_input,
                        freq_limit_high_input]])
             ],
            [const_wp_dim_checkbox, equal_stripes_checkbox,
             frequency_multiplier_input, sg.Text('Freq. res. multiplier')],
            [sg.Text('Const angles (rad)', size=(20, 1)), const_angles_input],
            [sg.Text('Const widths (μm)', size=(20, 1)), const_widths_input],
            [sg.Text('Width pattern', size=(20, 1)), width_pattern_input],
            [sg.Text('Stripe widths', size=(20, 1)), stripe_widths_input],
            [set_default_settings_button],
            [sg.Text('Method'), calculation_method],
            [sg.Frame('Bounds',
                      [[sg.Text('Angles (rad)', size=(20, 1)),
                        low_angles_bound_input, high_angles_bound_input],
                       [sg.Text('Widths (μm)', size=(20, 1)),
                        low_width_bound_input, high_width_bound_input],
                       [sg.Text('Stripes (μm)', size=(20, 1)),
                        low_stripes_bound_input, high_stripes_bound_input]]),
             sg.Frame('Stuff',
                      [[sg.Text('Randomizer seed', size=(15, 1)), randomizer_seed_input,
                        sg.Text('Anisotropy', size=(15, 1)),
                        sg.Text('k_s'), low_anisotropy_input,
                        sg.Text('k_p'), high_anisotropy_input],
                       [sg.Text('Wp type', size=(15, 1)), wp_type_dropdown,
                        log_of_res_checkbox],
                       [sg.Text('x - slices:'), x_slicing_input, weak_absorption_checkbox]]),
             sg.Frame('Initial values',
                      [[sg.Text('angles (rad)')],
                       [x0_angles_input],
                       [sg.Text('widths (μm)')],
                       [x0_widths_input],
                       [sg.Text('stripes (μm)')],
                       [x0_stripes_input],
                       [new_x0_button]])
             ]
        ]
        return tab1_layout

    def get_tab2_layout(self):
        selected_material_data_path_input = Input(
            self.material_manager.get_material(self.settings[self.material_drop_down_key]).path,
            key=self.selected_material_data_path_key,
            size=(100, 1),
            disabled=True)
        fast_material_data_path_input = Input(
            self.material_manager.get_material(self.settings[self.fast_material_dropdown_key]).path,
            key=self.selected_fast_material_data_path_key,
            size=(100, 1),
            disabled=True)
        slow_material_data_path_input = Input(
            self.material_manager.get_material(self.settings[self.slow_material_dropdown_key]).path,
            key=self.selected_slow_material_data_path_key,
            size=(100, 1),
            disabled=True)
        materials_dropdown = Drop(self.material_list,
                                  default_value=self.settings[self.material_drop_down_key],
                                  auto_size_text=True,
                                  key=self.material_drop_down_key)
        new_material_name_input = Input(default_text=self.settings[self.new_material_name_key],
                                        size=(10, 1),
                                        key=self.new_material_name_key)
        new_material_path_input = Input(default_text=self.settings[self.new_material_path_key],
                                        size=(50, 1),
                                        key=self.new_material_path_key)
        add_material_button = sg.Button(button_text='Add material',
                                        key=self.add_material_button_key)
        set_ri_real_part_checkbox = Checkbox('Overwrite real part',
                                             default=self.settings[self.set_ri_real_part_checkbox_key],
                                             key=self.set_ri_real_part_checkbox_key)
        set_ri_img_part_checkbox = Checkbox('Overwrite imaginary part',
                                            default=self.settings[self.set_ri_img_part_checkbox_key],
                                            key=self.set_ri_img_part_checkbox_key)
        const_refr_index_input_n_s = Input(self.settings[self.const_refr_index_input_n_s_key],
                                           size=(5, 1),
                                           key=self.const_refr_index_input_n_s_key)
        const_refr_index_input_k_s = Input(self.settings[self.const_refr_index_input_k_s_key],
                                           size=(5, 1),
                                           key=self.const_refr_index_input_k_s_key)
        const_refr_index_input_n_p = Input(self.settings[self.const_refr_index_input_n_p_key],
                                           size=(5, 1),
                                           key=self.const_refr_index_input_n_p_key)
        const_refr_index_input_k_p = Input(self.settings[self.const_refr_index_input_k_p_key],
                                           size=(5, 1),
                                           key=self.const_refr_index_input_k_p_key)
        birefringence_type_dropdown = Drop(['Natural', 'Form'],
                                           default_value=self.settings[self.birefringence_type_dropdown_key],
                                           auto_size_text=True,
                                           key=self.birefringence_type_dropdown_key)
        fast_material_dropdown = Drop(self.material_list,
                                      default_value=self.settings[self.fast_material_dropdown_key],
                                      auto_size_text=True,
                                      key=self.fast_material_dropdown_key)
        slow_material_dropdown = Drop(self.material_list,
                                      default_value=self.settings[self.slow_material_dropdown_key],
                                      auto_size_text=True,
                                      key=self.slow_material_dropdown_key)

        tab2_layout = [
            [sg.Text('Birefringence type:'), birefringence_type_dropdown],
            [sg.Frame('Overwrite RI',
                      [[set_ri_real_part_checkbox, set_ri_img_part_checkbox],
                       [sg.Text('n_s'), const_refr_index_input_n_s,
                        sg.Text('k_s'), const_refr_index_input_k_s,
                        sg.Text('n_p'), const_refr_index_input_n_p,
                        sg.Text('k_p'), const_refr_index_input_k_p
                        ]])],
            [sg.Frame('Add material',
                      [[sg.Text('Name'), new_material_name_input,
                        sg.Text('Path'), new_material_path_input,
                        sg.FileBrowse(), add_material_button]])
             ],
            [sg.Frame('Form birefringence',
                      [[sg.Text('Form birefringence material:'), materials_dropdown],
                       [sg.Text('Data path'), selected_material_data_path_input],
                       ])],
            [sg.Frame('Natural birefringence',
                      [[sg.Text('Fast material:'), fast_material_dropdown],
                       [sg.Text('Fast material path'), fast_material_data_path_input],
                       [sg.Text('Slow material:'), slow_material_dropdown],
                       [sg.Text('Slow material path'), slow_material_data_path_input]]),
             ],
        ]

        return tab2_layout

    def get_tab3_layout(self):
        disable_callback_checkbox = Checkbox('Disable callback',
                                             size=(20, 1),
                                             default=self.settings[self.disable_callback_key],
                                             key=self.disable_callback_key)
        periodic_restart_checkbox = Checkbox('Periodic restart',
                                             size=(20, 1),
                                             default=self.settings[self.periodic_restart_key],
                                             key=self.periodic_restart_key)
        iteration_cnt_input = Input(default_text=self.settings[self.iterations_key],
                                    size=(10, 1),
                                    key=self.iterations_key)
        temperature_input = Input(default_text=self.settings[self.temperature_key],
                                  size=(10, 1),
                                  key=self.temperature_key)
        local_opt_tol_input = Input(default_text=self.settings[self.local_opt_tol_key],
                                    size=(10, 1),
                                    key=self.local_opt_tol_key)
        angle_step_input = Input(default_text=self.settings[self.angle_step_key],
                                 size=(10, 1),
                                 key=self.angle_step_key)
        width_step_input = Input(default_text=self.settings[self.width_step_key],
                                 size=(10, 1),
                                 key=self.width_step_key)
        stripe_step_input = Input(default_text=self.settings[self.stripe_width_step_key],
                                  size=(10, 1),
                                  key=self.stripe_width_step_key)
        print_precision_input = Input(default_text=self.settings[self.print_precision_key],
                                      size=(10, 1),
                                      key=self.print_precision_key)
        print_interval_input = Input(default_text=self.settings[self.print_interval_key],
                                     size=(10, 1),
                                     key=self.print_interval_key)

        tab3_layout = [[sg.Text('Optimization things')],
                       [periodic_restart_checkbox, disable_callback_checkbox,
                        sg.Text('Print interval', size=(15, 1)), print_interval_input,
                        sg.Text('Print precision', size=(15, 1)), print_precision_input
                        ],
                       [sg.Frame('Basin hopping',
                                 [[sg.Text('Iterations', size=(15, 1)), iteration_cnt_input,
                                   sg.Text('Temperature', size=(15, 1)), temperature_input],
                                  [sg.Text('Local opt. tol', size=(15, 1)), local_opt_tol_input]]),
                        sg.Frame('Step settings',
                                 [[sg.Text('Angle step (rad)', size=(15, 1)), angle_step_input,
                                   sg.Text('Width step (μm)', size=(15, 1)), width_step_input],
                                  [sg.Text('Stripe step (μm)', size=(15, 1)), stripe_step_input]]),
                        ]
                       ]

        return tab3_layout

    def get_tab4_layout(self):
        save_iterations_checkbox = Checkbox('Save iterations',
                                            size=(20, 1),
                                            key=self.save_iterations_key,
                                            default=self.settings[self.save_iterations_key])
        save_all_results_checkbox = Checkbox('Save all results',
                                             size=(20, 1),
                                             default=self.settings[self.save_all_results_key],
                                             key=self.save_all_results_key)
        save_settings_checkbox = Checkbox('Save settings',
                                          size=(20, 1),
                                          default=self.settings[self.save_settings_key],
                                          key=self.save_settings_key)
        save_name_input = Input(default_text=self.settings[self.save_name_key],
                                key=self.save_name_key)
        save_folder_name_input = Input(default_text=self.settings[self.save_folder_name_key],
                                       key=self.save_folder_name_key)

        tab4_layout = [
            [save_all_results_checkbox, save_settings_checkbox, save_iterations_checkbox],
            [sg.Text('Save name', size=(15, 1)), save_name_input],
            [sg.Text('Folder name', size=(15, 1)), save_folder_name_input]
        ]

        return tab4_layout

    def get_tab5_layout(self):
        output = sg.Output(size=(90, 12),
                           text_color='Green')
        start_process_button = sg.Button(button_text='Start process',
                                         key=self.start_process_button_key)
        stop_process_button = sg.Button(button_text='Stop process',
                                        key=self.stop_process_button_key)
        stop_all_processes_button = sg.Button(button_text='Stop all processes',
                                              key=self.stop_all_processes_button_key)
        process_list = Listbox(values=(),
                               size=(20, 13),
                               key=self.process_list_key)

        basic_info_tab_l0 = sg.Text('', size=(20, 1), justification='left', key=self.basic_info_tab_l0_key)
        basic_info_tab_l1 = sg.Text('', size=(15, 1), justification='left', key=self.basic_info_tab_l1_key)
        basic_info_tab_l2 = sg.Text('', size=(15, 1), justification='left', key=self.basic_info_tab_l2_key)

        info_tab_l0 = sg.Text('', size=(20, 1), justification='left', key=self.info_tab_l0_key)
        info_tab_l1 = sg.Text('', size=(15, 1), justification='left', key=self.info_tab_l1_key)
        info_tab_l2 = sg.Text('', size=(15, 1), justification='left', key=self.info_tab_l2_key)
        info_tab_l3 = sg.Text('', size=(15, 1), justification='left', key=self.info_tab_l3_key)

        angles_input = Input(default_text=cast_to_ui(self.settings[self.angle_input_key]),
                             size=(30, 1),
                             key=self.angle_input_key,
                             enable_events=False)
        widths_input = Input(default_text=cast_to_ui(self.settings[self.widths_input_key]),
                             size=(30, 1),
                             key=self.widths_input_key,
                             enable_events=False)
        stripes_input = Input(default_text=cast_to_ui(self.settings[self.stripes_input_key]),
                              size=(30, 1),
                              key=self.stripes_input_key,
                              enable_events=False)

        run_once_button = sg.Button('Run once',
                                    key=self.run_once_button_key)

        run_once_label_input = Input(self.settings[self.run_once_label_input_key],
                                     key=self.run_once_label_input_key,
                                     size=(20, 1))

        optimizer_test_button = sg.Button('Test',
                                          key=self.optimizer_test_button_key)

        dbo_widths_input = Input(default_text=cast_to_ui(self.settings[self.dbo_widths_input_key]),
                                 key=self.dbo_widths_input_key,
                                 size=(30, 1))
        dbo_task_info_tab_l0 = sg.Text('', size=(15, 1), justification='left', key=self.dbo_task_info_tab_l0_key)
        dbo_task_info_tab_l1 = sg.Text('', size=(15, 1), justification='left', key=self.dbo_task_info_tab_l1_key)
        dbo_save_name_input = Input(default_text=cast_to_ui(self.settings[self.dbo_save_name_input_key]),
                                    key=self.dbo_save_name_input_key,
                                    size=(30, 1))
        dbo_start_job_button = sg.Button('Start',
                                         key=self.dbo_start_job_button_key)
        dbo_job_info_tab_l0 =  sg.Text('', size=(15, 1), justification='left', key=self.dbo_job_info_tab_l0_key)
        dbo_job_info_tab_l1 = sg.Text('', size=(15, 1), justification='left', key=self.dbo_job_info_tab_l1_key)


        tab5_layout = [[sg.Frame('Output',
                                 [[output],
                                  [start_process_button, stop_process_button,
                                   stop_all_processes_button, optimizer_test_button]]),
                        sg.Frame('Basic info',
                                 [[basic_info_tab_l0],
                                  [basic_info_tab_l1],
                                  [basic_info_tab_l2]
                                  ]),
                        sg.Frame('Info',
                                 [[info_tab_l0],
                                  [info_tab_l1],
                                  [info_tab_l2],
                                  [info_tab_l3],
                                  ]),
                        sg.Frame('Running processes',
                                 [[process_list]]),
                        sg.Frame('Run once',
                                 [[sg.Text('Angles (deg)')],
                                  [angles_input],
                                  [sg.Text('Widths (μm)')],
                                  [widths_input],
                                  [sg.Text('Stripes (μm)')],
                                  [stripes_input],
                                  [sg.Text('Label:'), run_once_label_input],
                                  [run_once_button]])
                        ],
                       [sg.Frame('DBO',
                        [[sg.Text('Widths:', size=(15, 1)), dbo_widths_input],
                         [sg.Text('Save name:', size=(15, 1)), dbo_save_name_input],
                         [sg.Frame('Task info', [[dbo_task_info_tab_l0], [dbo_task_info_tab_l1]]),
                          sg.Frame('Job info', [[dbo_job_info_tab_l0], [dbo_job_info_tab_l1]])],
                         [dbo_start_job_button],
                         [sg.ProgressBar(1000, orientation='h', size=(20, 20), key=self.dbo_progressbar_key)]
                        ])
                        ],
                       ]

        return tab5_layout

    def get_tab6_layout(self):
        result_list = Listbox(values=(),
                              size=(40, 15),
                              key=self.result_list_key)
        update_result_list_button = sg.Button(button_text='Update results',
                                              key=self.update_result_list_button_key)
        folder_list = Listbox(values=(),
                              size=(40, 15),
                              key=self.folder_list_key)
        update_folder_list_button = sg.Button('Update folders',
                                              key=self.update_folder_list_button_key)
        plot_selected_result_button = sg.Button('Plot result',
                                                key=self.plot_selected_result_button_key)
        plot_in_cst_window_checkbox = Checkbox('Add to CST plot',
                                               default=False,
                                               key=self.plot_in_cst_key)

        result_info_l0 = Input('', size=(50, 1), justification='left', key=self.result_info_l0_key, disabled=True)
        result_info_l1 = Input('', size=(50, 1), justification='left', key=self.result_info_l1_key, disabled=True)
        result_info_l2 = Input('', size=(50, 1), justification='left', key=self.result_info_l2_key, disabled=True)
        result_info_l3 = Input('', size=(50, 1), justification='left', key=self.result_info_l3_key, disabled=True)

        erf_settings_info_button = sg.Button('Erf. settings',
                                             key=self.erf_settings_key)
        optimizer_settings_info_button = sg.Button('Optimizer settings',
                                                   key=self.optimizer_settings_key)
        saver_settings_info_button = sg.Button('Saver settings',
                                               key=self.saver_settings_key)
        selected_frequency_input = Input(default_text=self.settings[self.selected_frequency_key],
                                         size=(5, 1),
                                         key=self.selected_frequency_key)
        set_selected_result_settings_button = sg.Button(button_text='Set selected result settings',
                                                        key=self.set_selected_result_settings_button_key)
        fix_old_settings_button = sg.Button('Fix old settings',
                                            key=self.fix_old_settings_button_key)

        tab6_layout = [
            [sg.Frame('Folders',
                      [[folder_list]]),
             sg.Frame('Saved results',
                      [[result_list]]),
             sg.Frame('Result values',
                      [[result_info_l0],
                       [sg.Text('Angles (deg):', size=(30, 1))],
                       [result_info_l1],
                       [sg.Text('Widths (μm):', size=(30, 1))],
                       [result_info_l2],
                       [sg.Text('Stripe widths (μm):', size=(30, 1))],
                       [result_info_l3],
                       [erf_settings_info_button, optimizer_settings_info_button,
                        saver_settings_info_button]]),
             sg.Frame('Stokes parameters',
                      [[sg.Text('Frequency'), selected_frequency_input,
                        sg.Text(f'({selected_frequency_input.DefaultText} THz)',
                                key=self.actual_frequency_key, size=(10, 1))],
                       [sg.Text('Stokes vector:'),
                        sg.Text(key=self.stokes_vector_key, size=(40, 1))],
                       [sg.Text('Jones vector:'),
                        sg.Text(key=self.jones_vector_key, size=(40, 1))],
                       [sg.Text(' ', size=(40, 1))],
                       [sg.Text(key=self.stokes_parameters_key1, size=(20, 20)),
                        sg.Text(key=self.stokes_parameters_key2, size=(20, 20))],
                       ])
             ],
            [update_folder_list_button, update_result_list_button],
            [plot_selected_result_button, plot_in_cst_window_checkbox],
            [set_selected_result_settings_button, fix_old_settings_button]
        ]

        return tab6_layout

    def get_tab7_layout(self):
        plot_result_with_error = sg.Button('Plot w/ error',
                                           key=self.plot_error_button_key)
        plot_original_result = sg.Button('Plot w/o error',
                                         key=self.plot_without_error_key)
        difference_plot_checkbox = Checkbox('Plot x-y difference',
                                            default=False,
                                            key=self.difference_plot_key)
        polarization_components_plot_checkbox = Checkbox('Plot polarization components',
                                                         default=False,
                                                         key=self.polarization_plot_key)
        angle_err_checkbox = Checkbox('Add angle(deg) error',
                                      default=False,
                                      key=self.angle_err_checkbox_key)
        width_err_checkbox = Checkbox('Add width(μm) error',
                                      default=False,
                                      key=self.width_err_checkbox_key)
        stripe_err_checkbox = Checkbox('Add stripe error(μm)',
                                       default=False,
                                       key=self.stripe_err_checkbox_key)
        angle_err_slider = Slider(range=(self.settings[self.min_angle_error_input_key],
                                         self.settings[self.max_angle_error_input_key]),
                                  orientation='h',
                                  size=(20, 20),
                                  key=self.angle_err_slider_key,
                                  pad=((0, 0), (0, 20)),
                                  resolution=0.001)
        width_err_slider = Slider(range=(self.settings[self.min_width_error_input_key],
                                         self.settings[self.max_width_error_input_key]),
                                  orientation='h',
                                  size=(20, 20),
                                  key=self.width_err_slider_key,
                                  pad=((0, 0), (0, 20)),
                                  resolution=0.001)
        stripe_err_slider = Slider(range=(self.settings[self.min_stripe_error_input_key],
                                          self.settings[self.max_stripe_error_input_key]),
                                   orientation='h',
                                   size=(20, 20),
                                   key=self.stripe_err_slider_key,
                                   pad=((0, 0), (0, 20)),
                                   resolution=0.001)
        min_angle_error_input = Input(default_text=self.settings[self.min_angle_error_input_key],
                                      size=(5, 1),
                                      key=self.min_angle_error_input_key)
        max_angle_error_input = Input(default_text=self.settings[self.max_angle_error_input_key],
                                      size=(5, 1),
                                      key=self.max_angle_error_input_key)
        min_width_error_input = Input(default_text=self.settings[self.min_width_error_input_key],
                                      size=(5, 1),
                                      key=self.min_width_error_input_key)
        max_width_error_input = Input(default_text=self.settings[self.max_width_error_input_key],
                                      size=(5, 1),
                                      key=self.max_width_error_input_key)
        min_stripe_error_input = Input(default_text=self.settings[self.min_stripe_error_input_key],
                                       size=(5, 1),
                                       key=self.min_stripe_error_input_key)
        max_stripe_error_input = Input(default_text=self.settings[self.max_stripe_error_input_key],
                                       size=(5, 1),
                                       key=self.max_stripe_error_input_key)
        plotted_error_angles = sg.Text(key=self.plotted_error_angles_key,
                                       size=(40, 1))
        plotted_error_widths = sg.Text(key=self.plotted_error_widths_key,
                                       size=(40, 1))
        plotted_error_stripes = sg.Text(key=self.plotted_error_stripes_key,
                                        size=(40, 1))
        overwrite_result_checkbox = Checkbox('Replace stripes',
                                             key=self.overwrite_stripes_key,
                                             default=self.settings[self.overwrite_stripes_key])
        stripe0_manual_input = Input(default_text=self.settings[self.stripe0_input_key],
                                     size=(5, 1),
                                     key=self.stripe0_input_key)
        stripe1_manual_input = Input(default_text=self.settings[self.stripe1_input_key],
                                     size=(5, 1),
                                     key=self.stripe1_input_key)

        plot_bf_real_part_checkbox = Checkbox('Plot real bf',
                                              key=self.plot_bf_real_key,
                                              default=self.settings[self.plot_bf_real_key])
        plot_bf_img_part_checkbox = Checkbox('Plot img bf',
                                             key=self.plot_bf_imag_key,
                                             default=self.settings[self.plot_bf_imag_key])

        plot_np_checkbox = Checkbox('Plot np',
                                    key=self.plot_np_key,
                                    default=self.settings[self.plot_np_key],
                                    )
        plot_ns_checkbox = Checkbox('Plot ns',
                                    key=self.plot_ns_key,
                                    default=self.settings[self.plot_ns_key],
                                    )
        plot_kp_checkbox = Checkbox('Plot kp',
                                    key=self.plot_kp_key,
                                    default=self.settings[self.plot_kp_key],
                                    )
        plot_ks_checkbox = Checkbox('Plot ks',
                                    key=self.plot_ks_key,
                                    default=self.settings[self.plot_ks_key],
                                    )

        plot_birefringence_button = sg.Button('Plot birefringence',
                                              key=self.plot_birefringence_button_key)
        plot_refractive_indices_button = sg.Button('Plot refractive indices',
                                                   key=self.plot_refractive_indices_button_key)

        tab7_layout = [[plot_original_result, plot_result_with_error,
                        difference_plot_checkbox, polarization_components_plot_checkbox
                        ],
                       [angle_err_checkbox, min_angle_error_input, angle_err_slider,
                        max_angle_error_input, plotted_error_angles],
                       [width_err_checkbox, min_width_error_input, width_err_slider,
                        max_width_error_input, plotted_error_widths],
                       [stripe_err_checkbox, min_stripe_error_input, stripe_err_slider,
                        max_stripe_error_input, plotted_error_stripes],
                       [overwrite_result_checkbox],
                       [sg.Text('Material 0:'), stripe0_manual_input, sg.Text('μm')],
                       [sg.Text('Material 1 (air):'), stripe1_manual_input, sg.Text('μm')],
                       [plot_bf_real_part_checkbox, plot_bf_img_part_checkbox,
                        plot_birefringence_button,
                        plot_np_checkbox, plot_ns_checkbox,
                        plot_kp_checkbox, plot_ks_checkbox,
                        plot_refractive_indices_button],
                       ]

        return tab7_layout

    def get_tab8_layout(self):
        update_cst_file_list_button = sg.Button('Update CST results',
                                                key=self.update_cst_list_button_key)
        plot_selected_cst_results = sg.Button(button_text='Plot selections',
                                              key=self.plot_cst_selections_button_key)
        cst_folder_list_listbox = Listbox(values=(),
                                          size=(45, 15),
                                          key=self.cst_folders_key)
        cst_file_list_listbox = Listbox(values=(),
                                        size=(45, 15),
                                        key=self.cst_file_list_key,
                                        select_mode='multiple')
        plot_x_polarizer_only_checkbox = Checkbox('plot x pol.',
                                                  key=self.cst_plot_x_key,
                                                  default=self.settings[self.cst_plot_x_key])
        plot_y_polarizer_only_checkbox = Checkbox('plot y pol.',
                                                  key=self.cst_plot_y_key,
                                                  default=self.settings[self.cst_plot_y_key])

        tab8_layout = [[update_cst_file_list_button, plot_selected_cst_results,
                        plot_x_polarizer_only_checkbox, plot_y_polarizer_only_checkbox],
                       [cst_folder_list_listbox, cst_file_list_listbox]
                       ]

        return tab8_layout

    def get_tab9_layout(self):
        tab9_desc = sg.Text('Single waveplate calculations')
        single_wp_angle_input = Input(default_text=self.settings[self.single_wp_angle_input_key],
                                      size=(10, 1),
                                      key=self.single_wp_angle_input_key)
        single_wp_width_input = Input(default_text=self.settings[self.single_wp_width_input_key],
                                      size=(10, 1),
                                      key=self.single_wp_width_input_key)
        single_wp_stripe1_width_input = Input(default_text=self.settings[self.single_wp_stripe1_width_input_key],
                                              size=(10, 1),
                                              key=self.single_wp_stripe1_width_input_key,
                                              )
        single_wp_stripe2_width_input = Input(default_text=self.settings[self.single_wp_stripe2_width_input_key],
                                              size=(10, 1),
                                              key=self.single_wp_stripe2_width_input_key,
                                              )
        single_wp_label_input = Input(default_text=self.settings[self.single_wp_label_input_key],
                                      size=(10, 1),
                                      key=self.single_wp_label_input_key)
        single_wp_intensity_plot_button = sg.Button('Intensity plot',
                                                    key=self.single_wp_intensity_plot_button_key)
        single_wp_refractive_indices_plot_button = sg.Button('Refractive indices',
                                                             key=self.single_wp_refractive_indices_plot_button_key)
        single_wp_plot_ns_checkbox = Checkbox('n_s',
                                              key=self.single_wp_plot_ns_checkbox_key,
                                              default=self.settings[self.single_wp_plot_ns_checkbox_key],
                                              )
        single_wp_plot_np_checkbox = Checkbox('n_p',
                                              key=self.single_wp_plot_np_checkbox_key,
                                              default=self.settings[self.single_wp_plot_np_checkbox_key],
                                              )
        single_wp_plot_ks_checkbox = Checkbox('k_s',
                                              key=self.single_wp_plot_ks_checkbox_key,
                                              default=self.settings[self.single_wp_plot_ks_checkbox_key],
                                              )
        single_wp_plot_kp_checkbox = Checkbox('k_p',
                                              key=self.single_wp_plot_kp_checkbox_key,
                                              default=self.settings[self.single_wp_plot_kp_checkbox_key],
                                              )
        zeroth_order_freq_input = Input(
            default_text=self.settings[self.zeroth_order_freq_input_key],
            key=self.zeroth_order_freq_input_key, size=(10, 1)
        )

        calculate_zeroth_order_width_button = sg.Button('Calculate width',
                                                        key=self.calculate_zeroth_order_width_button_key)
        zeroth_order_width_result_l2_input = Input(key=self.zeroth_order_width_result_l2_input_key,
                                                   disabled=True,
                                                   size=(10, 1))
        zeroth_order_width_result_l4_input = Input(key=self.zeroth_order_width_result_l4_input_key,
                                                   disabled=True,
                                                   size=(10, 1))

        tab9_layout = [
            [tab9_desc],
            [sg.Frame('Input',
                      [[sg.Text('Angle (deg)')],
                       [single_wp_angle_input],
                       [sg.Text('Width (μm)')],
                       [single_wp_width_input],
                       [sg.Text('Stripe mat. 1 (μm)')],
                       [sg.Text('(Usually material)')],
                       [single_wp_stripe1_width_input],
                       [sg.Text('Stripes mat. 2 (μm)')],
                       [sg.Text('(Usually air gap)')],
                       [single_wp_stripe2_width_input],
                       [sg.Text('Label')],
                       [single_wp_label_input]]
                      ),
             ],
            [single_wp_intensity_plot_button, single_wp_refractive_indices_plot_button],
            [sg.Text('Show:'), single_wp_plot_ns_checkbox, single_wp_plot_np_checkbox,
             single_wp_plot_ks_checkbox, single_wp_plot_kp_checkbox],
            [sg.Frame('Calculate zero order width',
                      [[sg.Text('Zeroth order frequency (GHz): '), zeroth_order_freq_input],
                       [calculate_zeroth_order_width_button],
                       [sg.Text('λ/2 width (μm)'), zeroth_order_width_result_l2_input,
                        sg.Text('λ/4 width (μm)'), zeroth_order_width_result_l4_input]
                       ]
                      ),
             ],
        ]

        return tab9_layout
