from modules.identifiers.custom_key import DictKey


class DictKeys:
    def __init__(self):
        # erf settings keys
        self.wp_cnt_key = DictKey('wp_cnt', label='erf_settings')
        self.const_wp_dim_key = DictKey('const_wp_dim', label='erf_settings')
        self.equal_stripes_key = DictKey('equal_stripes', label='erf_settings')
        self.const_angles_key = DictKey('const_angles', label='erf_settings')
        self.const_widths_key = DictKey('const_widths', label='erf_settings')
        self.width_pattern_key = DictKey('width_pattern', label='erf_settings')
        self.initial_stripe_widths_key = DictKey('initial_stripe_widths', label='erf_settings')
        self.frequency_resolution_multiplier_key = DictKey('frequency_resolution_multiplier', label='erf_settings')
        self.min_freq_key = DictKey('min_freq', label='erf_settings')
        self.max_freq_key = DictKey('max_freq', label='erf_settings')
        self.randomizer_seed_key = DictKey('randomizer_seed', label='erf_settings')
        self.log_of_res_key = DictKey('log_of_res', label='erf_settings')
        self.min_width_key = DictKey('min_width', label='erf_settings')
        self.max_width_key = DictKey('max_width', label='erf_settings')
        self.min_angle_key = DictKey('min_angle', label='erf_settings')
        self.max_angle_key = DictKey('max_angle', label='erf_settings')
        self.min_stripe_width_key = DictKey('min_stripe_width', label='erf_settings')
        self.max_stripe_width_key = DictKey('max_stripe_width', label='erf_settings')
        self.anisotropy_p_key = DictKey('anisotropy_p', label='erf_settings')
        self.anisotropy_s_key = DictKey('anisotropy_s', label='erf_settings')
        self.wp_type_key = DictKey('wp_type', label='erf_settings')
        self.x_slicing_key = DictKey('x_slicing', label='erf_settings')
        self.calculation_method_key = DictKey('calculation_method', label='erf_settings')
        self.const_refr_index_input_n_s_key = DictKey('const_refr_index_input_n_s', label='erf_settings')
        self.const_refr_index_input_k_s_key = DictKey('const_refr_index_input_k_s', label='erf_settings')
        self.const_refr_index_input_n_p_key = DictKey('const_refr_index_input_n_p', label='erf_settings')
        self.const_refr_index_input_k_p_key = DictKey('const_refr_index_input_k_p', label='erf_settings')
        self.test_key = DictKey('test')

        # optimizer settings keys
        self.iterations_key = DictKey('iterations', label='optimizer_settings')
        self.angle_step_key = DictKey('angle_step', label='optimizer_settings')
        self.width_step_key = DictKey('width_step', label='optimizer_settings')
        self.stripe_width_step_key = DictKey('stripe_width_step', label='optimizer_settings')
        self.temperature_key = DictKey('temperature', label='optimizer_settings')
        self.local_opt_tol_key = DictKey('local_opt_tol', label='optimizer_settings')
        self.print_precision_key = DictKey('print_precision', label='optimizer_settings')
        self.print_interval_key = DictKey('print_interval', label='optimizer_settings')
        self.periodic_restart_key = DictKey('periodic_restart', label='optimizer_settings')
        self.disable_callback_key = DictKey('disable_callback', label='optimizer_settings')
        self.process_name_key = DictKey('process_name', label='optimizer_settings')

        # saver settings keys
        self.save_iterations_key = DictKey('save_iterations', label='saver_settings')
        self.save_all_results_key = DictKey('save_all_results', label='saver_settings')
        self.save_settings_key = DictKey('save_settings', label='saver_settings')
        self.save_name_key = DictKey('save_name', label='saver_settings')
        self.save_folder_name_key = DictKey('save_folder_name', label='saver_settings')

        # UI keys
        # tab1
        self.birefringence_type_dropdown_key = DictKey('birefringence_type_dropdown')
        self.selected_material_data_path_key = DictKey('selected_material_data_path')
        self.selected_fast_material_data_path_key = DictKey('selected_fast_material_data_path')
        self.selected_slow_material_data_path_key = DictKey('selected_slow_material_data_path')
        self.material_drop_down_key = DictKey('material_drop_down', label='material')
        self.fast_material_dropdown_key = DictKey('fast_material_dropdown', label='material')
        self.slow_material_dropdown_key = DictKey('slow_material_dropdown', label='material')
        self.new_material_name_key = DictKey('new_material_name')
        self.new_material_path_key = DictKey('new_material_path')
        self.x0_angle_input_key = DictKey('x0_angle_input')
        self.x0_widths_input_key = DictKey('x0_widths_input')
        self.x0_stripes_input_key = DictKey('x0_stripes_input')
        self.add_material_button_key = DictKey('add_material_button')
        self.new_x0_key = DictKey('set_x0_checkbox')
        self.random_x0_button_key = DictKey('random_x0_button')
        self.set_default_settings_button_key = DictKey('set_default_settings_button')
        self.enable_ri_overwrite_checkbox_key = DictKey('enable_ri_overwrite_checkbox')

        # tab4
        self.process_list_key = DictKey('process_list')
        self.basic_info_tab_l0_key = DictKey('basic_info_tab_l0')
        self.basic_info_tab_l1_key = DictKey('basic_info_tab_l1')
        self.basic_info_tab_l2_key = DictKey('basic_info_tab_l2')
        self.info_tab_l0_key = DictKey('info_tab_l0')
        self.info_tab_l1_key = DictKey('info_tab_l1')
        self.info_tab_l2_key = DictKey('info_tab_l2')
        self.info_tab_l3_key = DictKey('info_tab_l3')
        self.angle_input_key = DictKey('angle_input')
        self.widths_input_key = DictKey('widths_input')
        self.stripes_input_key = DictKey('stripes_input')
        self.start_process_button_key = DictKey('start_process_button')
        self.stop_process_button_key = DictKey('stop_process_button')
        self.stop_all_processes_button_key = DictKey('stop_all_processes_button')
        self.run_once_button_key = DictKey('run_once_button')
        self.set_selected_result_settings_button_key = DictKey('set_selected_result_settings_button')
        self.optimizer_test_button_key = DictKey('optimizer_test_button')
        self.run_once_label_input_key = DictKey('run_once_label_input')

        # tab5
        self.result_list_key = DictKey('result_list')
        self.folder_list_key = DictKey('folder_list')
        self.plot_in_cst_key = DictKey('plot_in_cst')
        self.erf_settings_key = DictKey('erf_settings')
        self.optimizer_settings_key = DictKey('optimizer_settings')
        self.saver_settings_key = DictKey('saver_settings')
        self.selected_frequency_key = DictKey('selected_frequency')
        self.actual_frequency_key = DictKey('actual_frequency')
        self.stokes_vector_key = DictKey('stokes_vector')
        self.jones_vector_key = DictKey('jones_vector')
        self.stokes_parameters_key1 = DictKey('stokes_parameters1')
        self.stokes_parameters_key2 = DictKey('stokes_parameters2')
        self.result_info_l0_key = DictKey('result_info_l0')
        self.result_info_l1_key = DictKey('result_info_l1')
        self.result_info_l2_key = DictKey('result_info_l2')
        self.result_info_l3_key = DictKey('result_info_l3')
        self.update_result_list_button_key = DictKey('update_result_list_button')
        self.update_folder_list_button_key = DictKey('update_folder_list_button')
        self.plot_selected_result_button_key = DictKey('plot_selected_result_button')

        # tab6
        self.plot_error_button_key = DictKey('plot_error_button')
        self.plot_without_error_key = DictKey('update_error_button')
        self.difference_plot_key = DictKey('difference_plot')
        self.polarization_plot_key = DictKey('polarization_plot')
        self.angle_err_checkbox_key = DictKey('angle_err_checkbox')
        self.width_err_checkbox_key = DictKey('width_err_checkbox')
        self.stripe_err_checkbox_key = DictKey('stripe_err_checkbox')
        self.angle_err_slider_key = DictKey('angle_err_slider')
        self.width_err_slider_key = DictKey('width_err_slider')
        self.stripe_err_slider_key = DictKey('stripe_err_slider')
        self.min_angle_error_input_key = DictKey('min_angle_error_input', label='error_sliders')
        self.max_angle_error_input_key = DictKey('max_angle_error_input', label='error_sliders')
        self.min_width_error_input_key = DictKey('min_width_error_input', label='error_sliders')
        self.max_width_error_input_key = DictKey('max_width_error_input', label='error_sliders')
        self.min_stripe_error_input_key = DictKey('min_stripe_error_input', label='error_sliders')
        self.max_stripe_error_input_key = DictKey('max_stripe_error_input', label='error_sliders')
        self.overwrite_stripes_key = DictKey('overwrite_stripes')
        self.stripe0_input_key = DictKey('stripe0_input')
        self.stripe1_input_key = DictKey('stripe1_input')
        self.plot_bf_real_key = DictKey('plot_bf_real')
        self.plot_bf_imag_key = DictKey('plot_bf_imag')
        self.plot_np_key = DictKey('plot_np')
        self.plot_ns_key = DictKey('plot_ns')
        self.plot_kp_key = DictKey('plot_kp')
        self.plot_ks_key = DictKey('plot_ks')
        self.plotted_error_angles_key = DictKey('plotted_error_angles')
        self.plotted_error_widths_key = DictKey('plotted_error_widths')
        self.plotted_error_stripes_key = DictKey('plotted_error_stripes')
        self.plot_birefringence_button_key = DictKey('plot_birefringence_button')
        self.plot_refractive_indices_button_key = DictKey('plot_refractive_indices_button')
        self.fix_old_settings_button_key = DictKey('fix_old_settings_button')

        # tab7
        self.update_cst_list_button_key = DictKey('update_cst_list_button')
        self.plot_cst_selections_button_key = DictKey('plot_cst_selections')
        self.cst_folders_key = DictKey('cst_folders')
        self.cst_file_list_key = DictKey('cst_file_list')
        self.cst_plot_x_key = DictKey('cst_plot_x')
        self.cst_plot_y_key = DictKey('cst_plot_y')

        # tab8
        self.single_wp_angle_input_key = DictKey('single_wp_angle_input')
        self.single_wp_width_input_key = DictKey('single_wp_width_input')
        self.single_wp_stripe1_width_input_key = DictKey('single_wp_stripe1_width_input')
        self.single_wp_stripe2_width_input_key = DictKey('single_wp_stripe2_width_input')
        self.single_wp_intensity_plot_button_key = DictKey('single_wp_intensity_plot_button')
        self.single_wp_label_input_key = DictKey('single_wp_label_input')
        self.single_wp_refractive_indices_plot_button_key = DictKey('single_wp_refractive_indices_plot_button')
        self.single_wp_plot_ns_checkbox_key = DictKey('plot_ns_checkbox_key')
        self.single_wp_plot_np_checkbox_key = DictKey('plot_np_checkbox_key')
        self.single_wp_plot_ks_checkbox_key = DictKey('plot_ks_checkbox_key')
        self.single_wp_plot_kp_checkbox_key = DictKey('plot_kp_checkbox_key')
        self.zeroth_order_freq_input_key = DictKey('zeroth_order_freq_input')
        self.calculate_zeroth_order_width_button_key = DictKey('calculate_zeroth_order_width_button')
        self.zeroth_order_width_result_l2_input_key = DictKey('zeroth_order_width_result_l2_input')
        self.zeroth_order_width_result_l4_input_key = DictKey('zeroth_order_width_result_l4_input')


if __name__ == '__main__':
    keys = DictKeys()
    for key in keys.__dict__:
        print(keys.__dict__[key].label)
