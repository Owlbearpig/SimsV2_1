from modules.identifiers.dict_keys import DictKeys
keys = DictKeys()

default_settings_dict = {
    # erf settings
    keys.wp_cnt_key: 5,
    keys.const_wp_dim_key: False,
    keys.equal_stripes_key: True,
    keys.initial_stripe_widths_key: [[50] * 5,
                                     [100] * 5],
    keys.const_angles_key: [0]*5,
    keys.const_widths_key: [0]*5,
    keys.width_pattern_key: list(range(1, 6)),
    keys.frequency_resolution_multiplier_key: 1.0,
    keys.min_freq_key: 0.25,
    keys.max_freq_key: 1.0,
    keys.material_drop_down_key: "",
    keys.birefringence_type_dropdown_key: "Form",
    keys.fast_material_dropdown_key: "",
    keys.slow_material_dropdown_key: "",
    keys.const_refr_index_input_n_s_key: 0.0,
    keys.const_refr_index_input_k_s_key: 0.0,
    keys.const_refr_index_input_n_p_key: 0.0,
    keys.const_refr_index_input_k_p_key: 0.0,
    keys.randomizer_seed_key: None,
    keys.log_of_res_key: False,
    keys.min_width_key: 0.0,
    keys.max_width_key: 500,
    keys.min_angle_key: 0.0,
    keys.max_angle_key: 2*3.141592,
    keys.min_stripe_width_key: 5.0,
    keys.max_stripe_width_key: 90.0,
    keys.anisotropy_p_key: 1.0,
    keys.anisotropy_s_key: 1.0,
    keys.wp_type_key: "λ/2",
    keys.x_slicing_key: [[0, 5], [5, 10], [10, 12]],
    keys.calculation_method_key: "Stokes",

    # optimizer settings
    keys.iterations_key: 10**5,
    keys.angle_step_key: 0.1,
    keys.width_step_key: 10.0,
    keys.stripe_width_step_key: 5.0,
    keys.temperature_key: 0.5,
    keys.local_opt_tol_key: 0.001,
    keys.print_precision_key: 5,
    keys.print_interval_key: 100,
    keys.periodic_restart_key: False,
    keys.disable_callback_key: False,

    # saver settings
    keys.save_all_results_key: False,
    keys.save_settings_key: True,
    keys.save_iterations_key: False,
    keys.save_name_key: "",
    keys.save_folder_name_key: "",

    # UI settings
    # tab1
    keys.selected_material_data_path_key: "",
    keys.selected_fast_material_data_path_key: "",
    keys.selected_slow_material_data_path_key: "",
    keys.new_material_name_key: "",
    keys.new_material_path_key: "",
    keys.enable_ri_overwrite_checkbox_key: False,

    keys.x0_angle_input_key: [0, 0, 0, 0, 0],
    keys.x0_widths_input_key: [0, 0, 0, 0, 0],
    keys.x0_stripes_input_key: [0, 0],

    # tab2 (optimizer settings)
    # tab3 (save settings)
    # tab4
    keys.basic_info_tab_l0_key: "",
    keys.basic_info_tab_l1_key: "",
    keys.info_tab_l0_key: "",
    keys.info_tab_l1_key: "",
    keys.info_tab_l2_key: "",
    keys.info_tab_l3_key: "",
    keys.angle_input_key: [0, 0, 0, 0, 0],
    keys.widths_input_key: [0, 0, 0, 0, 0],
    keys.stripes_input_key: [0, 0],
    keys.run_once_label_input_key: "",

    # tab5
    keys.plot_in_cst_key: False,
    keys.selected_frequency_key: 0.5,

    # tab6
    keys.difference_plot_key: False,
    keys.polarization_plot_key: False,
    keys.angle_err_checkbox_key: False,
    keys.width_err_checkbox_key: False,
    keys.stripe_err_checkbox_key: False,
    keys.min_angle_error_input_key: 0.0,
    keys.max_angle_error_input_key: 1.0,
    keys.min_width_error_input_key: 1.0,
    keys.max_width_error_input_key: 5.0,
    keys.min_stripe_error_input_key: 1.0,
    keys.max_stripe_error_input_key: 5.0,
    keys.overwrite_stripes_key: False,
    keys.stripe0_input_key: 1.0,
    keys.stripe1_input_key: 5.0,
    keys.plot_bf_real_key: True,
    keys.plot_bf_imag_key: True,
    keys.plot_np_key: True,
    keys.plot_ns_key: True,
    keys.plot_kp_key: False,
    keys.plot_ks_key: False,

    # tab7
    keys.cst_plot_x_key: True,
    keys.cst_plot_y_key: True,

    # tab8
    keys.single_wp_angle_input_key: 45.0,
    keys.single_wp_width_input_key: 1200.0,
    keys.single_wp_stripe1_width_input_key: 100.0,
    keys.single_wp_stripe2_width_input_key: 50.0,
    keys.single_wp_label_input_key: "",
    keys.single_wp_plot_ns_checkbox_key: True,
    keys.single_wp_plot_np_checkbox_key: True,
    keys.single_wp_plot_ks_checkbox_key: True,
    keys.single_wp_plot_kp_checkbox_key: True,
    keys.zeroth_order_freq_input_key: 300.0,
}
