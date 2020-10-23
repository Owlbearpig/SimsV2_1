from numpy.random import uniform as rnd
from modules.identifiers.dict_keys import DictKeys
from modules.utils.constants import *


class InitPrep(DictKeys):
    def __init__(self, ui_values):
        super().__init__()
        self.ui_values = ui_values
        self.wp_cnt = ui_values[self.wp_cnt_key]
        self.x_slices = self.x_slicing()
        self.bounds = self.set_bounds()
        self.angles0, self.widths0, self.stripes0 = self.get_x0()

    def x_slicing(self):
        """
        which entries in x are angles, widths and bar widths. This doesn't change
        :return: ((int, int), (int, int), (int, int))
        first tuple is indices of angles,
        second: waveplate thicknesses,
        third: the two material widths
        """
        const_angles = np.array(self.ui_values[self.const_angles_key])
        const_widths = np.array(self.ui_values[self.const_widths_key])
        width_pattern = np.array(self.ui_values[self.width_pattern_key])

        angles2vary = const_angles[const_angles == 0].shape[0]
        widths2vary = np.unique(width_pattern[np.where(const_widths == 0)]).shape[0]
        stripes2vary = self.wp_cnt * 2

        if self.ui_values[self.const_wp_dim_key]:
            x_indices = [angles2vary, widths2vary, 0]
        else:
            if self.ui_values[self.equal_stripes_key]:
                x_indices = [angles2vary, widths2vary, 2]
            else:
                x_indices = [angles2vary, widths2vary, stripes2vary]

        angle_slice = (0, x_indices[0])
        width_slice = (x_indices[0], x_indices[0] + x_indices[1])
        stripe_slice = (x_indices[0] + x_indices[1], x_indices[0] + x_indices[1] + x_indices[2])

        if self.ui_values[self.birefringence_type_dropdown_key] in 'Natural' or self.ui_values[self.const_wp_dim_key]:
            stripe_slice = [0, 0]

        return list(angle_slice), list(width_slice), list(stripe_slice)

    def set_bounds(self):
        """
        sets bounds for optimization, needed for starting value x0

        :return: dict,
        "min_angles", "max_angles", in deg
        "min_widths", "max_widths", in um
        "min_stripes", "max_stripes" in um
        """
        angles2vary = self.x_slices[0][1] - self.x_slices[0][0]
        widths2vary = self.x_slices[1][1] - self.x_slices[1][0]
        stripes2vary = self.x_slices[2][1] - self.x_slices[2][0]

        min_angle, max_angle = self.ui_values[self.min_angle_key], self.ui_values[self.max_angle_key]
        min_width, max_width = self.ui_values[self.min_width_key], self.ui_values[self.max_width_key],
        min_stripe, max_stripe = self.ui_values[self.min_stripe_width_key], self.ui_values[self.max_stripe_width_key],

        new_bounds = {"min_angles": [min_angle] * angles2vary,
                      "max_angles": [max_angle] * angles2vary,
                      "min_widths": [min_width * um] * widths2vary,
                      "max_widths": [max_width * um] * widths2vary,
                      "min_stripes": [min_stripe * um] * stripes2vary,
                      "max_stripes": [max_stripe * um] * stripes2vary}

        return new_bounds

    def get_x0(self):
        """
        random start array inside bounds

        :return: np array
        """

        min_angles, max_angles = self.bounds["min_angles"], self.bounds["max_angles"]
        min_widths, max_widths = self.bounds["min_widths"], self.bounds["max_widths"]
        min_stripes, max_stripes = self.bounds["min_stripes"], self.bounds["max_stripes"]

        x_slices = self.x_slices

        angles2vary = x_slices[0][1] - x_slices[0][0]
        widths2vary = x_slices[1][1] - x_slices[1][0]
        stripes2vary = x_slices[2][1] - x_slices[2][0]

        angles0 = rnd(min_angles, max_angles, angles2vary)
        widths0 = rnd(min_widths, max_widths, widths2vary)
        stripes0 = rnd(min_stripes, max_stripes, stripes2vary)

        return list(np.round(angles0, 2)), list(np.round(widths0 * m_to_um, 1)), list(np.round(stripes0 * m_to_um, 1))
