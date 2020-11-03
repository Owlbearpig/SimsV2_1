from modules.utils.constants import *
from py_pol.stokes import Stokes
from py_pol.mueller import Mueller
from py_pol.jones_vector import Jones_vector
from py_pol.jones_matrix import Jones_matrix


# creates array(freq_len, 2, 2) (one chain of matrices for each frequency) for the input
def make_j_matrix_stack(erf_setup, refractive_indices, values):
    # don't need the stripes here :)
    angles, widths, _ = values
    theta, x, y = erf_setup.j_matrix_input(angles, widths, *refractive_indices)
    J_all_freq = erf_setup.build_j_matrix_stack(theta, x, y)
    return J_all_freq


# creates array(freq_len, 4, 4) (one chain of matrices for each frequency) for the input
def make_m_matrix_stack(erf_setup, refractive_indices, values):
    # don't need the stripes here :)
    angles, widths, _ = values
    theta, ret, p_s, p_p = erf_setup.m_matrix_input(angles, widths, *refractive_indices)
    M_all_freq = erf_setup.build_m_matrix_stack(theta, ret, p_s, p_p)

    return M_all_freq


# returns jones vector at the end of the stack at the selected_frequency_index (closest one to selected_frequency)
def calculate_final_jones_vector(j_matrix_stack, selected_frequency_index):
    # Jones matrix at selected frequency for the wp stack
    J_one_freq = Jones_matrix('stack')
    J_one_freq.from_matrix(j_matrix_stack[selected_frequency_index])

    # initial polarization is along x-axis
    x_lin_pol_jones = Jones_vector('x_pol')
    x_lin_pol_jones.linear_light()

    return J_one_freq * x_lin_pol_jones


# returns stokes vector at the end of the stack at the selected_frequency_index (closest one to selected_frequency)
def calculate_final_stokes_vector(m_matrix_stack, selected_frequency_index):
    # Mueller matrix at selected frequency for the wp stack
    M_one_freq = Mueller('stack')
    M_one_freq.from_matrix(m_matrix_stack[selected_frequency_index])

    # initial polarization is along x-axis
    x_lin_pol_stokes = Stokes('x_pol')
    x_lin_pol_stokes.linear_light()

    return M_one_freq * x_lin_pol_stokes


def calculate_final_vectors(m_matrix_stack, j_matrix_stack, erf_setup, selected_frequency):
    selected_frequency_index = np.argmin(np.abs(erf_setup.frequencies - selected_frequency))

    sf = calculate_final_stokes_vector(m_matrix_stack, selected_frequency_index)
    jf = calculate_final_jones_vector(j_matrix_stack, selected_frequency_index)

    return sf, jf, erf_setup.frequencies[selected_frequency_index]


# calculate intensities for polarizer along x and y, given matrix stack
def calc_final_stokes_intensities(m_matrix_stack):
    x_pol_w_stack = np.einsum('ij,fjk->fik', x_pol_m, m_matrix_stack)
    y_pol_w_stack = np.einsum('ij,fjk->fik', y_pol_m, m_matrix_stack)
    int_x = np.einsum('fjk,k->fj', x_pol_w_stack, stokes_initial)
    int_y = np.einsum('fjk,k->fj', y_pol_w_stack, stokes_initial)

    return 10*np.log10(int_x[:, 0]), 10*np.log10(int_y[:, 0])


# calculate intensities for polarizer along x and y, given matrix stack
def calc_final_jones_intensities(j_matrix_stack):
    int_x = j_matrix_stack[:, 0, 0]*np.conjugate(j_matrix_stack[:, 0, 0])
    int_y = j_matrix_stack[:, 1, 0]*np.conjugate(j_matrix_stack[:, 1, 0])

    return 10*np.log10(int_x.real), 10*np.log10(int_y.real)


# calculate polarization degrees
def calc_polarization_degrees_m(matrix_stack):
    stokes_final_all = np.einsum('fjk,k->fj', matrix_stack, stokes_initial)
    linear_comp_lst, circular_comp_lst = [], []
    for stokes_final in stokes_final_all:
        sf = Stokes()
        sf.from_matrix(stokes_final)

        linear_comp_lst.append(sf.parameters.degree_linear_polarization())
        circular_comp_lst.append(sf.parameters.degree_circular_polarization())

    return linear_comp_lst, circular_comp_lst


def calc_polarization_degrees_j(matrix_stack):
    jones_final_all = np.einsum('fjk,k->fj', matrix_stack, jones_initial)
    linear_comp_lst, circular_comp_lst = [], []
    for jones_final in jones_final_all:
        jf = Jones_vector()
        jf.from_matrix(jones_final)

        linear_comp_lst.append(jf.parameters.degree_linear_polarization())
        circular_comp_lst.append(jf.parameters.degree_circular_polarization())

    return linear_comp_lst, circular_comp_lst


def rotate_matrix(matrix, theta):
    r, r_inv = r_z_j(theta), r_z_j(-theta)
    return np.einsum('ab,vbc,cd->vad', r_inv, matrix, r)


def fft(t, y):
    delta = np.float(np.mean(np.diff(t)))
    Y = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(t), delta)
    idx = freqs > 0

    return freqs[idx], Y[idx]


if __name__ == '__main__':
    pass
"""
def delays(M, freqs, wp_angle=0):
    delay_lst = []
    for freq_I, freq in enumerate(freqs):
        M_singlef = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                M_singlef[i, j] = np.array(M[i, j])[freq_I]

        M_stack = Mueller('M_singlef')
        M_stack.from_elements(*M_singlef.flatten())
        M_stack = M_stack.rotate(angle=wp_angle * degrees, keep=True)

        if freq_I == 1000:
            print(M_singlef)

        s0 = Stokes('s0')
        s0.linear_light(angle=0 * degrees, intensity=1)

        delay_lst.append((M_stack * s0).parameters.delay())

    return delay_lst
"""