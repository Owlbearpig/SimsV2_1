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


def is_conjugate_symmetric(j_stack):
    J00, J01, J10, J11 = j_stack[0, 0], j_stack[0, 1], j_stack[1, 0], j_stack[1, 1]
    # See if J01 is equal to J10
    cond1 = np.abs(J01 - np.conj(J10)) < tol_default ** 2
    cond2 = np.abs(J00 - np.conj(J00)) < tol_default ** 2
    cond3 = np.abs(J11 - np.conj(J11)) < tol_default ** 2
    cond = cond1 * cond2 * cond3

    return cond


def eig(j_stack):
    cond = is_conjugate_symmetric(j_stack)
    # Calculate the eigenstates
    val, vect = np.linalg.eig(j_stack)
    if np.any(cond):
        val2, vect2 = np.linalg.eigh(j_stack)
    # Order the values in the py_pol way
    v1, v2 = (val[:, 0], val[:, 1])
    e1 = np.array([vect[:, 0, 0], vect[:, 1, 0]])
    e2 = np.array([vect[:, 0, 1], vect[:, 1, 1]])
    if np.any(cond):
        v1[cond], v2[cond] = (val2[cond, 0], val2[cond, 1])
        e1[0, cond] = vect2[cond, 0, 0]
        e1[1, cond] = vect2[cond, 1, 0]
        e2[0, cond] = vect2[cond, 0, 1]
        e2[1, cond] = vect2[cond, 1, 1]
    if v1.size == 1 and v1.ndim > 1:
        v1, v2 = (v1[0], v2[0])

    E1, E2 = (e1, e2)

    return v1, v2, E1, E2


def eigenvalues(j_stack):
    cond = is_conjugate_symmetric(j_stack)
    M = np.moveaxis(j_stack, -1, 0)
    val = np.linalg.eigvals(M)
    if np.any(cond):
        val2 = np.linalg.eigvalsh(M)
    # Order the values in the py_pol way
    v1, v2 = (val[:, 0], val[:, 1])
    if np.any(cond):
        v1[cond], v2[cond] = (val2[cond, 0], val2[cond, 1])
    if v1.size == 1 and v1.ndim > 1:
        v1, v2 = (v1[0], v2[0])

    return v1, v2


def determinant(j_stack):
    M = np.moveaxis(j_stack, -1, 0)

    return np.linalg.det(M)


def inhomogeneity(j_stack):
    # Calculate the values
    det = determinant(j_stack)
    trace = np.trace(j_stack)
    norm2 = np.linalg.norm(j_stack, axis=(0, 1)) ** 2
    # Calculate the parameter
    a = norm2 - 0.5 * np.abs(trace)**2
    b = 0.5 * np.abs(trace**2 - 4 * det)
    eta = (a - b) / (a + b)

    return eta


# copied from py-pol
def retardance(j_stack):
    # Makes j_stack look like a py_pol array
    j_stack = np.moveaxis(j_stack, 0, -1)

    v1, v2 = eigenvalues(j_stack)
    a1, a2 = (np.abs(v1), np.abs(v2))
    det = determinant(j_stack)
    trace = np.trace(j_stack)
    norm2 = np.linalg.norm(j_stack, axis=(0, 1)) ** 2
    eta = inhomogeneity(j_stack)
    # Act differently if the object is homogeneous
    cond1 = eta < tol_default ** 2
    R = np.zeros_like(eta)
    # Homogeneous case
    if np.any(cond1):
        cond2 = np.abs(det) < tol_default ** 2
        R[cond1 * cond2] = 2 * np.arccos(
            np.abs(trace[cond1 * cond2]) / np.sqrt(norm2[cond1 * cond2]))
        cond2 = ~cond2
        num = np.abs(trace[cond1 * cond2] +
                     det[cond1 * cond2] * np.conj(trace[cond1 * cond2]) /
                     np.abs(det[cond1 * cond2]))
        den = 2 * np.sqrt(norm2[cond1 * cond2] +
                          2 * np.abs(det[cond1 * cond2]))
        R[cond1 * cond2] = 2 * np.arccos(num / den)
    # Inhomogeneous case
    cond1 = ~cond1
    if np.any(cond1):
        num = (1 - eta[cond1] ** 2) * (a1[cond1] + a2[cond1]) ** 2
        den = (a1[cond1] + a2[cond1]) ** 2 - eta[cond1] ** 2 * (
                2 * v1[cond1] * a1[cond1] * a2[cond1] +
                np.conj(v2[cond1] + v2[cond1] * np.conj(v1[cond1])))
        co = np.cos((np.angle(v1[cond1]) - np.angle(v2[cond1])) / 2)
        R[cond1] = 2 * np.arccos(np.sqrt(num / den) * co).real
    # D must be real, but complex numbers are used during calculation
    R = np.array(R, dtype=float)

    return R


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