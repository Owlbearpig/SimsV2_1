import numpy as np


class CSTData:
    def __init__(self, path):
        self.path = path
        self.f_s_parameters = self.make_freq_axis()

    def extract_float(self, line, column_index):
        line = str(line)
        line = line.replace('n', '')
        splits = line.split('\t')

        return float(splits[column_index])

    def make_freq_axis(self):
        freq_list = []
        with open(self.path) as file:
            lines = list(file)
            start_found, start = False, len(lines)
            for i, line in enumerate(lines):
                if 'Frequency' in str(line):
                    start_found, start = True, i

                if start_found and i > start+1:
                    if '#' in str(line):
                        break
                    freq_list.append(self.extract_float(line, 0))

        return np.array(freq_list)

    def get_s_parameters(self, in_port_index, out_port_index, data_type='magnitude'):

        if data_type == 'magnitude' or data_type == 1:
            column_index = 1
        elif data_type == 'phase' or data_type == 2:
            column_index = 2
        else:
            column_index = np.NaN

        default_port_names = ['Zmin(1)', 'Zmin(2)', 'Zmax(1)', 'Zmax(2)']

        in_, out_ = in_port_index, out_port_index
        in_name, out_name = default_port_names[in_-1], default_port_names[out_-1]

        data_list = []
        with open(self.path) as file:
            lines = list(file)
            start_found, start = False, len(lines)
            for i, line in enumerate(lines):
                if (f'S{in_},{out_}' in str(line)) or (f'{in_name},{out_name}' in str(line)):
                    start_found, start = True, i

                if start_found and i > start+1:
                    if '#' in str(line):
                        break
                    data_list.append(self.extract_float(line, column_index))

        return np.array(data_list)



if __name__ == '__main__':
    cst = CSTData('sim_results/s-parameters/30-11-2019_11-04-08_lowres.txt')
    print(cst.f_s_parameters)
    z24 = cst.get_s_parameters(4, 1)
    print(z24)

