import numpy as np

# ---------------------------------------------------------- #
# class to provide test configurations

class Parameters:

    def __init__(self):
        pass

    config_water_saturarion = {
    "rw":(1.1,5.0,"f"),
    "rt":(1.1,5.0,"c"),
    "phi":(0.0,2.65,"c"),
    "a":(0.0,1.0,"f"),
    "m":(0.0,1.0,"f"),
    "n":(0.0,1.0,"f"),
    }

    def sorted_values (configuration, size = 15, curve_size = 30, seed = 99):

            np.random.seed(seed)

            list_names = list(configuration.keys())
            values_names = ','.join(list_names)

            properties_values = []
            for k in configuration:
                if configuration[k][2] == "f":
                    properties = np.random.uniform(low = configuration[k][0], high = configuration[k][1], size = size)
                    properties_values.append(properties)
                if configuration[k][2] == "c":
                    curves = []
                    for i in range(size):
                        curves.append(np.random.uniform(low = configuration[k][0], high = configuration[k][1], size = curve_size))
                    properties_values.append(curves)

            properties_values = list(map(list, zip(*properties_values)))

            return properties_values,values_names