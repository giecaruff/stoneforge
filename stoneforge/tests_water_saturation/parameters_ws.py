import numpy as np

# ---------------------------------------------------------- #
# class to provide test configurations

class Parameters:

    def __init__(self):
        pass

    config_archie = {
    "rw":(1.1,5.0,"f"),
    "rt":(1.1,5.0,"c"),
    "phi":(0.0,1.0,"c"),
    "a":(0.0,1.0,"f"),
    "m":(0.1,3.0,"f"),
    "n":(0.0,1.0,"f")
    }

    config_simandoux = {
    "rw":(1.1,5.0,"f"),
    "rt":(1.1,5.0,"c"),
    "phi":(0.0,1.0,"c"),
    "a":(0.0,1.0,"f"),
    "m":(0.1,3.0,"f"),
    "n":(0.0,1.0,"f"),
    "rsh":(0.1,4.0,"f"),
    "vsh":(0.0,1.0,"c")
    }

    config_indonesia = {
    "rw":(1.1,5.0,"f"),
    "rt":(1.1,5.0,"c"),
    "phi":(0.0,1.0,"c"),
    "a":(0.0,1.0,"f"),
    "m":(0.1,3.0,"f"),
    "n":(0.0,1.0,"f"),
    "rsh":(0.1,4.0,"f"),
    "vsh":(0.0,1.0,"c")
    }

    config_fertl = {
    "rw":(1.1,5.0,"f"),
    "rt":(1.1,5.0,"c"),
    "phi":(0.0,1.0,"c"),
    "a":(0.0,1.0,"f"),
    "m":(0.1,3.0,"f"),
    "vsh":(0.0,1.0,"c"),
    "alpha":(0.0,1.0,"f")
    }

    def sorted_values (configuration, size = 15, curve_size = 30, seed = 98):

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