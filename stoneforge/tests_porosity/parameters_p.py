import numpy as np

# ---------------------------------------------------------- #
# class to provide test configurations

class Parameters:

    def __init__(self):
        pass

    config_density = {
    "rhob":(1.1,5.0,"c"),
    "rhom":(1.1,5.0,"f"),
    "rhof":(0.0,2.65,"f")}

    config_neutron = {
    "nphi":(0.0,1.0,"c"),
    "vsh":(0.0,1.0,"f"),
    "nphi_sh":(0.0,1.0,"f"),
    }

    config_neutron_density = {
    "phid":(0.0,1.0,"c"),
    "phin":(0.0,1.0,"c"),
    }

    config_sonic = {
    "dt":(50.0,100.0,"c"),
    "dtma":(10.0,100.0,"f"),
    "dtf":(150.0,300.0,"f"),
    }

    config_gaymard = {
    "phid":(0.0,1.0,"c"),
    "phin":(0.0,1.0,"c"),
    }

    def sorted_values (configuration, size = 15, curve_size = 30, seed = 99):

        # curve_size = 15 issue

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

