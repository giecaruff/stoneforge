import numpy as np

# ---------------------------------------------------------- #
# class to provide test configurations

class Parameters:

    def __init__(self):
        pass

    config_water_saturarion = {
    "rw":(1.1,5.0),
    "rt":(1.1,5.0),
    "phi":(0.0,2.65),
    "a":(0.0,1.0),
    "m":(0.0,1.0),
    "n":(0.0,1.0),
    }

    def sorted_values (configuration, size = 15, seed = 99):

        np.random.seed(seed)

        # transform ["a","b","c"] into "a,b,c"
        list_names = list(configuration.keys())
        values_names = ','.join(list_names)

        properties_values = []
        for k in configuration:
            properties = np.random.uniform(low = configuration[k][0], high = configuration[k][1], size = size)
            properties_values.append(properties)

        properties_values = np.array(properties_values).T

        return properties_values,values_names
