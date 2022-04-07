import numpy as np

# ---------------------------------------------------------- #
# class to provide test configurations

class Parameters:
    
    def __init__(self):
        pass    
    
    config_density = {
    "rhob":(0.0,1.0),
    "rhom":(0.0,1.0),
    "rhof":(0.0,1.0),
    }

    config_neutron = {
    "nphi":(0.0,1.0),
    "vsh":(0.0,1.0),
    "nphi_sh":(0.0,1.0),
    }

    config_neutron_density = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
    }

    config_sonic = {
    "dt":(50.0,100.0),
    "dtma":(10.0,100.0),
    "dtf":(150.0,300.0),
    }

    config_gaymard = {
    "phid":(0.0,1.0),
    "phin":(0.0,1.0),
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
