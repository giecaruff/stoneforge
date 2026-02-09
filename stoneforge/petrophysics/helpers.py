# Ensure petrophysic output data (water saturation, shale volue and porosity) be at range [0, 1]
import numpy as np

def correct_petrophysic_estimation_range(petrophysics_data):
    return np.clip(petrophysics_data, 0.0, 1.0)
