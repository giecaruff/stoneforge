
# Ensure petrophysic output data (water saturation, shale volue and porosity) be at range [0, 1]
def correct_petrophysic_estimation_range(petrophysics_data):
    petrophysics_data[petrophysics_data > 1] = 1
    petrophysics_data[petrophysics_data < 0] = 1

    return petrophysics_data