
def correct_step_value(correction_parameter, depth_step):
        return correction_parameter - depth_step
def round_values(depth_step):
       return round(depth_step, 4)


def get_tvdss(measured_depth_data, rotary_table, topography_or_water_blade):

    if topography_or_water_blade is not None:
        correction_parameter = rotary_table + topography_or_water_blade
    else:
        correction_parameter = rotary_table
    tvdss_data = [correct_step_value(correction_parameter, i) for i in measured_depth_data]
    tvdss_data = [round_values(i) for i in tvdss_data]

    return tvdss_data
