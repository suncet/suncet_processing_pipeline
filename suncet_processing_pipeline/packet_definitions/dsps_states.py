CONVERT_dsps_cmd_last_status_dict = { 0 : 'CMD_SUCCESS', 1 : 'CMD_FAIL_OPCODE', 2 : 'CMD_FAIL_PARAM_MISSING', 4 : 'CMD_FAIL_PARAM_BAD', 255 : 'CMD_FAIL',}
def CONVERT_dsps_cmd_last_status(val):
    try:
        return CONVERT_dsps_cmd_last_status_dict[val]
    except:
        return val
CONVERT_dsps_flare_magnitude_phase_dict = { 0 : 'UNKNOWN', 1 : 'BACKGROUND', 3 : 'ELEVATED_BACKGROUND', 16 : 'RISING_FLARE', 20 : 'DECLINING_FLARE',}
def CONVERT_dsps_flare_magnitude_phase(val):
    try:
        return CONVERT_dsps_flare_magnitude_phase_dict[val]
    except:
        return val
CONVERT_dsps_param_34_flare_channel_dict = { 0 : 'OFF', 1 : 'SPS-1', 2 : 'SPS-2',}
def CONVERT_dsps_param_34_flare_channel(val):
    try:
        return CONVERT_dsps_param_34_flare_channel_dict[val]
    except:
        return val