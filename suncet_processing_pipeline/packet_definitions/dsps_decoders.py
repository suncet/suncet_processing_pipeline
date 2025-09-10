import struct
import datetime
from suncet_processing_pipeline.packet_definitions import dsps_eus
from suncet_processing_pipeline.packet_definitions import dsps_states

def getUnsigned(packet, endian, shift, mask):
    val = int.from_bytes(packet, endian)
    val = (val >> shift) & mask
    return val


def getSigned(packet, endian, shift, mask):
    val = int.from_bytes(packet, endian, signed=True)
    subtract = 0
    if(val < 0):
        subtract = mask + 1 #python casts integers to unsigned during bitwise and, save this to ensure we can recover the true value after the mask
    val = (val >> shift) & mask
    return val - subtract

def getFloat(packet, endian):
    if endian == 'BIG':
        return struct.unpack('>f', packet)[0]
    else:
        return struct.unpack('<f', packet)[0]

class DSPS_DATA:
    def __str__(self):
        return 'DSPS_DATA'

    def __init__(self, packet, header, file_origin):
        apidField = int.from_bytes(header[0:2], 'big')
        self.version = apidField >> 13
        self.type = apidField >> 12 & 0x01
        self.secHdr = apidField >> 11 & 0x01
        self.apid = apidField & 0x7FF
        self.src_seq_ctr = int.from_bytes(header[2:4], 'big') & 0x3FFF
        self.group = int.from_bytes(header[2:4], 'big') & 0xC000 >> 14
        self.length = int.from_bytes(header[4:6], 'big')
        if len(header) > 6:
            self.SHFINE = int.from_bytes(header[6:8], 'big')
            self.SHCOARSE = int.from_bytes(header[8:12], 'big')
            ertTime = self.SHCOARSE * 1000 + self.SHFINE
            self.ground_isotime = datetime.datetime.fromtimestamp(ertTime/1000.0).isoformat()
        
        self.ccsdsSecHeader2_sec_dsps_data = getUnsigned(packet[0:4], 'big', 0, 4294967295)
        
        self.ccsdsSecHeader2_sub_dsps_data = getUnsigned(packet[4:6], 'big', 0, 65535)
        
        self.dsps_sw_ver_major = getUnsigned(packet[6:7], 'big', 0, 255)
        
        self.dsps_sw_ver_minor = getUnsigned(packet[7:8], 'big', 0, 255)
        
        self.dsps_cmd_last_opcode = getUnsigned(packet[8:9], 'big', 0, 255)
        
        self.dsps_cmd_last_status = getUnsigned(packet[9:10], 'little', 0, 255)
        try:
            self.dsps_cmd_last_status = dsps_states.CONVERT_dsps_cmd_last_status(self.dsps_cmd_last_status)
        except:
            pass
        
        self.dsps_cmd_accept_cnt = getUnsigned(packet[10:12], 'little', 0, 65535)
        
        self.dsps_cmd_reject_cnt = getUnsigned(packet[12:14], 'little', 0, 65535)
        
        self.dsps_cdh_mode_flags = getUnsigned(packet[14:16], 'little', 0, 65535)
        
        self.dsps_inst_mode_flags = getUnsigned(packet[16:18], 'little', 0, 65535)
        
        self.dsps_power_3P3V_mv = getUnsigned(packet[18:20], 'little', 0, 65535)
        
        self.dsps_power_3P3V_I_mA = getUnsigned(packet[20:22], 'little', 0, 65535)
        
        self.dsps_power_5V_mV = getUnsigned(packet[22:24], 'little', 0, 65535)
        
        self.dsps_power_5V_I_mA = getUnsigned(packet[24:26], 'little', 0, 65535)
        
        self.dsps_power_Vbat_mV = getUnsigned(packet[26:28], 'little', 0, 65535)
        
        self.dsps_power_Vbat_I_mA = getUnsigned(packet[28:30], 'little', 0, 65535)
        
        self.dsps_cdh_dac_1 = getUnsigned(packet[30:32], 'little', 0, 65535)
        
        self.dsps_power_5V_mon = getUnsigned(packet[32:34], 'little', 0, 65535)
        try:
            self.dsps_power_5V_mon = dsps_eus.CONVERT_dsps_power_5V_mon(self.dsps_power_5V_mon)
        except:
            pass
        
        self.dsps_power_3P3V_mon = getUnsigned(packet[34:36], 'little', 0, 65535)
        try:
            self.dsps_power_3P3V_mon = dsps_eus.CONVERT_dsps_power_3P3V_mon(self.dsps_power_3P3V_mon)
        except:
            pass
        
        self.dsps_power_2P5V_mon = getUnsigned(packet[36:38], 'little', 0, 65535)
        try:
            self.dsps_power_2P5V_mon = dsps_eus.CONVERT_dsps_power_2P5V_mon(self.dsps_power_2P5V_mon)
        except:
            pass
        
        self.dsps_power_1P8V_mon = getUnsigned(packet[38:40], 'little', 0, 65535)
        try:
            self.dsps_power_1P8V_mon = dsps_eus.CONVERT_dsps_power_1P8V_mon(self.dsps_power_1P8V_mon)
        except:
            pass
        
        self.dsps_arm_proc_temp_V = getUnsigned(packet[40:42], 'little', 0, 65535)
        try:
            self.dsps_arm_proc_temp_V = dsps_eus.CONVERT_dsps_arm_proc_temp_V(self.dsps_arm_proc_temp_V)
        except:
            pass
        
        self.dsps_ana_Vbat_mon = getUnsigned(packet[42:44], 'little', 0, 65535)
        try:
            self.dsps_ana_Vbat_mon = dsps_eus.CONVERT_dsps_ana_Vbat_mon(self.dsps_ana_Vbat_mon)
        except:
            pass
        
        self.dsps_adc_temp_V = getUnsigned(packet[44:46], 'little', 0, 65535)
        try:
            self.dsps_adc_temp_V = dsps_eus.CONVERT_dsps_adc_temp_V(self.dsps_adc_temp_V)
        except:
            pass
        
        self.dsps_sensor_diode_1_temp_C = getUnsigned(packet[46:48], 'little', 0, 65535)
        try:
            self.dsps_sensor_diode_1_temp_C = dsps_eus.CONVERT_dsps_sensor_diode_1_temp_C(self.dsps_sensor_diode_1_temp_C)
        except:
            pass
        
        self.dsps_sensor_diode_2_temp_C = getUnsigned(packet[48:50], 'little', 0, 65535)
        try:
            self.dsps_sensor_diode_2_temp_C = dsps_eus.CONVERT_dsps_sensor_diode_2_temp_C(self.dsps_sensor_diode_2_temp_C)
        except:
            pass
        
        self.dsps_sensor_board_temp_C = getUnsigned(packet[50:52], 'little', 0, 65535)
        try:
            self.dsps_sensor_board_temp_C = dsps_eus.CONVERT_dsps_sensor_board_temp_C(self.dsps_sensor_board_temp_C)
        except:
            pass
        
        self.dsps_flare_magnitude = getUnsigned(packet[52:53], 'little', 0, 255)
        
        self.dsps_flare_magnitude_phase = getUnsigned(packet[53:54], 'little', 0, 255)
        try:
            self.dsps_flare_magnitude_phase = dsps_states.CONVERT_dsps_flare_magnitude_phase(self.dsps_flare_magnitude_phase)
        except:
            pass
        
        self.dsps_sps_1_loop_count = getUnsigned(packet[54:56], 'little', 0, 65535)
        
        self.dsps_sps_1_err_count = getUnsigned(packet[56:58], 'little', 0, 65535)
        
        self.dsps_sps_1_diodes_0 = getUnsigned(packet[58:62], 'little', 0, 4294967295)
        
        self.dsps_sps_1_diodes_1 = getUnsigned(packet[62:66], 'little', 0, 4294967295)
        
        self.dsps_sps_1_diodes_2 = getUnsigned(packet[66:70], 'little', 0, 4294967295)
        
        self.dsps_sps_1_diodes_3 = getUnsigned(packet[70:74], 'little', 0, 4294967295)
        
        self.dsps_sps_1_sum = getUnsigned(packet[74:78], 'little', 0, 4294967295)
        
        self.dsps_sps_1_x_pos = getUnsigned(packet[78:80], 'little', 0, 65535)
        
        self.dsps_sps_1_y_pos = getUnsigned(packet[80:82], 'little', 0, 65535)
        
        self.dsps_sps_2_loop_count = getUnsigned(packet[82:84], 'little', 0, 65535)
        
        self.dsps_sps_2_err_count = getUnsigned(packet[84:86], 'little', 0, 65535)
        
        self.dsps_sps_2_diodes_0 = getUnsigned(packet[86:90], 'little', 0, 4294967295)
        
        self.dsps_sps_2_diodes_1 = getUnsigned(packet[90:94], 'little', 0, 4294967295)
        
        self.dsps_sps_2_diodes_2 = getUnsigned(packet[94:98], 'little', 0, 4294967295)
        
        self.dsps_sps_2_diodes_3 = getUnsigned(packet[98:102], 'little', 0, 4294967295)
        
        self.dsps_sps_2_sum = getUnsigned(packet[102:106], 'little', 0, 4294967295)
        
        self.dsps_sps_2_x_pos = getUnsigned(packet[106:108], 'little', 0, 65535)
        
        self.dsps_sps_2_y_pos = getUnsigned(packet[108:110], 'little', 0, 65535)
        
        self.dsps_tlm_integ_time = getUnsigned(packet[110:112], 'little', 0, 65535)
        
        self.dsps_data_checksum = getUnsigned(packet[112:114], 'little', 0, 4294967295)

        try:
            self.pktTimestamp = self.SHCOARSE
        except:
            self.pktTimestamp = 0

        self.file_origin = file_origin

class DSPS_LOG:
    def __str__(self):
        return 'DSPS_LOG'

    def __init__(self, packet, header, file_origin):
        apidField = int.from_bytes(header[0:2], 'big')
        self.version = apidField >> 13
        self.type = apidField >> 12 & 0x01
        self.secHdr = apidField >> 11 & 0x01
        self.apid = apidField & 0x7FF
        self.src_seq_ctr = int.from_bytes(header[2:4], 'big') & 0x3FFF
        self.group = int.from_bytes(header[2:4], 'big') & 0xC000 >> 14
        self.length = int.from_bytes(header[4:6], 'big')
        if len(header) > 6:
            self.SHFINE = int.from_bytes(header[6:8], 'big')
            self.SHCOARSE = int.from_bytes(header[8:12], 'big')
            ertTime = self.SHCOARSE * 1000 + self.SHFINE
            self.ground_isotime = datetime.datetime.fromtimestamp(ertTime/1000.0).isoformat()
        
        self.ccsdsSecHeader2_sec_dsps_log = getUnsigned(packet[0:4], 'big', 0, 4294967295)
        
        self.ccsdsSecHeader2_sub_dsps_log = getUnsigned(packet[4:6], 'big', 0, 65535)
        
        self.dsps_log_message_0 = getUnsigned(packet[6:7], 'little', 0, 255)
        
        self.dsps_log_message_1 = getUnsigned(packet[7:8], 'little', 0, 255)
        
        self.dsps_log_message_2 = getUnsigned(packet[8:9], 'little', 0, 255)
        
        self.dsps_log_message_3 = getUnsigned(packet[9:10], 'little', 0, 255)
        
        self.dsps_log_message_4 = getUnsigned(packet[10:11], 'little', 0, 255)
        
        self.dsps_log_message_5 = getUnsigned(packet[11:12], 'little', 0, 255)
        
        self.dsps_log_message_6 = getUnsigned(packet[12:13], 'little', 0, 255)
        
        self.dsps_log_message_7 = getUnsigned(packet[13:14], 'little', 0, 255)
        
        self.dsps_log_message_8 = getUnsigned(packet[14:15], 'little', 0, 255)
        
        self.dsps_log_message_9 = getUnsigned(packet[15:16], 'little', 0, 255)
        
        self.dsps_log_message_10 = getUnsigned(packet[16:17], 'little', 0, 255)
        
        self.dsps_log_message_11 = getUnsigned(packet[17:18], 'little', 0, 255)
        
        self.dsps_log_message_12 = getUnsigned(packet[18:19], 'little', 0, 255)
        
        self.dsps_log_message_13 = getUnsigned(packet[19:20], 'little', 0, 255)
        
        self.dsps_log_message_14 = getUnsigned(packet[20:21], 'little', 0, 255)
        
        self.dsps_log_message_15 = getUnsigned(packet[21:22], 'little', 0, 255)
        
        self.dsps_log_message_16 = getUnsigned(packet[22:23], 'little', 0, 255)
        
        self.dsps_log_message_17 = getUnsigned(packet[23:24], 'little', 0, 255)
        
        self.dsps_log_message_18 = getUnsigned(packet[24:25], 'little', 0, 255)
        
        self.dsps_log_message_19 = getUnsigned(packet[25:26], 'little', 0, 255)
        
        self.dsps_log_message_20 = getUnsigned(packet[26:27], 'little', 0, 255)
        
        self.dsps_log_message_21 = getUnsigned(packet[27:28], 'little', 0, 255)
        
        self.dsps_log_message_22 = getUnsigned(packet[28:29], 'little', 0, 255)
        
        self.dsps_log_message_23 = getUnsigned(packet[29:30], 'little', 0, 255)
        
        self.dsps_log_message_24 = getUnsigned(packet[30:31], 'little', 0, 255)
        
        self.dsps_log_message_25 = getUnsigned(packet[31:32], 'little', 0, 255)
        
        self.dsps_log_message_26 = getUnsigned(packet[32:33], 'little', 0, 255)
        
        self.dsps_log_message_27 = getUnsigned(packet[33:34], 'little', 0, 255)
        
        self.dsps_log_message_28 = getUnsigned(packet[34:35], 'little', 0, 255)
        
        self.dsps_log_message_29 = getUnsigned(packet[35:36], 'little', 0, 255)
        
        self.dsps_log_message_30 = getUnsigned(packet[36:37], 'little', 0, 255)
        
        self.dsps_log_message_31 = getUnsigned(packet[37:38], 'little', 0, 255)
        
        self.dsps_log_message_32 = getUnsigned(packet[38:39], 'little', 0, 255)
        
        self.dsps_log_message_33 = getUnsigned(packet[39:40], 'little', 0, 255)
        
        self.dsps_log_message_34 = getUnsigned(packet[40:41], 'little', 0, 255)
        
        self.dsps_log_message_35 = getUnsigned(packet[41:42], 'little', 0, 255)
        
        self.dsps_log_message_36 = getUnsigned(packet[42:43], 'little', 0, 255)
        
        self.dsps_log_message_37 = getUnsigned(packet[43:44], 'little', 0, 255)
        
        self.dsps_log_message_38 = getUnsigned(packet[44:45], 'little', 0, 255)
        
        self.dsps_log_message_39 = getUnsigned(packet[45:46], 'little', 0, 255)
        
        self.dsps_log_message_40 = getUnsigned(packet[46:47], 'little', 0, 255)
        
        self.dsps_log_message_41 = getUnsigned(packet[47:48], 'little', 0, 255)
        
        self.dsps_log_message_42 = getUnsigned(packet[48:49], 'little', 0, 255)
        
        self.dsps_log_message_43 = getUnsigned(packet[49:50], 'little', 0, 255)
        
        self.dsps_log_message_44 = getUnsigned(packet[50:51], 'little', 0, 255)
        
        self.dsps_log_message_45 = getUnsigned(packet[51:52], 'little', 0, 255)
        
        self.dsps_log_message_46 = getUnsigned(packet[52:53], 'little', 0, 255)
        
        self.dsps_log_message_47 = getUnsigned(packet[53:54], 'little', 0, 255)
        
        self.dsps_log_message_48 = getUnsigned(packet[54:55], 'little', 0, 255)
        
        self.dsps_log_message_49 = getUnsigned(packet[55:56], 'little', 0, 255)
        
        self.dsps_log_message_50 = getUnsigned(packet[56:57], 'little', 0, 255)
        
        self.dsps_log_message_51 = getUnsigned(packet[57:58], 'little', 0, 255)
        
        self.dsps_log_message_52 = getUnsigned(packet[58:59], 'little', 0, 255)
        
        self.dsps_log_message_53 = getUnsigned(packet[59:60], 'little', 0, 255)
        
        self.dsps_log_message_54 = getUnsigned(packet[60:61], 'little', 0, 255)
        
        self.dsps_log_message_55 = getUnsigned(packet[61:62], 'little', 0, 255)
        
        self.dsps_log_message_56 = getUnsigned(packet[62:63], 'little', 0, 255)
        
        self.dsps_log_message_57 = getUnsigned(packet[63:64], 'little', 0, 255)
        
        self.dsps_log_message_58 = getUnsigned(packet[64:65], 'little', 0, 255)
        
        self.dsps_log_message_59 = getUnsigned(packet[65:66], 'little', 0, 255)
        
        self.dsps_log_message_60 = getUnsigned(packet[66:67], 'little', 0, 255)
        
        self.dsps_log_message_61 = getUnsigned(packet[67:68], 'little', 0, 255)
        
        self.dsps_log_message_62 = getUnsigned(packet[68:69], 'little', 0, 255)
        
        self.dsps_log_message_63 = getUnsigned(packet[69:70], 'little', 0, 255)
        
        self.dsps_log_message_64 = getUnsigned(packet[70:71], 'little', 0, 255)
        
        self.dsps_log_message_65 = getUnsigned(packet[71:72], 'little', 0, 255)
        
        self.dsps_log_message_66 = getUnsigned(packet[72:73], 'little', 0, 255)
        
        self.dsps_log_message_67 = getUnsigned(packet[73:74], 'little', 0, 255)
        
        self.dsps_log_message_68 = getUnsigned(packet[74:75], 'little', 0, 255)
        
        self.dsps_log_message_69 = getUnsigned(packet[75:76], 'little', 0, 255)
        
        self.dsps_log_message_70 = getUnsigned(packet[76:77], 'little', 0, 255)
        
        self.dsps_log_message_71 = getUnsigned(packet[77:78], 'little', 0, 255)
        
        self.dsps_log_message_72 = getUnsigned(packet[78:79], 'little', 0, 255)
        
        self.dsps_log_message_73 = getUnsigned(packet[79:80], 'little', 0, 255)
        
        self.dsps_log_message_74 = getUnsigned(packet[80:81], 'little', 0, 255)
        
        self.dsps_log_message_75 = getUnsigned(packet[81:82], 'little', 0, 255)
        
        self.dsps_log_message_76 = getUnsigned(packet[82:83], 'little', 0, 255)
        
        self.dsps_log_message_77 = getUnsigned(packet[83:84], 'little', 0, 255)
        
        self.dsps_log_message_78 = getUnsigned(packet[84:85], 'little', 0, 255)
        
        self.dsps_log_message_79 = getUnsigned(packet[85:86], 'little', 0, 255)
        
        self.REUSABLE_SPARE_16 = getUnsigned(packet[86:88], 'little', 0, 65535)
        
        self.dsps_log_checksum = getUnsigned(packet[88:90], 'big', 0, 4294967295)

        try:
            self.pktTimestamp = self.SHCOARSE
        except:
            self.pktTimestamp = 0

        self.file_origin = file_origin

class DSPS_TIME:
    def __str__(self):
        return 'DSPS_TIME'

    def __init__(self, packet, header, file_origin):
        apidField = int.from_bytes(header[0:2], 'big')
        self.version = apidField >> 13
        self.type = apidField >> 12 & 0x01
        self.secHdr = apidField >> 11 & 0x01
        self.apid = apidField & 0x7FF
        self.src_seq_ctr = int.from_bytes(header[2:4], 'big') & 0x3FFF
        self.group = int.from_bytes(header[2:4], 'big') & 0xC000 >> 14
        self.length = int.from_bytes(header[4:6], 'big')
        if len(header) > 6:
            self.SHFINE = int.from_bytes(header[6:8], 'big')
            self.SHCOARSE = int.from_bytes(header[8:12], 'big')
            ertTime = self.SHCOARSE * 1000 + self.SHFINE
            self.ground_isotime = datetime.datetime.fromtimestamp(ertTime/1000.0).isoformat()
        
        self.ccsdsSecHeader2_sec_dsps_time = getUnsigned(packet[0:4], 'big', 0, 4294967295)
        
        self.ccsdsSecHeader2_sub_dsps_time = getUnsigned(packet[4:6], 'big', 0, 65535)
        
        self.dsps_millisec_since_turn_on = getUnsigned(packet[6:10], 'little', 0, 4294967295)
        
        self.REUSABLE_SPARE_16 = getUnsigned(packet[10:12], 'little', 0, 65535)
        
        self.dsps_time_checksum = getUnsigned(packet[12:14], 'big', 0, 4294967295)

        try:
            self.pktTimestamp = self.SHCOARSE
        except:
            self.pktTimestamp = 0

        self.file_origin = file_origin

class DSPS_PARAMETER_PKT:
    def __str__(self):
        return 'DSPS_PARAMETER_PKT'

    def __init__(self, packet, header, file_origin):
        apidField = int.from_bytes(header[0:2], 'big')
        self.version = apidField >> 13
        self.type = apidField >> 12 & 0x01
        self.secHdr = apidField >> 11 & 0x01
        self.apid = apidField & 0x7FF
        self.src_seq_ctr = int.from_bytes(header[2:4], 'big') & 0x3FFF
        self.group = int.from_bytes(header[2:4], 'big') & 0xC000 >> 14
        self.length = int.from_bytes(header[4:6], 'big')
        if len(header) > 6:
            self.SHFINE = int.from_bytes(header[6:8], 'big')
            self.SHCOARSE = int.from_bytes(header[8:12], 'big')
            ertTime = self.SHCOARSE * 1000 + self.SHFINE
            self.ground_isotime = datetime.datetime.fromtimestamp(ertTime/1000.0).isoformat()
        
        self.ccsdsSecHeader2_sec_dsps_parameter_pkt = getUnsigned(packet[0:4], 'big', 0, 4294967295)
        
        self.ccsdsSecHeader2_sub_dsps_parameter_pkt = getUnsigned(packet[4:6], 'big', 0, 65535)
        
        self.dsps_param_0_gps_write_time = getUnsigned(packet[6:10], 'little', 0, 4294967295)
        
        self.dsps_param_1_reboot_flag = getUnsigned(packet[10:14], 'little', 0, 4294967295)
        
        self.dsps_param_2_leap_seconds = getUnsigned(packet[14:18], 'little', 0, 4294967295)
        
        self.dsps_param_3_integration_period = getUnsigned(packet[18:22], 'little', 0, 4294967295)
        
        self.dsps_param_4_time_packet_cadence = getUnsigned(packet[22:26], 'little', 0, 4294967295)
        
        self.dsps_param_5_background_diode_1 = getUnsigned(packet[26:30], 'little', 0, 4294967295)
        
        self.dsps_param_6_background_diode_2 = getUnsigned(packet[30:34], 'little', 0, 4294967295)
        
        self.dsps_param_7_background_diode_3 = getUnsigned(packet[34:38], 'little', 0, 4294967295)
        
        self.dsps_param_8_background_diode_4 = getUnsigned(packet[38:42], 'little', 0, 4294967295)
        
        self.dsps_param_9_background_diode_5 = getUnsigned(packet[42:46], 'little', 0, 4294967295)
        
        self.dsps_param_10_background_diode_6 = getUnsigned(packet[46:50], 'little', 0, 4294967295)
        
        self.dsps_param_11_background_diode_7 = getUnsigned(packet[50:54], 'little', 0, 4294967295)
        
        self.dsps_param_12_background_diode_8 = getUnsigned(packet[54:58], 'little', 0, 4294967295)
        
        self.dsps_param_13_sps_1_x_fov_neg = getUnsigned(packet[58:62], 'little', 0, 4294967295)
        
        self.dsps_param_14_sps_1_y_fov_neg = getUnsigned(packet[62:66], 'little', 0, 4294967295)
        
        self.dsps_param_15_sps_2_x_fov_neg = getUnsigned(packet[66:70], 'little', 0, 4294967295)
        
        self.dsps_param_16_sps_2_Y_fov_neg = getUnsigned(packet[70:74], 'little', 0, 4294967295)
        
        self.dsps_param_17_sps_1_x_fov_POS = getUnsigned(packet[74:78], 'little', 0, 4294967295)
        
        self.dsps_param_18_sps_1_y_fov_POS = getUnsigned(packet[78:82], 'little', 0, 4294967295)
        
        self.dsps_param_19_sps_2_x_fov_POS = getUnsigned(packet[82:86], 'little', 0, 4294967295)
        
        self.dsps_param_20_sps_2_y_fov_POS = getUnsigned(packet[86:90], 'little', 0, 4294967295)
        
        self.dsps_param_21_sps_1_x_offset = getUnsigned(packet[90:94], 'little', 0, 4294967295)
        
        self.dsps_param_22_sps_1_y_offset = getUnsigned(packet[94:98], 'little', 0, 4294967295)
        
        self.dsps_param_23_sps_2_x_offset = getUnsigned(packet[98:102], 'little', 0, 4294967295)
        
        self.dsps_param_24_sps_2_y_offset = getUnsigned(packet[102:106], 'little', 0, 4294967295)
        
        self.dsps_param_25_sps_1_x_config = getUnsigned(packet[106:110], 'little', 0, 4294967295)
        
        self.dsps_param_26_sps_1_y_config = getUnsigned(packet[110:114], 'little', 0, 4294967295)
        
        self.dsps_param_27_sps_2_x_config = getUnsigned(packet[114:118], 'little', 0, 4294967295)
        
        self.dsps_param_28_sps_2_y_config = getUnsigned(packet[118:122], 'little', 0, 4294967295)
        
        self.dsps_param_29_sps_1_threshold = getUnsigned(packet[122:126], 'little', 0, 4294967295)
        
        self.dsps_param_30_sps_2_threshold = getUnsigned(packet[126:130], 'little', 0, 4294967295)
        
        try:
            self.dsps_param_31_adc_glitch = getFloat(packet[130:134], 'little')
        except:
            print('Could not decode floating point item "dsps_param_31_adc_glitch" in "dsps_parameter_pkt"')
            self.dsps_param_31_adc_glitch = 0
        
        self.dsps_param_32_debug_mode = getUnsigned(packet[134:138], 'little', 0, 4294967295)
        
        self.dsps_param_33_rtc_reset_period = getUnsigned(packet[138:142], 'little', 0, 4294967295)
        
        self.dsps_param_34_flare_channel = getUnsigned(packet[142:146], 'little', 0, 4294967295)
        try:
            self.dsps_param_34_flare_channel = dsps_states.CONVERT_dsps_param_34_flare_channel(self.dsps_param_34_flare_channel)
        except:
            pass
        
        try:
            self.dsps_param_35_goes_const = getFloat(packet[146:150], 'little')
        except:
            print('Could not decode floating point item "dsps_param_35_goes_const" in "dsps_parameter_pkt"')
            self.dsps_param_35_goes_const = 0
        
        try:
            self.dsps_param_36_goes_slope = getFloat(packet[150:154], 'little')
        except:
            print('Could not decode floating point item "dsps_param_36_goes_slope" in "dsps_parameter_pkt"')
            self.dsps_param_36_goes_slope = 0
        
        try:
            self.dsps_param_37_flare_slope = getFloat(packet[154:158], 'little')
        except:
            print('Could not decode floating point item "dsps_param_37_flare_slope" in "dsps_parameter_pkt"')
            self.dsps_param_37_flare_slope = 0
        
        self.dsps_param_38_flare_time = getUnsigned(packet[158:162], 'little', 0, 4294967295)
        
        try:
            self.dsps_param_39_flare_level = getFloat(packet[162:166], 'little')
        except:
            print('Could not decode floating point item "dsps_param_39_flare_level" in "dsps_parameter_pkt"')
            self.dsps_param_39_flare_level = 0
        
        self.dsps_param_40_gps_write_time = getUnsigned(packet[166:170], 'little', 0, 4294967295)
        
        try:
            self.dsps_param_41_reboot_flag = getFloat(packet[170:174], 'little')
        except:
            print('Could not decode floating point item "dsps_param_41_reboot_flag" in "dsps_parameter_pkt"')
            self.dsps_param_41_reboot_flag = 0
        
        self.REUSABLE_SPARE_32 = getUnsigned(packet[174:178], 'little', 0, 4294967295)
        
        self.REUSABLE_SPARE_32 = getUnsigned(packet[178:182], 'little', 0, 4294967295)
        
        self.REUSABLE_SPARE_32 = getUnsigned(packet[182:186], 'little', 0, 4294967295)
        
        self.REUSABLE_SPARE_32 = getUnsigned(packet[186:190], 'little', 0, 4294967295)
        
        self.REUSABLE_SPARE_32 = getUnsigned(packet[190:194], 'little', 0, 4294967295)
        
        self.dsps_param_47_parameter_checksum = getUnsigned(packet[194:198], 'little', 0, 4294967295)
        
        self.dsps_parameter_pkt_checksum = getUnsigned(packet[198:200], 'big', 0, 4294967295)

        try:
            self.pktTimestamp = self.SHCOARSE
        except:
            self.pktTimestamp = 0

        self.file_origin = file_origin
