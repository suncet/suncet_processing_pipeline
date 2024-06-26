# Copyright 2017, 2018, 2019, 2020 Daniel Estevez <daniel@destevez.net>
# Copyright 2022 The Regents of the University of Colorado
# Adapted by James Mason in 2024

from satellites.telemetry.ax25 import Header
from satellites.ccsds import space_packet as ccsds_space_packet
from satellites.adapters import PolynomialAdapter
from construct import Adapter, BitsInteger, BitStruct, Bytes, Container, \
                Default, Enum, Flag, Float32l, Hex, If, Int8ub, Int8sl, \
                Int8ul, Int16sl, Int16ul, Int16sb, Int16ub, Int32ub, \
                Int32ul, Int32sl, Int32sb, Padding, Struct, Switch

SSID = BitStruct(
    'ch' / Flag,  # C / H bit
    Default(BitsInteger(2), 3),  # reserved bits
    'ssid' / BitsInteger(4),
    'extension' / Flag  # last address bit
    )


class CallsignAdapter(Adapter):
    def _encode(self, obj, context, path=None):
        return bytes([x << 1 for x in bytes(
            (obj.upper() + ' '*6)[:6], encoding='ascii')])

    def _decode(self, obj, context, path=None):
        return str(bytes([x >> 1 for x in obj]), encoding='ascii').strip()


Callsign = CallsignAdapter(Bytes(6))
CallsignUnshift = Bytes(6)
Control = Hex(Int8ub)
PID = Hex(Int8ub)

Address = Struct(
        'callsign' / Callsign,
        'ssid' / SSID
        )

UnshiftAddress = Struct(
        'callsign' / CallsignUnshift,
        'ssid' / SSID
        )

Header = Struct(
    'dest' / Address,
    'src' / UnshiftAddress,
    'control' / Control,
    'pid' / PID
    )

SecondaryHeader = Struct(
    'sh_coarse' / Int32ul,
    'sh_fine' / Int16ub,
)

enum_fields = {f"OPCODE_{i}": i for i in range(250)}
tmp_opcodes = Enum(Int16ub, **enum_fields)


seq_opcodes = Enum(
        Int16ub,
        RAM_SMALL0=0, RAM_SMALL1=1, RAM_SMALL2=2,
        RAM_SMALL3=3, RAM_SMALL4=4, RAM_SMALL5=5,
        RAM_SMALL6=6, RAM_SMALL7=7, RAM_LARGE0=8, RAM_LARGE1=9,
        RAM_LARGE2=10, RAM_LARGE3=11, NVM_SMALL0=12, NVM_SMALL1=13,
        NVM_SMALL2=14, NVM_SMALL3=15, NVM_SMALL4=16, NVM_SMALL5=17,
        NVM_SMALL6=18, NVM_SMALL7=19, NVM_SMALL8=20, NVM_SMALL9=21,
        NVM_SMALL10=22, NVM_SMALL11=23, NVM_SMALL12=24, NVM_SMALL13=25,
        NVM_SMALL14=26, NVM_SMALL15=27, NVM_SMALL16=28,
        NVM_SMALL17=29, NVM_SMALL18=30, NVM_SMALL19=31, NVM_SMALL20=32,
        NVM_SMALL21=33, NVM_SMALL22=34, NVM_SMALL23=35,
        NVM_SMALL24=36, NVM_SMALL25=37, NVM_SMALL26=38, NVM_SMALL27=39,
        NVM_SMALL28=40, NVM_SMALL29=41, NVM_SMALL30=42,
        NVM_SMALL31=43, NVM_SMALL32=44, NVM_SMALL33=45, NVM_SMALL34=46,
        NVM_SMALL35=47, NVM_SMALL36=48, NVM_SMALL37=49,
        NVM_SMALL38=50, NVM_SMALL39=51, NVM_SMALL40=52, NVM_SMALL41=53,
        NVM_SMALL42=54, NVM_SMALL43=55, NVM_SMALL44=56,
        NVM_SMALL45=57, NVM_SMALL46=58, NVM_SMALL47=59, NVM_SMALL48=60,
        NVM_SMALL49=61, NVM_SMALL50=62, NVM_SMALL51=63,
        NVM_SMALL52=64, NVM_SMALL53=65, NVM_SMALL54=66, NVM_SMALL55=67,
        NVM_SMALL56=68, NVM_SMALL57=69, NVM_SMALL58=70,
        NVM_SMALL59=71, NVM_SMALL60=72, NVM_SMALL61=73, NVM_SMALL62=74,
        NVM_SMALL63=75, NVM_SMALL64=76, NVM_SMALL65=77,
        NVM_SMALL66=78, NVM_SMALL67=79, NVM_SMALL68=80, NVM_SMALL69=81,
        NVM_SMALL70=82, NVM_SMALL71=83, NVM_SMALL72=84,
        NVM_SMALL73=85, NVM_SMALL74=86, NVM_SMALL75=87, NVM_SMALL76=88,
        NVM_SMALL77=89, NVM_SMALL78=90, NVM_SMALL79=91,
        NVM_SMALL80=92, NVM_SMALL81=93, NVM_SMALL82=94, NVM_SMALL83=95,
        NVM_SMALL84=96, NVM_SMALL85=97, NVM_SMALL86=98,
        NVM_SMALL87=99, NVM_SMALL88=100, NVM_SMALL89=101, NVM_SMALL90=102,
        NVM_SMALL91=103, NVM_SMALL92=104, NVM_SMALL93=105,
        NVM_SMALL94=106, NVM_SMALL95=107, NVM_SMALL96=108,
        NVM_SMALL97=109, NVM_SMALL98=110, NVM_SMALL99=111,
        NVM_SMALL100=112, NVM_SMALL101=113, NVM_SMALL102=114,
        NVM_SMALL103=115, NVM_SMALL104=116, NVM_SMALL105=117,
        NVM_SMALL106=118, NVM_SMALL107=119, NVM_SMALL108=120,
        NVM_SMALL109=121, NVM_SMALL110=122, NVM_SMALL111=123,
        NVM_SMALL112=124, NVM_SMALL113=125, NVM_SMALL114=126,
        NVM_SMALL115=127, NVM_SMALL116=128, NVM_SMALL117=129,
        NVM_SMALL118=130, NVM_SMALL119=131, NVM_SMALL120=132,
        NVM_SMALL121=133, NVM_SMALL122=134, NVM_SMALL123=135,
        NVM_SMALL124=136, NVM_SMALL125=137, NVM_SMALL126=138,
        NVM_SMALL127=139, HOLDING0=140)

alive_arm_state = BitStruct(
    'reusable_spare_1' / BitsInteger(1),
    'reusable_spare_2' / BitsInteger(1),
    'reusable_spare_3' / BitsInteger(1),
    'reusable_spare_4' / BitsInteger(1),
    'reusable_spare_5' / BitsInteger(1),
    'sw_cmd_arm_state_uhf' / Enum(BitsInteger(1), OFF=0, ARMED=1),
    'sw_cmd_arm_state_seq' / Enum(BitsInteger(1), OFF=0, ARMED=1),
    'sw_cmd_arm_state_ext' / Enum(BitsInteger(1), OFF=0, ARMED=1),
    'reusable_spare_6' / BitsInteger(1),
    'sw_eps_pwr_state_deploy_pwr' / Enum(BitsInteger(1), OFF=0, ON=1),
    'sw_eps_pwr_state_deploy' / Enum(BitsInteger(1), OFF=0, ON=1),
    'sw_eps_pwr_state_iic' / Enum(BitsInteger(1), OFF=0, ON=1),
    'sw_eps_pwr_state_inst' / Enum(BitsInteger(1), OFF=0, ON=1),
    'sw_eps_pwr_state_sband' / Enum(BitsInteger(1), OFF=0, ON=1),
    'sw_eps_pwr_state_uhf' / Enum(BitsInteger(1), OFF=0, ON=1),
    'sw_eps_pwr_state_adcs' / Enum(BitsInteger(1), OFF=0, ON=1),
    'reusable_spare_7' / BitsInteger(1),
    'reusable_spare_8' / BitsInteger(1),
    'reusable_spare_9' / BitsInteger(1),
    'reusable_spare_10' / BitsInteger(1),
    'reusable_spare_11' / BitsInteger(1),
    'reusable_spare_12' / BitsInteger(1),
    'sw_bat_alive_state_battery1' / Enum(BitsInteger(1), DEAD=0, ALIVE=1),
    'sw_bat_alive_state_battery0' / Enum(BitsInteger(1), DEAD=0, ALIVE=1)
)

payload_state = BitStruct(
    'spare_2' / BitsInteger(2),
    'sw_payload_pwr_cycle_req' / Enum(BitsInteger(1), INACTIVE=0, ACTIVE=1),
    'sw_payload_pwr_off_req' / Enum(BitsInteger(1), INACTIVE=0, ACTIVE=1),
    'sw_payload_stat_msg_state' / Enum(BitsInteger(1), DIS=0, ENA=1),
    'sw_payload_time_msg_state' / Enum(BitsInteger(1), DIS=0, ENA=1),
    'sw_payload_alive_state' / Enum(BitsInteger(2), OFF=0, DEAD=1, ALIVE=2)
)

shutter_state = BitStruct(
    'sw_shutter_state_b4_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_b3_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_b2_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_b1_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_a4_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_a3_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_a2_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1),
    'sw_shutter_state_a1_suncet' / Enum(BitsInteger(1), OPEN=0, CLOSE=1)
)

adcs_info = BitStruct(
    'sw_adcs_att_valid' / Enum(BitsInteger(1), YES=1, NO=0),
    'sw_adcs_ref_valid' / Enum(BitsInteger(1), YES=1, NO=0),
    'sw_adcs_time_valid' / Enum(BitsInteger(1), YES=1, NO=0),
    'sw_adcs_mode' / Enum(BitsInteger(1), FINE_REF_POINT=1, SUN_POINT=0),
    'sw_adcs_recommend_sun_point' / Enum(BitsInteger(1), YES=1, NO=0),
    'sw_adcs_sun_point_state' / Enum(BitsInteger(3), SEARCHING=3, WAITING=4,
                                     CONVERGING=5, NOT_ACTIVE=7, ON_SUN=6,
                                     SEARCH_INIT=2)
)

suncet_beacon = Struct(
    'sw_major_version' / Int8ul,
    'sw_minor_version' / Int8ul,
    'sw_patch_version' / Int8ul,
    'sw_image_id' / Int8ul,
    'sw_cmd_recv_count' / Int16ub,
    'sw_cmd_fmt_count' / Int16ub,
    'sw_cmd_rjct_count' / Int16ub,
    'sw_cmd_succ_count' / Int16ub,
    'sw_cmd_succ_op' / tmp_opcodes,
    'sw_cmd_rjct_op' / tmp_opcodes,
    'sw_cmd_fail_code' / Enum(Int8ul, SUCCESS=0, MODE=1, ARM=2,
                              SOURCE=3, OPCODE=4, METHOD=5,
                              LENGTH=6, RANGE=7, CHECKSUM=8,
                              PKT_TYPE=9),
    'sw_cmd_xsum_state' / Enum(Int8ul, DIS=0, ENA=1),
    'alive_arm_state' / alive_arm_state,
    'sw_mode_clt_count' / Int8ul,
    'sw_mode_system_mode' / Enum(Int8ul, PHOENIX=0, SAFE=1, NOMINAL=2),
    'sw_sband_sync_state' / Enum(Int8ul, DIS=0, ENA=1),
    'sw_time_since_rx' / Int16ub,
    'sw_sband_timeout' / Int16ub,
    'payload_state' / payload_state,
    'sw_uhf_alive' / Int8ul,
    'sw_uhf_temp' / Int8sl,
    'sw_adcs_alive' / Enum(Int8ul, OFF=0, DEAD=1, ALIVE=2),
    'sw_inst_cmd_succ_count_suncet' / Int16ub,
    'sw_inst_cmd_rjct_count_suncet' / Int8ul,
    'sw_esr_obs_id_suncet' / Int8ul,
    'sw_thrm_a1_a_suncet' / Int16ub,
    'sw_thrm_a1_b_suncet' / Int16ub,
    'sw_fss_q1_suncet' / Int16ub,
    'sw_fss_q2_suncet' / Int16ub,
    'sw_fss_q3_suncet' / Int16ub,
    'sw_fss_q4_suncet' / Int16ub,
    'sw_volt_p12v_suncet' / Int16ub,
    'sw_thrm_pwm_suncet' / Int16ub,
    'sw_inst_fp_resp_count_suncet' / Int16ub,
    'shutter_state' / shutter_state,
    'sw_inst_cmd_fail_code_suncet' / Enum(Int8ul, SUCCESS=0, MODE=1,
                                        ARM=2, SOURCE=3, OPCODE=4,
                                        METHOD=5, LENGTH=6, RANGE=7,
                                        CHECKSUM=8, PKT_TYPE=9),
    'sw_esr_filtered_a12_suncet' / Float32l,
    'sw_esr_filtered_b12_suncet' / Float32l,
    'sw_seq_state_auto' / Enum(Int8ul, IDLE=0, ACTIVE=1,
                               SUSPEND=2, PAUSE=3, STALE=4),
    'sw_seq_state_op1' / Enum(Int8ul, IDLE=0, ACTIVE=1,
                              SUSPEND=2, PAUSE=3, STALE=4),
    'sw_seq_state_op2' / Enum(Int8ul, IDLE=0, ACTIVE=1,
                              SUSPEND=2, PAUSE=3, STALE=4),
    'sw_seq_state_op3' / Enum(Int8ul, IDLE=0, ACTIVE=1,
                              SUSPEND=2, PAUSE=3, STALE=4),
    'sw_seq_stop_code_auto' / Enum(Int8ul, NOMINAL=0, CMD=1, INIT=2,
                                   REJECT=3, STALE=4, BAD_INT=5,
                                   INT_FAIL=6),
    'sw_seq_stop_code_op1' / Enum(Int8ul, NOMINAL=0, CMD=1, INIT=2,
                                  REJECT=3, STALE=4, BAD_INT=5,
                                  INT_FAIL=6),
    'sw_seq_stop_code_op2' / Enum(Int8ul, NOMINAL=0, CMD=1, INIT=2,
                                  REJECT=3, STALE=4, BAD_INT=5,
                                  INT_FAIL=6),
    'sw_seq_stop_code_op3' / Enum(Int8ul, NOMINAL=0, CMD=1, INIT=2,
                                  REJECT=3, STALE=4, BAD_INT=5,
                                  INT_FAIL=6),
    'sw_seq_exec_buf_auto' / seq_opcodes,
    'sw_seq_exec_buf_op1' / seq_opcodes,
    'sw_seq_exec_buf_op2' / seq_opcodes,
    'sw_seq_exec_buf_op3' / seq_opcodes,
    'sw_store_partition_write_misc' / Int32ub,
    'sw_store_partition_read_misc' / Int32ub,
    'sw_store_partition_write_adcs' / Int32ub,
    'sw_store_partition_read_adcs' / Int32ub,
    'sw_store_partition_write_beacon' / Int32ub,
    'sw_store_partition_read_beacon' / Int32ub,
    'sw_store_partition_write_log' / Int32ub,
    'sw_store_partition_read_log' / Int32ub,
    'sw_store_partition_write_payload' / Int32ub,
    'sw_store_partition_read_payload' / Int32ub,
    'sw_fp_resp_count' / Int16ub,
    'sw_ana_bus_v' / PolynomialAdapter([0.0, 0.008862], Int16ub),
    'sw_ana_3p3_v' / PolynomialAdapter([0.0, 0.001611], Int16ub),
    'sw_ana_3p3_i' / PolynomialAdapter([0.0, 8.1e-05], Int16ub),
    'sw_ana_1p8_i' / PolynomialAdapter([0.0, 8.1e-05], Int16ub),
    'sw_ana_1p0_i' / PolynomialAdapter([0.0, 8.1e-05], Int16ub),
    'sw_ana_cdh_temp' / PolynomialAdapter([98.0, -0.05936, 1.641e-05,
                                          -0.000000002361], Int16ub),
    'sw_ana_sa1_v' / PolynomialAdapter([0.0, 0.009659], Int16ub),
    'sw_ana_sa1_i' / PolynomialAdapter([-0.04241, 0.002525], Int16ub),
    'sw_ana_sa2_v' / PolynomialAdapter([0.0, 0.009659], Int16ub),
    'sw_ana_sa2_i' / PolynomialAdapter([-0.04262, 0.002525], Int16ub),
    'sw_ana_bat1_v' / PolynomialAdapter([0.0, 0.008862], Int16ub),
    'sw_ana_bat2_v' / PolynomialAdapter([0.0, 0.008862], Int16ub),
    'sw_ana_eps_temp' / PolynomialAdapter([98.0, -0.05936, 1.641e-05,
                                          -0.000000002361], Int16ub),
    'sw_ana_eps_3p3_ref' / PolynomialAdapter([0.0, 0.001611], Int16ub),
    'sw_ana_eps_bus_i' / PolynomialAdapter([0.0, 0.001221], Int16ub),
    'sw_ana_xact_i' / PolynomialAdapter([-0.02489, 0.001992], Int16ub),
    'sw_ana_uhf_i' / PolynomialAdapter([-0.02489, 0.001992], Int16ub),
    'sw_ana_sband_i' / PolynomialAdapter([-0.02489, 0.001992], Int16ub),
    'sw_ana_inst_i' / PolynomialAdapter([0.0, 0.000807], Int16ub),
    'sw_ana_hk_3p3_ref' / PolynomialAdapter([0.0, 0.001611], Int16ub),
    'sw_ana_ifb_i' / PolynomialAdapter([0.0, 0.000269], Int16ub),
    'sw_ana_ifb_temp' / PolynomialAdapter([98.0, -0.05936, 1.641e-05,
                                          -0.000000002361], Int16ub),
    'sw_adcs_eclipse' / Int8ul,
    'adcs_info' / adcs_info,
    'sw_adcs_analogs_voltage_2p5' / PolynomialAdapter([0.0, 0.015], Int8ul),
    'sw_adcs_analogs_voltage_1p8' / PolynomialAdapter([0.0, 0.015], Int8ul),
    'sw_adcs_analogs_voltage_1p0' / PolynomialAdapter([0.0, 0.015], Int8ul),
    'sw_adcs_analogs_det_temp' / PolynomialAdapter([0.0, 0.8], Int8sl),
    'sw_adcs_analogs_motor1_temp' / PolynomialAdapter([0.0, 0.005], Int16sb),
    'sw_adcs_analogs_motor2_temp' / PolynomialAdapter([0.0, 0.005], Int16sb),
    'sw_adcs_analogs_motor3_temp' / PolynomialAdapter([0.0, 0.005], Int16sb),
    'spare_16' / Int16sl,
    'sw_adcs_analogs_digital_bus_v' / PolynomialAdapter([0.0, 0.00125],
                                                        Int16sb),
    'sw_adcs_cmd_acpt' / Int8ul,
    'sw_adcs_cmd_fail' / Int8ul,
    'sw_adcs_time' / Int32sl,
    'sw_adcs_sun_vec1' / PolynomialAdapter([0.0, 0.0001], Int16sb),
    'sw_adcs_sun_vec2' / PolynomialAdapter([0.0, 0.0001], Int16sb),
    'sw_adcs_sun_vec3' / PolynomialAdapter([0.0, 0.0001], Int16sb),
    'sw_adcs_wheel_sp1' / PolynomialAdapter([0.0, 0.4], Int16sb),
    'sw_adcs_wheel_sp2' / PolynomialAdapter([0.0, 0.4], Int16sb),
    'sw_adcs_wheel_sp3' / PolynomialAdapter([0.0, 0.4], Int16sb),
    'sw_adcs_body_rt1' / PolynomialAdapter([0.0, 5E-9], Int32sb),
    'sw_adcs_body_rt2' / PolynomialAdapter([0.0, 5E-9], Int32sb),
    'sw_adcs_body_rt3' / PolynomialAdapter([0.0, 5E-9], Int32sb),
    'sw_adcs_body_quat1' / PolynomialAdapter([0.0, 5E-10], Int32sb),
    'sw_adcs_body_quat2' / PolynomialAdapter([0.0, 5E-10], Int32sb),
    'sw_adcs_body_quat3' / PolynomialAdapter([0.0, 5E-10], Int32sb),
    'sw_adcs_body_quat4' / PolynomialAdapter([0.0, 5E-10], Int32sb),
    'sw_spare_0' / Int32ul,
    'sw_spare_1' / Int32ul,
    'sw_spare_2' / Int32ul,
    'sw_spare_3' / Int32ul
)


packet = Struct(
    'ax25_header' / Header,
    'primary_header' / ccsds_space_packet.PrimaryHeader,
    'secondary_header' / If(
        lambda c: c.primary_header.secondary_header_flag,
        SecondaryHeader
    ),
    'packet' / Switch(
        lambda c: c.primary_header.AP_ID,
        {
            0x01: suncet_beacon
        }
    )
)