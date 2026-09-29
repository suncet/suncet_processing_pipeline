# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild
# type: ignore

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import IntEnum


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 11):
    raise Exception("Incompatible Kaitai Struct Python API: 0.11 or later is required, but you have %s" % (kaitaistruct.__version__))

class SuncetApid1(KaitaiStruct):
    """Bare-CCSDS decoder for the CTDB 2.0.5 public SunCET APID 1 beacon.
    The AX.25 wrapper awaits validation of the SatNOGS receiver output boundary.
    Use decode_public_beacon for Fletcher-32 validation and public-only output.

    :field ccsds_version: ccsds_version
    :field ccsds_packet_type: ccsds_packet_type
    :field ccsds_secondary_header_flag: ccsds_secondary_header_flag
    :field ccsds_apid: ccsds_apid
    :field ccsds_sequence_flags: ccsds_sequence_flags
    :field ccsds_sequence_count: ccsds_sequence_count
    :field ccsds_packet_length_field: ccsds_packet_length_field
    :field spacecraft_time_seconds_since_2000: spacecraft_time_seconds_since_2000
    :field spacecraft_time_milliseconds: spacecraft_time_milliseconds
    :field partition_write_adcs: partition_write_adcs
    :field partition_read_adcs: partition_read_adcs
    :field partition_write_hk: partition_write_hk
    :field partition_read_hk: partition_read_hk
    :field partition_write_sci: partition_write_sci
    :field partition_read_sci: partition_read_sci
    :field partition_write_dsps: partition_write_dsps
    :field partition_read_dsps: partition_read_dsps
    :field store_partition_write_log: store_partition_write_log
    :field csie_nand_sci_write_ptr: csie_nand_sci_write_ptr
    :field time_since_boot: time_since_boot
    :field time_alive: time_alive
    :field time_mission_elapsed_time: time_mission_elapsed_time
    :field mode_seconds_since_mode_change: mode_seconds_since_mode_change
    :field dsps_flare_level: dsps_flare_level
    :field adcs_body_rate_1: adcs_body_rate_1
    :field adcs_body_rate_2: adcs_body_rate_2
    :field adcs_body_rate_3: adcs_body_rate_3
    :field csie_img_hist_0: csie_img_hist_0
    :field csie_img_hist_1: csie_img_hist_1
    :field csie_img_hist_2: csie_img_hist_2
    :field csie_img_hist_3: csie_img_hist_3
    :field csie_img_hist_4: csie_img_hist_4
    :field csie_img_hist_5: csie_img_hist_5
    :field adcs_wheel_speed_1: adcs_wheel_speed_1
    :field adcs_wheel_speed_2: adcs_wheel_speed_2
    :field adcs_wheel_speed_3: adcs_wheel_speed_3
    :field xband_pa_temp: xband_pa_temp
    :field xband_pa_current: xband_pa_current
    :field rail_3v3_voltage: rail_3v3_voltage
    :field rail_3v3_current: rail_3v3_current
    :field cdh_temp: cdh_temp
    :field cdh_3v3_reference_voltage: cdh_3v3_reference_voltage
    :field solar_array_8_cell_string_voltage: solar_array_8_cell_string_voltage
    :field solar_array_8_cell_string_current: solar_array_8_cell_string_current
    :field solar_array_9_cell_string_voltage: solar_array_9_cell_string_voltage
    :field solar_array_9_cell_string_current: solar_array_9_cell_string_current
    :field battery_1_voltage: battery_1_voltage
    :field battery_2_voltage: battery_2_voltage
    :field eps_temp: eps_temp
    :field eps_3v3_reference_voltage: eps_3v3_reference_voltage
    :field eps_bus_voltage: eps_bus_voltage
    :field eps_bus_current: eps_bus_current
    :field xact_voltage: xact_voltage
    :field xact_current: xact_current
    :field uhf_voltage: uhf_voltage
    :field uhf_current: uhf_current
    :field xband_voltage: xband_voltage
    :field xband_current: xband_current
    :field csie_voltage: csie_voltage
    :field csie_current: csie_current
    :field dsps_voltage: dsps_voltage
    :field dsps_current: dsps_current
    :field ifb_therm1: ifb_therm1
    :field sa_minus_y_temp: sa_minus_y_temp
    :field sa_plus_y_temp: sa_plus_y_temp
    :field csie_temp: csie_temp
    :field battery_1_temp: battery_1_temp
    :field batt_board_temp: batt_board_temp
    :field battery_1_charge_current: battery_1_charge_current
    :field battery_2_charge_current: battery_2_charge_current
    :field dsps_visible_sps_sun_pos_x: dsps_visible_sps_sun_pos_x
    :field dsps_visible_sps_sun_pos_y: dsps_visible_sps_sun_pos_y
    :field dsps_x_ray_sps_sun_pos_x: dsps_x_ray_sps_sun_pos_x
    :field dsps_x_ray_sps_sun_pos_y: dsps_x_ray_sps_sun_pos_y
    :field dsps_sensor_board_temp: dsps_sensor_board_temp
    :field adcs_ana_motor1_temp: adcs_ana_motor1_temp
    :field adcs_sun_point_angle_error: adcs_sun_point_angle_error
    :field num_sc_resets: num_sc_resets
    :field csie_capture_state: csie_capture_state
    :field fault_protection_watchpoint_6_state: fault_protection_watchpoint_6_state
    :field fault_protection_watchpoint_5_state: fault_protection_watchpoint_5_state
    :field fault_protection_watchpoint_4_state: fault_protection_watchpoint_4_state
    :field fault_protection_watchpoint_3_state: fault_protection_watchpoint_3_state
    :field fault_protection_watchpoint_2_state: fault_protection_watchpoint_2_state
    :field fault_protection_watchpoint_1_state: fault_protection_watchpoint_1_state
    :field fault_protection_watchpoint_0_state: fault_protection_watchpoint_0_state
    :field clt_hours_until_reboot: clt_hours_until_reboot
    :field mode_system_mode: mode_system_mode
    :field uhf_temp: uhf_temp
    :field fault_protection_task_state: fault_protection_task_state
    :field dsps_flare_magnitude: dsps_flare_magnitude
    :field dsps_flare_phase: dsps_flare_phase
    :field battery_1_charging_state: battery_1_charging_state
    :field eps_pwr_state_dsps: eps_pwr_state_dsps
    :field eps_pwr_state_csie: eps_pwr_state_csie
    :field eps_pwr_state_xband: eps_pwr_state_xband
    :field eps_pwr_state_uhf: eps_pwr_state_uhf
    :field eps_pwr_state_adcs: eps_pwr_state_adcs
    :field telescope_door_pin_pulled: telescope_door_pin_pulled
    :field csie_heater_enable: csie_heater_enable
    :field battery2_heater_enable: battery2_heater_enable
    :field battery1_heater_enable: battery1_heater_enable
    :field uhf_alive: uhf_alive
    :field adcs_alive: adcs_alive
    :field adcs_att_valid: adcs_att_valid
    :field adcs_ref_valid: adcs_ref_valid
    :field adcs_time_valid: adcs_time_valid
    :field adcs_mode: adcs_mode
    :field battery_2_charging_state: battery_2_charging_state
    :field adcs_sun_point_state: adcs_sun_point_state
    :field xband_data_source: xband_data_source

    .. seealso::
       Source - https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md
    """

    class AdcsAttValidValues(IntEnum):
        no = 0
        yes = 1

    class AdcsModeValues(IntEnum):
        sun_point = 0
        fine_ref_point = 1

    class AdcsRefValidValues(IntEnum):
        no = 0
        yes = 1

    class AdcsSunPointStateValues(IntEnum):
        sun_point = 0
        fine_ref_point = 1
        search_init = 2
        searching = 3
        waiting = 4
        converging = 5
        on_sun = 6
        not_actv = 7

    class AdcsTimeValidValues(IntEnum):
        no = 0
        yes = 1

    class Battery1HeaterEnableValues(IntEnum):
        no = 0
        yes = 1

    class Battery2HeaterEnableValues(IntEnum):
        no = 0
        yes = 1

    class Battery1ChargingStateValues(IntEnum):
        discharging = 0
        charging = 1

    class Battery2ChargingStateValues(IntEnum):
        discharging = 0
        charging = 1

    class CsieHeaterEnableValues(IntEnum):
        no = 0
        yes = 1

    class DspsFlarePhaseValues(IntEnum):
        not_in_sun = 0
        filling_history = 1
        not_in_flare = 2
        flare_start = 4
        declining_flare = 24
        rising_flare = 40

    class EpsPwrStateAdcsValues(IntEnum):
        off = 0
        on = 1

    class EpsPwrStateCsieValues(IntEnum):
        off = 0
        on = 1

    class EpsPwrStateDspsValues(IntEnum):
        off = 0
        on = 1

    class EpsPwrStateUhfValues(IntEnum):
        off = 0
        on = 1

    class EpsPwrStateXbandValues(IntEnum):
        off = 0
        on = 1

    class FaultProtectionTaskStateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint0StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint1StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint2StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint3StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint4StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint5StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class FaultProtectionWatchpoint6StateValues(IntEnum):
        disabled = 0
        passive = 1
        enabled = 2

    class ModeSystemModeValues(IntEnum):
        phoenix = 0
        safe = 1
        science = 2
        downlink = 3

    class TelescopeDoorPinPulledValues(IntEnum):
        pulled = 0
        engaged = 1

    class UhfAliveValues(IntEnum):
        off = 0
        alive = 1
        dead = 2

    class XbandDataSourceValues(IntEnum):
        test_pat = 0
        cdh = 1
    def __init__(self, _io, _parent=None, _root=None):
        super(SuncetApid1, self).__init__(_io)
        self._parent = _parent
        self._root = _root or self
        self._read()

    def _read(self):
        self.ccsds_primary_word = self._io.read_u2be()
        if not self.ccsds_primary_word == 2049:
            raise kaitaistruct.ValidationNotEqualError(2049, self.ccsds_primary_word, self._io, u"/seq/0")
        self.ccsds_sequence_word = self._io.read_u2be()
        self.ccsds_packet_length_field = self._io.read_u2be()
        _ = self.ccsds_packet_length_field
        if not  ((_ == 245) and (self._io.size() == 252)) :
            raise kaitaistruct.ValidationExprError(self.ccsds_packet_length_field, self._io, u"/seq/2")
        self.spacecraft_time_seconds_since_2000 = self._io.read_u4be()
        self.spacecraft_time_milliseconds = self._io.read_u2be()
        if not self.spacecraft_time_milliseconds <= 999:
            raise kaitaistruct.ValidationGreaterThanError(999, self.spacecraft_time_milliseconds, self._io, u"/seq/4")
        self.partition_write_adcs = self._io.read_u4be()
        self.partition_read_adcs = self._io.read_u4be()
        self.partition_write_hk = self._io.read_u4be()
        self.partition_read_hk = self._io.read_u4be()
        self.partition_write_sci = self._io.read_u4be()
        self.partition_read_sci = self._io.read_u4be()
        self.partition_write_dsps = self._io.read_u4be()
        self.partition_read_dsps = self._io.read_u4be()
        self.store_partition_write_log = self._io.read_u4be()
        self.csie_nand_sci_write_ptr = self._io.read_u4be()
        self.time_since_boot = self._io.read_u4be()
        self.time_alive = self._io.read_u4be()
        self.time_mission_elapsed_time = self._io.read_u4be()
        self.mode_seconds_since_mode_change = self._io.read_u4be()
        self.dsps_flare_level = self._io.read_f4be()
        self.adcs_body_rate_1_raw = self._io.read_s4be()
        self.adcs_body_rate_2_raw = self._io.read_s4be()
        self.adcs_body_rate_3_raw = self._io.read_s4be()
        self.csie_img_hist_0 = self._io.read_u4be()
        self.csie_img_hist_1 = self._io.read_u4be()
        self.csie_img_hist_2 = self._io.read_u4be()
        self.csie_img_hist_3 = self._io.read_u4be()
        self.csie_img_hist_4 = self._io.read_u4be()
        self.csie_img_hist_5 = self._io.read_u4be()
        self.adcs_wheel_speed_1_raw = self._io.read_s4be()
        self.adcs_wheel_speed_2_raw = self._io.read_s4be()
        self.adcs_wheel_speed_3_raw = self._io.read_s4be()
        self.xband_pa_temp_raw = self._io.read_s4be()
        self.opaque_1_bytes = self._io.read_bytes(4)
        self.xband_pa_current_raw = self._io.read_u2be()
        self.opaque_2_bytes = self._io.read_bytes(10)
        self.rail_3v3_voltage_raw = self._io.read_u2be()
        self.rail_3v3_current_raw = self._io.read_u2be()
        self.cdh_temp_raw = self._io.read_u2be()
        self.cdh_3v3_reference_voltage_raw = self._io.read_u2be()
        self.solar_array_8_cell_string_voltage_raw = self._io.read_u2be()
        self.solar_array_8_cell_string_current_raw = self._io.read_u2be()
        self.solar_array_9_cell_string_voltage_raw = self._io.read_u2be()
        self.solar_array_9_cell_string_current_raw = self._io.read_u2be()
        self.battery_1_voltage_raw = self._io.read_u2be()
        self.battery_2_voltage_raw = self._io.read_u2be()
        self.eps_temp_raw = self._io.read_u2be()
        self.eps_3v3_reference_voltage_raw = self._io.read_u2be()
        self.eps_bus_voltage_raw = self._io.read_u2be()
        self.eps_bus_current_raw = self._io.read_u2be()
        self.xact_voltage_raw = self._io.read_u2be()
        self.xact_current_raw = self._io.read_u2be()
        self.uhf_voltage_raw = self._io.read_u2be()
        self.uhf_current_raw = self._io.read_u2be()
        self.xband_voltage_raw = self._io.read_u2be()
        self.xband_current_raw = self._io.read_u2be()
        self.csie_voltage_raw = self._io.read_u2be()
        self.csie_current_raw = self._io.read_u2be()
        self.dsps_voltage_raw = self._io.read_u2be()
        self.dsps_current_raw = self._io.read_u2be()
        self.ifb_therm1_raw = self._io.read_u2be()
        self.sa_minus_y_temp_raw = self._io.read_u2be()
        self.sa_plus_y_temp_raw = self._io.read_u2be()
        self.csie_temp_raw = self._io.read_u2be()
        self.battery_1_temp_raw = self._io.read_u2be()
        self.batt_board_temp_raw = self._io.read_u2be()
        self.battery_1_charge_current_raw = self._io.read_u2be()
        self.battery_2_charge_current_raw = self._io.read_u2be()
        self.dsps_visible_sps_sun_pos_x = self._io.read_s2be()
        self.dsps_visible_sps_sun_pos_y = self._io.read_s2be()
        self.dsps_x_ray_sps_sun_pos_x = self._io.read_s2be()
        self.dsps_x_ray_sps_sun_pos_y = self._io.read_s2be()
        self.dsps_sensor_board_temp_raw = self._io.read_s2be()
        self.adcs_ana_motor1_temp_raw = self._io.read_s2be()
        self.adcs_sun_point_angle_error_raw = self._io.read_u2be()
        self.num_sc_resets = self._io.read_u2be()
        self.csie_capture_state = self._io.read_bits_int_be(2)
        self.fault_protection_watchpoint_6_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint6StateValues, self._io.read_bits_int_be(2))
        self.fault_protection_watchpoint_5_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint5StateValues, self._io.read_bits_int_be(2))
        self.fault_protection_watchpoint_4_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint4StateValues, self._io.read_bits_int_be(2))
        self.fault_protection_watchpoint_3_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint3StateValues, self._io.read_bits_int_be(2))
        self.fault_protection_watchpoint_2_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint2StateValues, self._io.read_bits_int_be(2))
        self.fault_protection_watchpoint_1_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint1StateValues, self._io.read_bits_int_be(2))
        self.fault_protection_watchpoint_0_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionWatchpoint0StateValues, self._io.read_bits_int_be(2))
        self.opaque_3_bytes = self._io.read_bytes(4)
        self.clt_hours_until_reboot = self._io.read_u1()
        self.mode_system_mode = KaitaiStream.resolve_enum(SuncetApid1.ModeSystemModeValues, self._io.read_u1())
        self.uhf_temp = self._io.read_s1()
        self.opaque_4_bytes = self._io.read_bytes(4)
        self.fault_protection_task_state = KaitaiStream.resolve_enum(SuncetApid1.FaultProtectionTaskStateValues, self._io.read_u1())
        self.dsps_flare_magnitude = self._io.read_u1()
        self.dsps_flare_phase = KaitaiStream.resolve_enum(SuncetApid1.DspsFlarePhaseValues, self._io.read_u1())
        self.opaque_5_bytes = self._io.read_bytes(7)
        self.opaque_5_bits = self._io.read_bits_int_be(2)
        self.battery_1_charging_state = KaitaiStream.resolve_enum(SuncetApid1.Battery1ChargingStateValues, self._io.read_bits_int_be(1))
        self.eps_pwr_state_dsps = KaitaiStream.resolve_enum(SuncetApid1.EpsPwrStateDspsValues, self._io.read_bits_int_be(1))
        self.eps_pwr_state_csie = KaitaiStream.resolve_enum(SuncetApid1.EpsPwrStateCsieValues, self._io.read_bits_int_be(1))
        self.eps_pwr_state_xband = KaitaiStream.resolve_enum(SuncetApid1.EpsPwrStateXbandValues, self._io.read_bits_int_be(1))
        self.eps_pwr_state_uhf = KaitaiStream.resolve_enum(SuncetApid1.EpsPwrStateUhfValues, self._io.read_bits_int_be(1))
        self.eps_pwr_state_adcs = KaitaiStream.resolve_enum(SuncetApid1.EpsPwrStateAdcsValues, self._io.read_bits_int_be(1))
        self.telescope_door_pin_pulled = KaitaiStream.resolve_enum(SuncetApid1.TelescopeDoorPinPulledValues, self._io.read_bits_int_be(1))
        self.csie_heater_enable = KaitaiStream.resolve_enum(SuncetApid1.CsieHeaterEnableValues, self._io.read_bits_int_be(1))
        self.battery2_heater_enable = KaitaiStream.resolve_enum(SuncetApid1.Battery2HeaterEnableValues, self._io.read_bits_int_be(1))
        self.battery1_heater_enable = KaitaiStream.resolve_enum(SuncetApid1.Battery1HeaterEnableValues, self._io.read_bits_int_be(1))
        self.uhf_alive = KaitaiStream.resolve_enum(SuncetApid1.UhfAliveValues, self._io.read_bits_int_be(2))
        self.adcs_alive = self._io.read_bits_int_be(2)
        self.adcs_att_valid = KaitaiStream.resolve_enum(SuncetApid1.AdcsAttValidValues, self._io.read_bits_int_be(1))
        self.adcs_ref_valid = KaitaiStream.resolve_enum(SuncetApid1.AdcsRefValidValues, self._io.read_bits_int_be(1))
        self.adcs_time_valid = KaitaiStream.resolve_enum(SuncetApid1.AdcsTimeValidValues, self._io.read_bits_int_be(1))
        self.adcs_mode = KaitaiStream.resolve_enum(SuncetApid1.AdcsModeValues, self._io.read_bits_int_be(1))
        self.battery_2_charging_state = KaitaiStream.resolve_enum(SuncetApid1.Battery2ChargingStateValues, self._io.read_bits_int_be(1))
        self.adcs_sun_point_state = KaitaiStream.resolve_enum(SuncetApid1.AdcsSunPointStateValues, self._io.read_bits_int_be(3))
        self.xband_data_source = KaitaiStream.resolve_enum(SuncetApid1.XbandDataSourceValues, self._io.read_u1())
        self.opaque_6_bytes = self._io.read_bytes(1)
        self.opaque_fletcher32_checksum = self._io.read_bytes(4)


    def _fetch_instances(self):
        pass

    @property
    def adcs_ana_motor1_temp(self):
        """Wheel 1 Temp
        Engineering units: degC (inferred)
        Conversion/status map: C0=0.000000e+00 C1=5.000000e-03
        """
        if hasattr(self, '_m_adcs_ana_motor1_temp'):
            return self._m_adcs_ana_motor1_temp

        self._m_adcs_ana_motor1_temp = 0 + self.adcs_ana_motor1_temp_raw * 0.005
        return getattr(self, '_m_adcs_ana_motor1_temp', None)

    @property
    def adcs_body_rate_1(self):
        """ADCS Body Frame Rate 1
        Engineering units: rad/s
        Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
        """
        if hasattr(self, '_m_adcs_body_rate_1'):
            return self._m_adcs_body_rate_1

        self._m_adcs_body_rate_1 = 0 + self.adcs_body_rate_1_raw * 5E-9
        return getattr(self, '_m_adcs_body_rate_1', None)

    @property
    def adcs_body_rate_2(self):
        """ADCS Body Frame Rate 2
        Engineering units: rad/s
        Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
        """
        if hasattr(self, '_m_adcs_body_rate_2'):
            return self._m_adcs_body_rate_2

        self._m_adcs_body_rate_2 = 0 + self.adcs_body_rate_2_raw * 5E-9
        return getattr(self, '_m_adcs_body_rate_2', None)

    @property
    def adcs_body_rate_3(self):
        """ADCS Body Frame Rate 3
        Engineering units: rad/s
        Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
        """
        if hasattr(self, '_m_adcs_body_rate_3'):
            return self._m_adcs_body_rate_3

        self._m_adcs_body_rate_3 = 0 + self.adcs_body_rate_3_raw * 5E-9
        return getattr(self, '_m_adcs_body_rate_3', None)

    @property
    def adcs_sun_point_angle_error(self):
        """Angle between the estimated and commanded Sun vectors.
        Engineering units: deg
        Conversion/status map: C0=0.000000e+00 C1=3.000000e-03
        """
        if hasattr(self, '_m_adcs_sun_point_angle_error'):
            return self._m_adcs_sun_point_angle_error

        self._m_adcs_sun_point_angle_error = 0 + self.adcs_sun_point_angle_error_raw * 0.003
        return getattr(self, '_m_adcs_sun_point_angle_error', None)

    @property
    def adcs_wheel_speed_1(self):
        """ADCS Wheel Speed 1
        Engineering units: rpm
        Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
        """
        if hasattr(self, '_m_adcs_wheel_speed_1'):
            return self._m_adcs_wheel_speed_1

        self._m_adcs_wheel_speed_1 = 0 + self.adcs_wheel_speed_1_raw * 0.002
        return getattr(self, '_m_adcs_wheel_speed_1', None)

    @property
    def adcs_wheel_speed_2(self):
        """ADCS Wheel Speed 2
        Engineering units: rpm
        Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
        """
        if hasattr(self, '_m_adcs_wheel_speed_2'):
            return self._m_adcs_wheel_speed_2

        self._m_adcs_wheel_speed_2 = 0 + self.adcs_wheel_speed_2_raw * 0.002
        return getattr(self, '_m_adcs_wheel_speed_2', None)

    @property
    def adcs_wheel_speed_3(self):
        """ADCS Wheel Speed 3
        Engineering units: rpm
        Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
        """
        if hasattr(self, '_m_adcs_wheel_speed_3'):
            return self._m_adcs_wheel_speed_3

        self._m_adcs_wheel_speed_3 = 0 + self.adcs_wheel_speed_3_raw * 0.002
        return getattr(self, '_m_adcs_wheel_speed_3', None)

    @property
    def batt_board_temp(self):
        """Battery Board Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_batt_board_temp'):
            return self._m_batt_board_temp

        self._m_batt_board_temp = 125.55 + self.batt_board_temp_raw * (-0.13622 + self.batt_board_temp_raw * (0.000098611 + self.batt_board_temp_raw * (-4.4176E-8 + self.batt_board_temp_raw * (1.0125E-11 + self.batt_board_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_batt_board_temp', None)

    @property
    def battery_1_charge_current(self):
        """Battery 1 Charge Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.366954e-04
        """
        if hasattr(self, '_m_battery_1_charge_current'):
            return self._m_battery_1_charge_current

        self._m_battery_1_charge_current = 0 + self.battery_1_charge_current_raw * 0.0002366954
        return getattr(self, '_m_battery_1_charge_current', None)

    @property
    def battery_1_temp(self):
        """Battery 1 Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_battery_1_temp'):
            return self._m_battery_1_temp

        self._m_battery_1_temp = 125.55 + self.battery_1_temp_raw * (-0.13622 + self.battery_1_temp_raw * (0.000098611 + self.battery_1_temp_raw * (-4.4176E-8 + self.battery_1_temp_raw * (1.0125E-11 + self.battery_1_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_battery_1_temp', None)

    @property
    def battery_1_voltage(self):
        """Battery 1 Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
        """
        if hasattr(self, '_m_battery_1_voltage'):
            return self._m_battery_1_voltage

        self._m_battery_1_voltage = 0 + self.battery_1_voltage_raw * 0.0088623
        return getattr(self, '_m_battery_1_voltage', None)

    @property
    def battery_2_charge_current(self):
        """Battery 2 Charge Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.366954e-04
        """
        if hasattr(self, '_m_battery_2_charge_current'):
            return self._m_battery_2_charge_current

        self._m_battery_2_charge_current = 0 + self.battery_2_charge_current_raw * 0.0002366954
        return getattr(self, '_m_battery_2_charge_current', None)

    @property
    def battery_2_voltage(self):
        """Battery 2 Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
        """
        if hasattr(self, '_m_battery_2_voltage'):
            return self._m_battery_2_voltage

        self._m_battery_2_voltage = 0 + self.battery_2_voltage_raw * 0.0088623
        return getattr(self, '_m_battery_2_voltage', None)

    @property
    def ccsds_apid(self):
        """CCSDS application process identifier; expected value is 1.
        Engineering units: dimensionless
        """
        if hasattr(self, '_m_ccsds_apid'):
            return self._m_ccsds_apid

        self._m_ccsds_apid = self.ccsds_primary_word & 2047
        return getattr(self, '_m_ccsds_apid', None)

    @property
    def ccsds_packet_type(self):
        """CCSDS packet type; telemetry is expected for the beacon.
        Engineering units: dimensionless
        """
        if hasattr(self, '_m_ccsds_packet_type'):
            return self._m_ccsds_packet_type

        self._m_ccsds_packet_type = self.ccsds_primary_word >> 12 & 1
        return getattr(self, '_m_ccsds_packet_type', None)

    @property
    def ccsds_secondary_header_flag(self):
        """Indicates presence of the SunCET secondary time header.
        Engineering units: dimensionless
        """
        if hasattr(self, '_m_ccsds_secondary_header_flag'):
            return self._m_ccsds_secondary_header_flag

        self._m_ccsds_secondary_header_flag = self.ccsds_primary_word >> 11 & 1
        return getattr(self, '_m_ccsds_secondary_header_flag', None)

    @property
    def ccsds_sequence_count(self):
        """CCSDS packet sequence counter.
        Engineering units: count
        """
        if hasattr(self, '_m_ccsds_sequence_count'):
            return self._m_ccsds_sequence_count

        self._m_ccsds_sequence_count = self.ccsds_sequence_word & 16383
        return getattr(self, '_m_ccsds_sequence_count', None)

    @property
    def ccsds_sequence_flags(self):
        """CCSDS packet sequence flags.
        Engineering units: dimensionless
        """
        if hasattr(self, '_m_ccsds_sequence_flags'):
            return self._m_ccsds_sequence_flags

        self._m_ccsds_sequence_flags = self.ccsds_sequence_word >> 14 & 3
        return getattr(self, '_m_ccsds_sequence_flags', None)

    @property
    def ccsds_version(self):
        """CCSDS Space Packet version number.
        Engineering units: dimensionless
        """
        if hasattr(self, '_m_ccsds_version'):
            return self._m_ccsds_version

        self._m_ccsds_version = self.ccsds_primary_word >> 13 & 7
        return getattr(self, '_m_ccsds_version', None)

    @property
    def cdh_3v3_reference_voltage(self):
        """CDH 3.3 V reference measurement.
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.611300e-03
        """
        if hasattr(self, '_m_cdh_3v3_reference_voltage'):
            return self._m_cdh_3v3_reference_voltage

        self._m_cdh_3v3_reference_voltage = 0 + self.cdh_3v3_reference_voltage_raw * 0.0016113
        return getattr(self, '_m_cdh_3v3_reference_voltage', None)

    @property
    def cdh_temp(self):
        """CDH Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_cdh_temp'):
            return self._m_cdh_temp

        self._m_cdh_temp = 125.55 + self.cdh_temp_raw * (-0.13622 + self.cdh_temp_raw * (0.000098611 + self.cdh_temp_raw * (-4.4176E-8 + self.cdh_temp_raw * (1.0125E-11 + self.cdh_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_cdh_temp', None)

    @property
    def csie_current(self):
        """CSIE Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
        """
        if hasattr(self, '_m_csie_current'):
            return self._m_csie_current

        self._m_csie_current = 0 + self.csie_current_raw * 0.0012207
        return getattr(self, '_m_csie_current', None)

    @property
    def csie_temp(self):
        """CSIE Detector Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_csie_temp'):
            return self._m_csie_temp

        self._m_csie_temp = 125.55 + self.csie_temp_raw * (-0.13622 + self.csie_temp_raw * (0.000098611 + self.csie_temp_raw * (-4.4176E-8 + self.csie_temp_raw * (1.0125E-11 + self.csie_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_csie_temp', None)

    @property
    def csie_voltage(self):
        """CSIE Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=3.963900e-03
        """
        if hasattr(self, '_m_csie_voltage'):
            return self._m_csie_voltage

        self._m_csie_voltage = 0 + self.csie_voltage_raw * 0.0039639
        return getattr(self, '_m_csie_voltage', None)

    @property
    def dsps_current(self):
        """DSPS Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
        """
        if hasattr(self, '_m_dsps_current'):
            return self._m_dsps_current

        self._m_dsps_current = 0 + self.dsps_current_raw * 0.0012207
        return getattr(self, '_m_dsps_current', None)

    @property
    def dsps_sensor_board_temp(self):
        """Dual-SPS Sensor Board Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.000000e-02
        """
        if hasattr(self, '_m_dsps_sensor_board_temp'):
            return self._m_dsps_sensor_board_temp

        self._m_dsps_sensor_board_temp = 0 + self.dsps_sensor_board_temp_raw * 0.01
        return getattr(self, '_m_dsps_sensor_board_temp', None)

    @property
    def dsps_voltage(self):
        """DSPS Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=3.963900e-03
        """
        if hasattr(self, '_m_dsps_voltage'):
            return self._m_dsps_voltage

        self._m_dsps_voltage = 0 + self.dsps_voltage_raw * 0.0039639
        return getattr(self, '_m_dsps_voltage', None)

    @property
    def eps_3v3_reference_voltage(self):
        """EPS 3.3 V reference measurement.
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.611330e-03
        """
        if hasattr(self, '_m_eps_3v3_reference_voltage'):
            return self._m_eps_3v3_reference_voltage

        self._m_eps_3v3_reference_voltage = 0 + self.eps_3v3_reference_voltage_raw * 0.00161133
        return getattr(self, '_m_eps_3v3_reference_voltage', None)

    @property
    def eps_bus_current(self):
        """EPS Bus Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
        """
        if hasattr(self, '_m_eps_bus_current'):
            return self._m_eps_bus_current

        self._m_eps_bus_current = 0 + self.eps_bus_current_raw * 0.0012207
        return getattr(self, '_m_eps_bus_current', None)

    @property
    def eps_bus_voltage(self):
        """EPS Bus Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
        """
        if hasattr(self, '_m_eps_bus_voltage'):
            return self._m_eps_bus_voltage

        self._m_eps_bus_voltage = 0 + self.eps_bus_voltage_raw * 0.0088623
        return getattr(self, '_m_eps_bus_voltage', None)

    @property
    def eps_temp(self):
        """EPS Board Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_eps_temp'):
            return self._m_eps_temp

        self._m_eps_temp = 125.55 + self.eps_temp_raw * (-0.13622 + self.eps_temp_raw * (0.000098611 + self.eps_temp_raw * (-4.4176E-8 + self.eps_temp_raw * (1.0125E-11 + self.eps_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_eps_temp', None)

    @property
    def ifb_therm1(self):
        """Interface Board Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_ifb_therm1'):
            return self._m_ifb_therm1

        self._m_ifb_therm1 = 125.55 + self.ifb_therm1_raw * (-0.13622 + self.ifb_therm1_raw * (0.000098611 + self.ifb_therm1_raw * (-4.4176E-8 + self.ifb_therm1_raw * (1.0125E-11 + self.ifb_therm1_raw * -9.3905E-16))))
        return getattr(self, '_m_ifb_therm1', None)

    @property
    def rail_3v3_current(self):
        """3p3 Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.056600e-05
        """
        if hasattr(self, '_m_rail_3v3_current'):
            return self._m_rail_3v3_current

        self._m_rail_3v3_current = 0 + self.rail_3v3_current_raw * 0.000080566
        return getattr(self, '_m_rail_3v3_current', None)

    @property
    def rail_3v3_voltage(self):
        """3p3 Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.611300e-03
        """
        if hasattr(self, '_m_rail_3v3_voltage'):
            return self._m_rail_3v3_voltage

        self._m_rail_3v3_voltage = 0 + self.rail_3v3_voltage_raw * 0.0016113
        return getattr(self, '_m_rail_3v3_voltage', None)

    @property
    def sa_minus_y_temp(self):
        """Solar Array 1 Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_sa_minus_y_temp'):
            return self._m_sa_minus_y_temp

        self._m_sa_minus_y_temp = 125.55 + self.sa_minus_y_temp_raw * (-0.13622 + self.sa_minus_y_temp_raw * (0.000098611 + self.sa_minus_y_temp_raw * (-4.4176E-8 + self.sa_minus_y_temp_raw * (1.0125E-11 + self.sa_minus_y_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_sa_minus_y_temp', None)

    @property
    def sa_plus_y_temp(self):
        """Solar Array 2 Temperature
        Engineering units: degC (inferred)
        Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
        """
        if hasattr(self, '_m_sa_plus_y_temp'):
            return self._m_sa_plus_y_temp

        self._m_sa_plus_y_temp = 125.55 + self.sa_plus_y_temp_raw * (-0.13622 + self.sa_plus_y_temp_raw * (0.000098611 + self.sa_plus_y_temp_raw * (-4.4176E-8 + self.sa_plus_y_temp_raw * (1.0125E-11 + self.sa_plus_y_temp_raw * -9.3905E-16))))
        return getattr(self, '_m_sa_plus_y_temp', None)

    @property
    def solar_array_8_cell_string_current(self):
        """Solar Array 8-Cell String Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
        """
        if hasattr(self, '_m_solar_array_8_cell_string_current'):
            return self._m_solar_array_8_cell_string_current

        self._m_solar_array_8_cell_string_current = 0 + self.solar_array_8_cell_string_current_raw * 0.0020142
        return getattr(self, '_m_solar_array_8_cell_string_current', None)

    @property
    def solar_array_8_cell_string_voltage(self):
        """Solar Array 8-Cell String Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=9.659200e-03
        """
        if hasattr(self, '_m_solar_array_8_cell_string_voltage'):
            return self._m_solar_array_8_cell_string_voltage

        self._m_solar_array_8_cell_string_voltage = 0 + self.solar_array_8_cell_string_voltage_raw * 0.0096592
        return getattr(self, '_m_solar_array_8_cell_string_voltage', None)

    @property
    def solar_array_9_cell_string_current(self):
        """9-Cell String Solar Array Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
        """
        if hasattr(self, '_m_solar_array_9_cell_string_current'):
            return self._m_solar_array_9_cell_string_current

        self._m_solar_array_9_cell_string_current = 0 + self.solar_array_9_cell_string_current_raw * 0.0020142
        return getattr(self, '_m_solar_array_9_cell_string_current', None)

    @property
    def solar_array_9_cell_string_voltage(self):
        """9-Cell String Solar Array Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=9.659200e-03
        """
        if hasattr(self, '_m_solar_array_9_cell_string_voltage'):
            return self._m_solar_array_9_cell_string_voltage

        self._m_solar_array_9_cell_string_voltage = 0 + self.solar_array_9_cell_string_voltage_raw * 0.0096592
        return getattr(self, '_m_solar_array_9_cell_string_voltage', None)

    @property
    def uhf_current(self):
        """UHF Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
        """
        if hasattr(self, '_m_uhf_current'):
            return self._m_uhf_current

        self._m_uhf_current = 0 + self.uhf_current_raw * 0.0020142
        return getattr(self, '_m_uhf_current', None)

    @property
    def uhf_voltage(self):
        """UHF Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
        """
        if hasattr(self, '_m_uhf_voltage'):
            return self._m_uhf_voltage

        self._m_uhf_voltage = 0 + self.uhf_voltage_raw * 0.0088623
        return getattr(self, '_m_uhf_voltage', None)

    @property
    def xact_current(self):
        """XACT Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
        """
        if hasattr(self, '_m_xact_current'):
            return self._m_xact_current

        self._m_xact_current = 0 + self.xact_current_raw * 0.0020142
        return getattr(self, '_m_xact_current', None)

    @property
    def xact_voltage(self):
        """XACT Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
        """
        if hasattr(self, '_m_xact_voltage'):
            return self._m_xact_voltage

        self._m_xact_voltage = 0 + self.xact_voltage_raw * 0.0088623
        return getattr(self, '_m_xact_voltage', None)

    @property
    def xband_current(self):
        """XBAND Current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
        """
        if hasattr(self, '_m_xband_current'):
            return self._m_xband_current

        self._m_xband_current = 0 + self.xband_current_raw * 0.0020142
        return getattr(self, '_m_xband_current', None)

    @property
    def xband_pa_current(self):
        """XBAND Power Amplifier current
        Engineering units: A (inferred)
        Conversion/status map: C0=0.000000e+00 C1=1.654473e-03
        """
        if hasattr(self, '_m_xband_pa_current'):
            return self._m_xband_pa_current

        self._m_xband_pa_current = 0 + self.xband_pa_current_raw * 0.001654473
        return getattr(self, '_m_xband_pa_current', None)

    @property
    def xband_pa_temp(self):
        """XBAND Power Amplifier Temp
        Engineering units: degC (inferred)
        Conversion/status map: C0=0.000000e+00 C1=9.765625e-04
        """
        if hasattr(self, '_m_xband_pa_temp'):
            return self._m_xband_pa_temp

        self._m_xband_pa_temp = 0 + self.xband_pa_temp_raw * 0.0009765625
        return getattr(self, '_m_xband_pa_temp', None)

    @property
    def xband_voltage(self):
        """XBAND Voltage
        Engineering units: V (inferred)
        Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
        """
        if hasattr(self, '_m_xband_voltage'):
            return self._m_xband_voltage

        self._m_xband_voltage = 0 + self.xband_voltage_raw * 0.0088623
        return getattr(self, '_m_xband_voltage', None)
