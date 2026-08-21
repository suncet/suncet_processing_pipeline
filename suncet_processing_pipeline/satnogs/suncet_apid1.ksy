meta:
  id: suncet_apid1
  title: SunCET public CCSDS APID 1 beacon (provisional)
  endian: be
  bit-endian: be
doc-ref: |
  https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md
doc: |
  Provisional bare-CCSDS decoder for the public SunCET APID 1 beacon.
  The AX.25 wrapper will be added after flight framing is confirmed.

  :field ccsds_version: ccsds_version
  :field ccsds_packet_type: ccsds_packet_type
  :field ccsds_secondary_header_flag: ccsds_secondary_header_flag
  :field ccsds_apid: ccsds_apid
  :field ccsds_sequence_flags: ccsds_sequence_flags
  :field ccsds_sequence_count: ccsds_sequence_count
  :field ccsds_packet_length_field: ccsds_packet_length_field
  :field spacecraft_time_seconds_since_2000: spacecraft_time_seconds_since_2000
  :field spacecraft_time_fine_raw: spacecraft_time_fine_raw
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
  :field csie_meta_nand_sci_write_ptr: csie_meta_nand_sci_write_ptr
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
  :field adcs_wheel_speed_1: adcs_wheel_speed_1
  :field adcs_wheel_speed_2: adcs_wheel_speed_2
  :field adcs_wheel_speed_3: adcs_wheel_speed_3
  :field adcs_sun_point_angle_error: adcs_sun_point_angle_error
  :field num_sc_resets: num_sc_resets
  :field clt_hours_until_reboot: clt_hours_until_reboot
  :field mode_system_mode: mode_system_mode
  :field uhf_temp: uhf_temp
  :field fault_protection_task_state: fault_protection_task_state
  :field csie_capture_state: csie_capture_state
  :field fault_protection_watchpoint_6_state: fault_protection_watchpoint_6_state
  :field fault_protection_watchpoint_5_state: fault_protection_watchpoint_5_state
  :field fault_protection_watchpoint_4_state: fault_protection_watchpoint_4_state
  :field fault_protection_watchpoint_3_state: fault_protection_watchpoint_3_state
  :field fault_protection_watchpoint_2_state: fault_protection_watchpoint_2_state
  :field fault_protection_watchpoint_1_state: fault_protection_watchpoint_1_state
  :field fault_protection_watchpoint_0_state: fault_protection_watchpoint_0_state
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
seq:
  - id: ccsds_primary_word
    type: u2
    valid: 2049
    doc: |
      Packed CCSDS version/type/secondary-header/APID word.
      The only accepted value describes telemetry APID 1 with
      the SunCET secondary header present.
  - id: ccsds_sequence_word
    type: u2
    doc: Packed CCSDS sequence flags and sequence counter.
  - id: ccsds_packet_length_field
    type: u2
    valid:
      expr: _ == _io.size - 7
    doc: |
      CCSDS packet length field: bytes after the primary header minus one.
      Engineering units: bytes
  - id: spacecraft_time_seconds_since_2000
    type: u4
    doc: |
      Whole seconds elapsed since 2000-01-01T00:00:00Z.
      Engineering units: s
  - id: spacecraft_time_fine_raw
    type: u2
    doc: |
      Raw fine-time field; intended as microseconds after the coarse second, but 16-bit serialization remains unresolved.
      Engineering units: µs intended; raw encoding TBC
  - id: partition_write_adcs
    type: u4
    doc: |
      Current write pointer in the NAND ADCS partition.
      Engineering units: raw address
  - id: partition_read_adcs
    type: u4
    doc: |
      Current read pointer in the NAND ADCS partition.
      Engineering units: raw address
  - id: partition_write_hk
    type: u4
    doc: |
      Current write pointer in the NAND housekeeping partition.
      Engineering units: raw address
  - id: partition_read_hk
    type: u4
    doc: |
      Current read pointer in the NAND housekeeping partition.
      Engineering units: raw address
  - id: partition_write_sci
    type: u4
    doc: |
      Current write pointer in the NAND science partition.
      Engineering units: raw address
  - id: partition_read_sci
    type: u4
    doc: |
      Current read pointer in the NAND science partition.
      Engineering units: raw address
  - id: partition_write_dsps
    type: u4
    doc: |
      Current write pointer in the NAND Dual-SPS partition.
      Engineering units: raw address
  - id: partition_read_dsps
    type: u4
    doc: |
      Current read pointer in the NAND Dual-SPS partition.
      Engineering units: raw address
  - id: store_partition_write_log
    type: u4
    doc: |
      Current write pointer in the NAND log partition.
      Engineering units: raw address
  - id: csie_nand_sci_write_ptr
    type: u4
    doc: |
      Current write pointer in the NAND CSIE science-image partition.
      Engineering units: raw address
  - id: csie_meta_nand_sci_write_ptr
    type: u4
    doc: |
      Current write pointer in the NAND CSIE image-metadata partition.
      Engineering units: raw address
  - id: time_since_boot
    type: u4
    doc: |
      Seconds since most recent boot
      Engineering units: s
  - id: time_alive
    type: u4
    doc: |
      Number of seconds spent powered on and processing (i.e Alive)
      Engineering units: s
  - id: time_mission_elapsed_time
    type: u4
    doc: |
      Time since Mission start
      Engineering units: s
  - id: mode_seconds_since_mode_change
    type: u4
    doc: |
      Second since mode switch
      Engineering units: s
  - id: dsps_flare_level
    type: f4
    doc: |
      Dual-SPS flare trigger threshold in log10 of estimated GOES XRS-B flux
      Engineering units: log10(XRS-B flux)
  - id: adcs_body_rate_1_raw
    type: s4
    doc: |
      ADCS Body Frame Rate 1
      Engineering units: rad/s
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
  - id: adcs_body_rate_2_raw
    type: s4
    doc: |
      ADCS Body Frame Rate 2
      Engineering units: rad/s
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
  - id: adcs_body_rate_3_raw
    type: s4
    doc: |
      ADCS Body Frame Rate 3
      Engineering units: rad/s
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
  - id: csie_img_hist_0
    type: u4
    doc: |
      CSIE pixel count in histogram bin 0; configurable DN range [offset, offset+width-1], default 0-31
      Engineering units: count
  - id: csie_img_hist_1
    type: u4
    doc: |
      CSIE pixel count in histogram bin 1; configurable DN range [offset+width, offset+2*width-1], default 32-63
      Engineering units: count
  - id: csie_img_hist_2
    type: u4
    doc: |
      CSIE pixel count in histogram bin 2; configurable DN range [offset+2*width, offset+3*width-1], default 64-95
      Engineering units: count
  - id: csie_img_hist_3
    type: u4
    doc: |
      CSIE pixel count in histogram bin 3; configurable DN range [offset+3*width, offset+4*width-1], default 96-127
      Engineering units: count
  - id: csie_img_hist_4
    type: u4
    doc: |
      CSIE pixel count in histogram bin 4; configurable DN range [offset+4*width, offset+5*width-1], default 128-159
      Engineering units: count
  - id: csie_img_hist_5
    type: u4
    doc: |
      CSIE pixel count in histogram bin 5; configurable DN range [offset+5*width, offset+6*width-1], default 160-191; beacon truncates the full histogram after bin 5
      Engineering units: count
  - id: opaque_1_bytes
    size: 2
    doc: Excluded public-beacon fields, consumed opaquely.
  - id: xband_pa_temp_raw
    type: u2
    doc: |
      XBAND Power Amplifier Temp
      Engineering units: degC (inferred)
      Conversion/status map: C0=0.000000e+00 C1=9.765625e-04
  - id: xband_pa_current_raw
    type: u2
    doc: |
      XBAND Power Amplifier current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.654473e-03
  - id: opaque_2_bytes
    size: 10
    doc: Excluded public-beacon fields, consumed opaquely.
  - id: rail_3v3_voltage_raw
    type: u2
    doc: |
      3p3 Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.611300e-03
  - id: rail_3v3_current_raw
    type: u2
    doc: |
      3p3 Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.056600e-05
  - id: cdh_temp_raw
    type: u2
    doc: |
      CDH Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: cdh_3v3_reference_voltage_raw
    type: u2
    doc: |
      CDH 3.3 V reference measurement.
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.611300e-03
  - id: solar_array_8_cell_string_voltage_raw
    type: u2
    doc: |
      Solar Array 8-Cell String Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=9.659200e-03
  - id: solar_array_8_cell_string_current_raw
    type: u2
    doc: |
      Solar Array 8-Cell String Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  - id: solar_array_9_cell_string_voltage_raw
    type: u2
    doc: |
      9-Cell String Solar Array Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=9.659200e-03
  - id: solar_array_9_cell_string_current_raw
    type: u2
    doc: |
      9-Cell String Solar Array Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  - id: battery_1_voltage_raw
    type: u2
    doc: |
      Battery 1 Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.056641e-03
  - id: battery_2_voltage_raw
    type: u2
    doc: |
      Battery 2 Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.056641e-03
  - id: eps_temp_raw
    type: u2
    doc: |
      EPS Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: eps_3v3_reference_voltage_raw
    type: u2
    doc: |
      EPS 3.3 V reference measurement.
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.611330e-03
  - id: eps_bus_voltage_raw
    type: u2
    doc: |
      EPS Bus Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  - id: eps_bus_current_raw
    type: u2
    doc: |
      EPS Bus Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
  - id: xact_voltage_raw
    type: u2
    doc: |
      XACT Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  - id: xact_current_raw
    type: u2
    doc: |
      XACT Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  - id: uhf_voltage_raw
    type: u2
    doc: |
      UHF Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  - id: uhf_current_raw
    type: u2
    doc: |
      UHF Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  - id: xband_voltage_raw
    type: u2
    doc: |
      XBAND Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  - id: xband_current_raw
    type: u2
    doc: |
      XBAND Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  - id: csie_voltage_raw
    type: u2
    doc: |
      CSIE Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=3.963900e-03
  - id: csie_current_raw
    type: u2
    doc: |
      CSIE Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
  - id: dsps_voltage_raw
    type: u2
    doc: |
      DSPS Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=3.963900e-03
  - id: dsps_current_raw
    type: u2
    doc: |
      DSPS Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
  - id: ifb_therm1_raw
    type: u2
    doc: |
      Interface Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: sa_minus_y_temp_raw
    type: u2
    doc: |
      Solar Array 1 Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: sa_plus_y_temp_raw
    type: u2
    doc: |
      Solar Array 2 Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: csie_temp_raw
    type: u2
    doc: |
      CSIE Detector Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: battery_1_temp_raw
    type: u2
    doc: |
      Battery 1 Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: batt_board_temp_raw
    type: u2
    doc: |
      Battery Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  - id: battery_1_charge_current_raw
    type: u2
    doc: |
      Battery 1 Charge Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.366954e-04
  - id: battery_2_charge_current_raw
    type: u2
    doc: |
      Battery 2 Charge Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.366954e-04
  - id: dsps_visible_sps_sun_pos_x
    type: s2
    doc: |
      Visible DSPS Sun position x (arcsecs)
      Engineering units: arcsec
  - id: dsps_visible_sps_sun_pos_y
    type: s2
    doc: |
      Visible DSPS Sun position y (arcsecs)
      Engineering units: arcsec
  - id: dsps_x_ray_sps_sun_pos_x
    type: s2
    doc: |
      X-RAY DSPS Sun position x (arcsecs)
      Engineering units: arcsec
  - id: dsps_x_ray_sps_sun_pos_y
    type: s2
    doc: |
      X-RAY DSPS Sun position y (arcsecs)
      Engineering units: arcsec
  - id: dsps_sensor_board_temp_raw
    type: u2
    doc: |
      Dual-SPS Sensor Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.000000e-02
  - id: adcs_ana_motor1_temp_raw
    type: s2
    doc: |
      Wheel 1 Temp
      Engineering units: degC (inferred)
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-03
  - id: adcs_wheel_speed_1_raw
    type: s4
    doc: |
      ADCS Wheel Speed 1
      Engineering units: rpm
      Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
  - id: adcs_wheel_speed_2_raw
    type: s4
    doc: |
      ADCS Wheel Speed 2
      Engineering units: rpm
      Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
  - id: adcs_wheel_speed_3_raw
    type: s4
    doc: |
      ADCS Wheel Speed 3
      Engineering units: rpm
      Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
  - id: adcs_sun_point_angle_error_raw
    type: u2
    doc: |
      Angle between the estimated and commanded Sun vectors.
      Engineering units: deg
      Conversion/status map: C0=0.000000e+00 C1=3.000000e-03
  - id: num_sc_resets
    type: u2
    doc: |
      Number of Spacecraft Reboots
      Engineering units: count
  - id: opaque_3_bytes
    size: 4
    doc: Excluded public-beacon fields, consumed opaquely.
  - id: clt_hours_until_reboot
    type: u1
    doc: |
      Command Loss Timer time remaining until the spacecraft resets.
      Engineering units: h
  - id: mode_system_mode
    type: u1
    enum: mode_system_mode_values
    doc: |
      System mode
      Engineering units: dimensionless
      Conversion/status map: 0/PHOENIX 1/SAFE 2/SCIENCE 3/DOWNLINK
  - id: uhf_temp
    type: s1
    doc: |
      UHF Temperature
      Engineering units: degC (inferred)
  - id: opaque_4_bytes
    size: 4
    doc: Excluded public-beacon fields, consumed opaquely.
  - id: fault_protection_task_state
    type: u1
    enum: fault_protection_task_state_values
    doc: |
      Fault-protection task state.
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: csie_capture_state
    type: b2
    doc: |
      CSIE image-capture state.
      Engineering units: dimensionless
  - id: fault_protection_watchpoint_6_state
    type: b2
    enum: fault_protection_watchpoint_6_state_values
    doc: |
      WP State wp6
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: fault_protection_watchpoint_5_state
    type: b2
    enum: fault_protection_watchpoint_5_state_values
    doc: |
      WP State wp5
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: fault_protection_watchpoint_4_state
    type: b2
    enum: fault_protection_watchpoint_4_state_values
    doc: |
      WP State wp4
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: fault_protection_watchpoint_3_state
    type: b2
    enum: fault_protection_watchpoint_3_state_values
    doc: |
      WP State wp3
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: fault_protection_watchpoint_2_state
    type: b2
    enum: fault_protection_watchpoint_2_state_values
    doc: |
      WP State wp2
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: fault_protection_watchpoint_1_state
    type: b2
    enum: fault_protection_watchpoint_1_state_values
    doc: |
      WP State wp1
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: fault_protection_watchpoint_0_state
    type: b2
    enum: fault_protection_watchpoint_0_state_values
    doc: |
      WP State wp0
      Engineering units: dimensionless
      Conversion/status map: 0/DISABLED 1/PASSIVE 2/ENABLED
  - id: dsps_flare_magnitude_raw
    type: s1
    doc: |
      Dual-SPS estimate of GOES XRS-B flare magnitude
      Engineering units: log10(XRS-B flux)
      Conversion/status map: C0=0.000000e+00 C1=1.000000e-01
  - id: dsps_flare_phase
    type: u1
    enum: dsps_flare_phase_values
    doc: |
      Dual-SPS flare-state bit flags
      Engineering units: dimensionless
      Conversion/status map: 0/NOT_IN_SUN 1/FILLING_HISTORY 2/NOT_IN_FLARE 4/FLARE_LIKELY 24/IN_FLARE_DECREASING 40/IN_FLARE_RISING
  - id: opaque_5_bytes
    size: 7
    doc: Excluded public-beacon fields, consumed opaquely.
  - id: opaque_5_bits
    type: b2
    doc: Excluded public-beacon bits, consumed opaquely.
  - id: battery_1_charging_state
    type: b1
    enum: battery_1_charging_state_values
    doc: |
      Charging State of the Battery
      Engineering units: dimensionless
      Conversion/status map: 1/CHARGING 0/DISCHARGING
  - id: eps_pwr_state_dsps
    type: b1
    enum: eps_pwr_state_dsps_values
    doc: |
      EPS Power state for Dual-SPS
      Engineering units: dimensionless
      Conversion/status map: 0/OFF 1/ON
  - id: eps_pwr_state_csie
    type: b1
    enum: eps_pwr_state_csie_values
    doc: |
      EPS Power state for CSIE
      Engineering units: dimensionless
      Conversion/status map: 0/OFF 1/ON
  - id: eps_pwr_state_xband
    type: b1
    enum: eps_pwr_state_xband_values
    doc: |
      EPS Power state for X-Band radio
      Engineering units: dimensionless
      Conversion/status map: 0/OFF 1/ON
  - id: eps_pwr_state_uhf
    type: b1
    enum: eps_pwr_state_uhf_values
    doc: |
      EPS Power state for UHF radio
      Engineering units: dimensionless
      Conversion/status map: 0/OFF 1/ON
  - id: eps_pwr_state_adcs
    type: b1
    enum: eps_pwr_state_adcs_values
    doc: |
      EPS Power state for ADCS/XACT
      Engineering units: dimensionless
      Conversion/status map: 0/OFF 1/ON
  - id: telescope_door_pin_pulled
    type: b1
    enum: telescope_door_pin_pulled_values
    doc: |
      Telescope Door Release Pin feedback
      Engineering units: dimensionless
      Conversion/status map: 1/ENGAGED 0/PULLED
  - id: csie_heater_enable
    type: b1
    enum: csie_heater_enable_values
    doc: |
      CSIE Heater enable flag
      Engineering units: dimensionless
      Conversion/status map: 0/NO 1/YES
  - id: battery2_heater_enable
    type: b1
    enum: battery2_heater_enable_values
    doc: |
      Battery 2 heater enable flag
      Engineering units: dimensionless
      Conversion/status map: 0/NO 1/YES
  - id: battery1_heater_enable
    type: b1
    enum: battery1_heater_enable_values
    doc: |
      Battery 1 heater enable flag
      Engineering units: dimensionless
      Conversion/status map: 0/NO 1/YES
  - id: uhf_alive
    type: b2
    enum: uhf_alive_values
    doc: |
      UHF Aliveness flag
      Engineering units: dimensionless
      Conversion/status map: 0/OFF 1/ALIVE 2/DEAD
  - id: adcs_alive
    type: b2
    doc: |
      ADCS Aliveness flag
      Engineering units: dimensionless
  - id: adcs_att_valid
    type: b1
    enum: adcs_att_valid_values
    doc: |
      Attitude Valid
      Engineering units: dimensionless
      Conversion/status map: 0/NO 1/YES
  - id: adcs_ref_valid
    type: b1
    enum: adcs_ref_valid_values
    doc: |
      Refs Valid
      Engineering units: dimensionless
      Conversion/status map: 0/NO 1/YES
  - id: adcs_time_valid
    type: b1
    enum: adcs_time_valid_values
    doc: |
      Time Valid
      Engineering units: dimensionless
      Conversion/status map: 1/YES 0/NO
  - id: adcs_mode
    type: b1
    enum: adcs_mode_values
    doc: |
      ADCS Mode
      Engineering units: dimensionless
      Conversion/status map: 0/SUN_POINT 1/FINE_REF_POINT
  - id: battery_2_charging_state
    type: b1
    enum: battery_2_charging_state_values
    doc: |
      Charging State of the Battery
      Engineering units: dimensionless
      Conversion/status map: 1/CHARGING 0/DISCHARGING
  - id: adcs_sun_point_state
    type: b3
    enum: adcs_sun_point_state_values
    doc: |
      Sun Point State
      Engineering units: dimensionless
      Conversion/status map: 0/SUN_POINT 1/FINE_REF_POINT 2/SEARCH_INIT 3/SEARCHING 4/WAITING 5/CONVERGING 6/ON_SUN 7/NOT_ACTV
  - id: xband_data_source
    type: u1
    enum: xband_data_source_values
    doc: |
      Selected X-band data source.
      Engineering units: dimensionless
      Conversion/status map: 0/TEST_PAT 1/CDH
  - id: provisional_extra_byte
    type: u1
    if: _io.size == 252
    doc: |
      Apparent additional pre-checksum byte in current 252-byte captures.
      Its flight definition is unresolved and it is not public telemetry.
  - id: opaque_fletcher32_checksum
    size: 4
    doc: Fletcher-32 bytes consumed for framing; not exposed as telemetry.
instances:
  ccsds_version:
    value: (ccsds_primary_word >> 13) & 7
    doc: |
      CCSDS Space Packet version number.
      Engineering units: dimensionless
  ccsds_packet_type:
    value: (ccsds_primary_word >> 12) & 1
    doc: |
      CCSDS packet type; telemetry is expected for the beacon.
      Engineering units: dimensionless
  ccsds_secondary_header_flag:
    value: (ccsds_primary_word >> 11) & 1
    doc: |
      Indicates presence of the SunCET secondary time header.
      Engineering units: dimensionless
  ccsds_apid:
    value: ccsds_primary_word & 2047
    doc: |
      CCSDS application process identifier; expected value is 1.
      Engineering units: dimensionless
  ccsds_sequence_flags:
    value: (ccsds_sequence_word >> 14) & 3
    doc: |
      CCSDS packet sequence flags.
      Engineering units: dimensionless
  ccsds_sequence_count:
    value: ccsds_sequence_word & 16383
    doc: |
      CCSDS packet sequence counter.
      Engineering units: count
  adcs_body_rate_1:
    value: (0 + adcs_body_rate_1_raw * 5e-09)
    doc: |
      ADCS Body Frame Rate 1
      Engineering units: rad/s
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
  adcs_body_rate_2:
    value: (0 + adcs_body_rate_2_raw * 5e-09)
    doc: |
      ADCS Body Frame Rate 2
      Engineering units: rad/s
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
  adcs_body_rate_3:
    value: (0 + adcs_body_rate_3_raw * 5e-09)
    doc: |
      ADCS Body Frame Rate 3
      Engineering units: rad/s
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-09
  xband_pa_temp:
    value: (0 + xband_pa_temp_raw * 0.0009765625)
    doc: |
      XBAND Power Amplifier Temp
      Engineering units: degC (inferred)
      Conversion/status map: C0=0.000000e+00 C1=9.765625e-04
  xband_pa_current:
    value: (0 + xband_pa_current_raw * 0.001654473)
    doc: |
      XBAND Power Amplifier current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.654473e-03
  rail_3v3_voltage:
    value: (0 + rail_3v3_voltage_raw * 0.0016113)
    doc: |
      3p3 Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.611300e-03
  rail_3v3_current:
    value: (0 + rail_3v3_current_raw * 8.0566e-05)
    doc: |
      3p3 Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.056600e-05
  cdh_temp:
    value: (125.55 + cdh_temp_raw * (-0.13622 + cdh_temp_raw * (9.8611e-05 + cdh_temp_raw * (-4.4176e-08 + cdh_temp_raw * (1.0125e-11 + cdh_temp_raw * -9.3905e-16)))))
    doc: |
      CDH Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  cdh_3v3_reference_voltage:
    value: (0 + cdh_3v3_reference_voltage_raw * 0.0016113)
    doc: |
      CDH 3.3 V reference measurement.
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.611300e-03
  solar_array_8_cell_string_voltage:
    value: (0 + solar_array_8_cell_string_voltage_raw * 0.0096592)
    doc: |
      Solar Array 8-Cell String Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=9.659200e-03
  solar_array_8_cell_string_current:
    value: (0 + solar_array_8_cell_string_current_raw * 0.0020142)
    doc: |
      Solar Array 8-Cell String Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  solar_array_9_cell_string_voltage:
    value: (0 + solar_array_9_cell_string_voltage_raw * 0.0096592)
    doc: |
      9-Cell String Solar Array Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=9.659200e-03
  solar_array_9_cell_string_current:
    value: (0 + solar_array_9_cell_string_current_raw * 0.0020142)
    doc: |
      9-Cell String Solar Array Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  battery_1_voltage:
    value: (0 + battery_1_voltage_raw * 0.008056641)
    doc: |
      Battery 1 Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.056641e-03
  battery_2_voltage:
    value: (0 + battery_2_voltage_raw * 0.008056641)
    doc: |
      Battery 2 Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.056641e-03
  eps_temp:
    value: (125.55 + eps_temp_raw * (-0.13622 + eps_temp_raw * (9.8611e-05 + eps_temp_raw * (-4.4176e-08 + eps_temp_raw * (1.0125e-11 + eps_temp_raw * -9.3905e-16)))))
    doc: |
      EPS Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  eps_3v3_reference_voltage:
    value: (0 + eps_3v3_reference_voltage_raw * 0.00161133)
    doc: |
      EPS 3.3 V reference measurement.
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.611330e-03
  eps_bus_voltage:
    value: (0 + eps_bus_voltage_raw * 0.0088623)
    doc: |
      EPS Bus Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  eps_bus_current:
    value: (0 + eps_bus_current_raw * 0.0012207)
    doc: |
      EPS Bus Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
  xact_voltage:
    value: (0 + xact_voltage_raw * 0.0088623)
    doc: |
      XACT Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  xact_current:
    value: (0 + xact_current_raw * 0.0020142)
    doc: |
      XACT Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  uhf_voltage:
    value: (0 + uhf_voltage_raw * 0.0088623)
    doc: |
      UHF Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  uhf_current:
    value: (0 + uhf_current_raw * 0.0020142)
    doc: |
      UHF Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  xband_voltage:
    value: (0 + xband_voltage_raw * 0.0088623)
    doc: |
      XBAND Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=8.862300e-03
  xband_current:
    value: (0 + xband_current_raw * 0.0020142)
    doc: |
      XBAND Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.014200e-03
  csie_voltage:
    value: (0 + csie_voltage_raw * 0.0039639)
    doc: |
      CSIE Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=3.963900e-03
  csie_current:
    value: (0 + csie_current_raw * 0.0012207)
    doc: |
      CSIE Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
  dsps_voltage:
    value: (0 + dsps_voltage_raw * 0.0039639)
    doc: |
      DSPS Voltage
      Engineering units: V (inferred)
      Conversion/status map: C0=0.000000e+00 C1=3.963900e-03
  dsps_current:
    value: (0 + dsps_current_raw * 0.0012207)
    doc: |
      DSPS Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.220700e-03
  ifb_therm1:
    value: (125.55 + ifb_therm1_raw * (-0.13622 + ifb_therm1_raw * (9.8611e-05 + ifb_therm1_raw * (-4.4176e-08 + ifb_therm1_raw * (1.0125e-11 + ifb_therm1_raw * -9.3905e-16)))))
    doc: |
      Interface Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  sa_minus_y_temp:
    value: (125.55 + sa_minus_y_temp_raw * (-0.13622 + sa_minus_y_temp_raw * (9.8611e-05 + sa_minus_y_temp_raw * (-4.4176e-08 + sa_minus_y_temp_raw * (1.0125e-11 + sa_minus_y_temp_raw * -9.3905e-16)))))
    doc: |
      Solar Array 1 Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  sa_plus_y_temp:
    value: (125.55 + sa_plus_y_temp_raw * (-0.13622 + sa_plus_y_temp_raw * (9.8611e-05 + sa_plus_y_temp_raw * (-4.4176e-08 + sa_plus_y_temp_raw * (1.0125e-11 + sa_plus_y_temp_raw * -9.3905e-16)))))
    doc: |
      Solar Array 2 Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  csie_temp:
    value: (125.55 + csie_temp_raw * (-0.13622 + csie_temp_raw * (9.8611e-05 + csie_temp_raw * (-4.4176e-08 + csie_temp_raw * (1.0125e-11 + csie_temp_raw * -9.3905e-16)))))
    doc: |
      CSIE Detector Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  battery_1_temp:
    value: (125.55 + battery_1_temp_raw * (-0.13622 + battery_1_temp_raw * (9.8611e-05 + battery_1_temp_raw * (-4.4176e-08 + battery_1_temp_raw * (1.0125e-11 + battery_1_temp_raw * -9.3905e-16)))))
    doc: |
      Battery 1 Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  batt_board_temp:
    value: (125.55 + batt_board_temp_raw * (-0.13622 + batt_board_temp_raw * (9.8611e-05 + batt_board_temp_raw * (-4.4176e-08 + batt_board_temp_raw * (1.0125e-11 + batt_board_temp_raw * -9.3905e-16)))))
    doc: |
      Battery Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=1.255500e+02 C1=-1.362200e-01 C2=9.861100e-05 C3=-4.417600e-08 C4=1.012500e-11 C5=-9.390500e-16
  battery_1_charge_current:
    value: (0 + battery_1_charge_current_raw * 0.0002366954)
    doc: |
      Battery 1 Charge Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.366954e-04
  battery_2_charge_current:
    value: (0 + battery_2_charge_current_raw * 0.0002366954)
    doc: |
      Battery 2 Charge Current
      Engineering units: A (inferred)
      Conversion/status map: C0=0.000000e+00 C1=2.366954e-04
  dsps_sensor_board_temp:
    value: (0 + dsps_sensor_board_temp_raw * 0.01)
    doc: |
      Dual-SPS Sensor Board Temperature
      Engineering units: degC (inferred)
      Conversion/status map: C0=0.000000e+00 C1=1.000000e-02
  adcs_ana_motor1_temp:
    value: (0 + adcs_ana_motor1_temp_raw * 0.005)
    doc: |
      Wheel 1 Temp
      Engineering units: degC (inferred)
      Conversion/status map: C0=0.000000e+00 C1=5.000000e-03
  adcs_wheel_speed_1:
    value: (0 + adcs_wheel_speed_1_raw * 0.002)
    doc: |
      ADCS Wheel Speed 1
      Engineering units: rpm
      Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
  adcs_wheel_speed_2:
    value: (0 + adcs_wheel_speed_2_raw * 0.002)
    doc: |
      ADCS Wheel Speed 2
      Engineering units: rpm
      Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
  adcs_wheel_speed_3:
    value: (0 + adcs_wheel_speed_3_raw * 0.002)
    doc: |
      ADCS Wheel Speed 3
      Engineering units: rpm
      Conversion/status map: C0=0.000000e+00 C1=2.000000e-03
  adcs_sun_point_angle_error:
    value: (0 + adcs_sun_point_angle_error_raw * 0.003)
    doc: |
      Angle between the estimated and commanded Sun vectors.
      Engineering units: deg
      Conversion/status map: C0=0.000000e+00 C1=3.000000e-03
  dsps_flare_magnitude:
    value: (0 + dsps_flare_magnitude_raw * 0.1)
    doc: |
      Dual-SPS estimate of GOES XRS-B flare magnitude
      Engineering units: log10(XRS-B flux)
      Conversion/status map: C0=0.000000e+00 C1=1.000000e-01
enums:
  mode_system_mode_values:
    0: 'phoenix'
    1: 'safe'
    2: 'science'
    3: 'downlink'
  fault_protection_task_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_6_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_5_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_4_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_3_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_2_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_1_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  fault_protection_watchpoint_0_state_values:
    0: 'disabled'
    1: 'passive'
    2: 'enabled'
  dsps_flare_phase_values:
    0: 'not_in_sun'
    1: 'filling_history'
    2: 'not_in_flare'
    4: 'flare_likely'
    24: 'in_flare_decreasing'
    40: 'in_flare_rising'
  battery_1_charging_state_values:
    1: 'charging'
    0: 'discharging'
  eps_pwr_state_dsps_values:
    0: 'off'
    1: 'on'
  eps_pwr_state_csie_values:
    0: 'off'
    1: 'on'
  eps_pwr_state_xband_values:
    0: 'off'
    1: 'on'
  eps_pwr_state_uhf_values:
    0: 'off'
    1: 'on'
  eps_pwr_state_adcs_values:
    0: 'off'
    1: 'on'
  telescope_door_pin_pulled_values:
    1: 'engaged'
    0: 'pulled'
  csie_heater_enable_values:
    0: 'no'
    1: 'yes'
  battery2_heater_enable_values:
    0: 'no'
    1: 'yes'
  battery1_heater_enable_values:
    0: 'no'
    1: 'yes'
  uhf_alive_values:
    0: 'off'
    1: 'alive'
    2: 'dead'
  adcs_att_valid_values:
    0: 'no'
    1: 'yes'
  adcs_ref_valid_values:
    0: 'no'
    1: 'yes'
  adcs_time_valid_values:
    1: 'yes'
    0: 'no'
  adcs_mode_values:
    0: 'sun_point'
    1: 'fine_ref_point'
  battery_2_charging_state_values:
    1: 'charging'
    0: 'discharging'
  adcs_sun_point_state_values:
    0: 'sun_point'
    1: 'fine_ref_point'
    2: 'search_init'
    3: 'searching'
    4: 'waiting'
    5: 'converging'
    6: 'on_sun'
    7: 'not_actv'
  xband_data_source_values:
    0: 'test_pat'
    1: 'cdh'
