# SunCET SatNOGS dashboard preparation

Last updated: 2026-09-29

Status: **Local design only. No live dashboard has been created. This plan has
not been validated against recorded telemetry or SatNOGS ingestion.**

## Scope and data contract

Prepare a compact public APID 1 health dashboard in SatNOGS Grafana. The
[public schema](../suncet_processing_pipeline/satnogs/public_beacon_schema.csv)
defines the permitted field aliases, conversions, units, and state maps. The
panel candidates below use approved CTDB 2.0.5 aliases. The decoder supports
only this version; matching packet length does not establish the correct
layout. Do not query a legacy layout as though it were current telemetry.

Bind panels to the public aliases and apply only their defined engineering
conversions; aliases without a conversion retain their raw values. Do not
rescale decoder output a second time. Preserve any
schema unit marked inferred in the panel description. Actual datasource,
measurement names, tags, and query paths must be discovered after editor access
and ingestion are available; the schema aliases are not yet verified Grafana
query identifiers. Include the decoder/schema revision in dashboard notes.

Only the approved public beacon fields belong here. Reception timestamps,
observation/station IDs, and decoder revision are provenance from the receiving
system, not additional spacecraft telemetry. Link those values when the actual
datasource exposes them.

## Initial panel mapping

| Group / panel | Public aliases | Display and interpretation |
| --- | --- | --- |
| Identity and spacecraft state | `ccsds_apid`, `ccsds_sequence_count`, `mode_system_mode`, `mode_seconds_since_mode_change`, `num_sc_resets` | Latest-value table plus mode timeline. APID must be 1. Label modes PHOENIX/SAFE/SCIENCE/DOWNLINK using the schema; a sequence discontinuity alone does not prove packet loss or reboot. |
| Spacecraft clock and uptime | `spacecraft_time_seconds_since_2000`, `spacecraft_time_milliseconds`, `time_since_boot`, `time_alive`, `time_mission_elapsed_time` | Raw coarse/fine clock and elapsed seconds, with human-readable uptime as a secondary display. Follow the time policy below. |
| Battery voltage and charging | `battery_1_voltage`, `battery_2_voltage`, `battery_1_charge_current`, `battery_2_charge_current`, `battery_1_charging_state`, `battery_2_charging_state` | Separate voltage (V) and current (A) plots; CHARGING/DISCHARGING state timeline. No inferred state-of-charge percentage. |
| Power buses and solar arrays | `eps_bus_voltage`, `eps_bus_current`, `rail_3v3_voltage`, `rail_3v3_current`, `solar_array_8_cell_string_voltage`, `solar_array_8_cell_string_current`, `solar_array_9_cell_string_voltage`, `solar_array_9_cell_string_current` | Grouped time series, separate V and A axes/panels. No flight alarm thresholds until reviewed limits exist. |
| Bus and battery thermal | `cdh_temp`, `eps_temp`, `ifb_therm1`, `battery_1_temp`, `batt_board_temp`, `sa_minus_y_temp`, `sa_plus_y_temp` | Temperature time series in degrees C with per-sample tooltips. The schema contains no battery-2 temperature alias; do not invent one. |
| Payload and radio thermal | `csie_temp`, `dsps_sensor_board_temp`, `uhf_temp`, `xband_pa_temp`, `adcs_ana_motor1_temp` | Separate subsystem temperature series; preserve inferred-unit caveats. |
| ADCS validity and pointing | `adcs_att_valid`, `adcs_ref_valid`, `adcs_time_valid`, `adcs_mode`, `adcs_sun_point_state`, `adcs_sun_point_angle_error` | State timeline and angular-error plot in degrees. Display validity beside the measurements; do not label a numerical pointing estimate valid when flags disagree. |
| ADCS motion | `adcs_body_rate_1`, `adcs_body_rate_2`, `adcs_body_rate_3`, `adcs_wheel_speed_1`, `adcs_wheel_speed_2`, `adcs_wheel_speed_3` | Separate rad/s and rpm plots, preserving signed values. |
| Payload activity and power | `csie_capture_state`, `eps_pwr_state_csie`, `eps_pwr_state_dsps`, `csie_voltage`, `csie_current`, `dsps_voltage`, `dsps_current` | ON/OFF state timeline plus V/A plots. Capture-state codes have no approved names in the schema; show their numeric values. |
| Dual-SPS flare telemetry | `dsps_flare_level`, `dsps_flare_magnitude`, `dsps_flare_phase` | Threshold in log10 estimated XRS-B flux; magnitude is an unsigned 8-bit raw value with no CTDB 2.0.5 engineering conversion. Plot these separately. Phase labels include FLARE_START, DECLINING_FLARE, and RISING_FLARE. Do not convert magnitude to GOES flux. |
| CSIE histogram | `csie_img_hist_0`, `csie_img_hist_1`, `csie_img_hist_2`, `csie_img_hist_3`, `csie_img_hist_4`, `csie_img_hist_5` | Six-bin bar chart from one packet, in pixel counts. Show bin indices; describe the default DN ranges 0–31 through 160–191 only while offset=0 and width=32 are confirmed. These six bins are not the full image histogram. |
| Radio health | `uhf_alive`, `eps_pwr_state_uhf`, `eps_pwr_state_xband`, `uhf_voltage`, `uhf_current`, `xband_voltage`, `xband_current`, `xband_pa_current`, `xband_data_source` | OFF/ALIVE/DEAD and power timelines plus V/A plots; source labels TEST_PAT/CDH. These fields do not measure RF reception quality. |
| Fault-protection status | `fault_protection_task_state`, `fault_protection_watchpoint_0_state`, `fault_protection_watchpoint_1_state`, `fault_protection_watchpoint_2_state`, `fault_protection_watchpoint_3_state`, `fault_protection_watchpoint_4_state`, `fault_protection_watchpoint_5_state`, `fault_protection_watchpoint_6_state` | State table/timeline with DISABLED/PASSIVE/ENABLED mappings. Preserve watchpoint numbers; do not invent meanings or imply that ENABLED means a fault occurred. |

Use numeric codes alongside state labels. An unrecognized enumeration must
display as unknown with its numeric value, never as a default healthy state.
Keep the reception-age/coverage banner visible above all groups.

## Time, gaps, and freshness

- Use a verified reception/acquisition UTC timestamp from the SatNOGS source
  as the initial plot axis, and label that time basis explicitly. Keep any
  ingestion timestamp separate. If the datasource's timestamp semantics are
  unknown, resolve them before displaying plots as UTC observations.
- Show the spacecraft coarse and fine wire values. Their combination is coarse
  seconds plus fine milliseconds divided by 1000; fine time must be 0–999.
  Do not label this as a fully validated spacecraft UTC conversion while the
  epoch/leap-second policy remains under review in the
  [public beacon specification](SUNCET_PUBLIC_BEACON_SPEC.md#secondary-time-header).
- Display the last valid reception time and sample age. Use **No data** when
  the selected range contains no valid samples and **Data unavailable** for a
  query/source failure. Neither state is zero, OFF, or healthy telemetry.
- Preserve permitted wire NaN/Inf values in decoder output. Render non-finite
  float readings as missing measurements, with a non-finite-value indication
  where useful; never substitute zero or infer healthy telemetry. Such a field
  does not by itself invalidate the packet or its other measurements.
- Show **Stale** only against an explicit, configurable freshness threshold
  accepted after cadence and observation coverage are checked. Always show age
  even before that threshold is approved. A gap between scheduled observations
  does not by itself mean spacecraft failure.
- Do not connect lines across missing observations or carry forward a state
  indefinitely. Any held last-known value must carry its timestamp and stale
  indication. Histogram bins and grouped latest-value summaries must use the
  same packet, rather than independently selecting each field's latest value.
- Synthetic fixtures may support an explicitly labeled preview. They must never
  be mixed into live queries or presented as received SunCET telemetry.

## Obtain dashboard editor access

The official [SatNOGS Satellite Operator Guide](https://wiki.satnogs.org/Satellite_Operator_Guide#2.2_Add_a_new_Mission)
was checked on 2026-09-29. Its dashboard procedure is:

1. Open [the dashboard login](https://dashboard.satnogs.org/login), select
   **Sign-in with Auth0**, and use the intended Libre Space Foundation SSO
   account; register if necessary.
2. Sign in to the dashboard at least once so the account exists there. Initial
   access is read-only.
3. Create an issue in [satnogs-ops](https://gitlab.com/librespacefoundation/satnogs-ops/-/issues)
   identifying the satellite, mission team, and the email used for that login.
   Mark it **Confidential** so the login email is not posted publicly.
4. After SatNOGS grants editor permissions, create the mission dashboard.

The guide does not list owning a ground station as an editor-access
prerequisite. Network scheduling permission is a separate matter. This plan
does not imply that editor access has been requested or granted.

Copy-ready request; replace placeholders in the confidential issue, not here:

```text
Title: SunCET mission dashboard editor access

Please grant dashboard editor access for the SunCET mission.

Satellite: SunCET (Sun Coronal Ejection Tracker)
SatNOGS ID: MNRC-9829-4319-5529-8975
DB link: https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975
Satellite team: SunCET — Johns Hopkins APL / University of Colorado Boulder LASP
Requester's role and team membership: [mission role / approved maintainer details]
SSO login email: [email used to sign in to dashboard.satnogs.org]
Initial dashboard sign-in completed: [date]
Additional/backup editors: [names and login emails, if requesting access for them]

We are preparing a public health dashboard for the approved CCSDS APID 1 beacon
fields. The decoder and receiver integration are still being validated; the
initial dashboard will be clearly labeled during development. No private CTDB
definitions or commanding information will be published.

Mission website: https://suncet.jhuapl.edu/
Public beacon specification:
https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md
```

## Implementation and acceptance

After access is granted, inspect the actual datasource and accepted decoder
fields, create a development dashboard, and save its reproducible export in
the repository. Validate one recorded observation end to end: source packet,
selected layout, checksum result, decoded engineering values, observation
timestamp, and displayed values. Exercise an empty range, interrupted source,
unknown state, non-finite float, stale sample, and observation gap. Review units,
time semantics, and thresholds before describing the dashboard as operational
or adding its public URL to the satellite record. Receiver-path validation and upstream
decoder acceptance remain tracked in the
[onboarding plan](SATNOGS_ONBOARDING_PLAN.md).
