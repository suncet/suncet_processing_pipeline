# SunCET Public Beacon Specification

Status: **Draft**

Revision: draft-0.11
Last updated: 2026-09-29

Canonical URL:
<https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md>

## Purpose and scope

This document provides the basic information to receive and decode the SunCET UHF beacons.

**TBC** indicates a parameter that has not yet been confirmed.

## Mission

| Item | Value |
| --- | --- |
| Spacecraft | SunCET |
| Expanded name | Sun Coronal Ejection Tracker |
| Form factor | 6U SmallSat/CubeSat |
| Mission status | Future |
| Mission purpose | Extreme-ultraviolet observations of coronal mass-ejection acceleration from the low corona into the extended corona |
| Lead institutions | Johns Hopkins Applied Physics Laboratory and University of Colorado Boulder Laboratory for Atmospheric and Space Physics |
| Funding program | NASA Heliophysics |
| Country | United States |
| Launch/deployment | No earlier than 2027-03-15; manifested on SpaceX Falcon 9 launch |
| Expected orbit | 510 km circular, Sun-synchronous, 18:00 mean local time at ascending node |
| Prime mission | 8 months |
| Primary website | <https://suncet.jhuapl.edu/> |
| Secondary website | <https://lasp.colorado.edu/missions/suncet/> |
| Public contact | `james.mason@jhuapl.edu` — Mission PI |
| Public image | [SunCET spacecraft with deployed solar arrays](assets/suncet_spacecraft.jpg); unrestricted public use with mission/JHUAPL attribution |

Proposed short SatNOGS description:

> SunCET is a 6U NASA Heliophysics CubeSat jointly developed by the Johns
> Hopkins Applied Physics Laboratory and the University of Colorado Boulder's
> Laboratory for Atmospheric and Space Physics. Its extreme-ultraviolet imager
> is designed to observe how coronal mass ejections accelerate from the low
> solar corona into the extended corona.

## UHF transmitter

| Parameter | Flight value |
| --- | --- |
| Downlink center frequency | 401.200 MHz |
| Authorized frequency range | 401.1904-401.2096 MHz |
| Frequency tolerance | 0.0001% (1 ppm, or approximately +/-401.2 Hz at the assigned center frequency) |
| Emission designator | `19K2F1D` |
| Modulation | GFSK |
| Symbol/baud rate | 9600 baud nominal; 19200 baud contingency mode if UHF science playback is required |
| Authorized/declared occupied bandwidth | 19.2 kHz (`19K2`) |
| Forward-error correction | **TBC** |
| Polarization | RHCP |
| Flight radio and antenna | SpaceQuest TRX-U with GomSpace NanoCom ANT-6F |
| Radio output / licensed ERP | 2.0 W transmitter output; 1.53 W authorized ERP for the space station |
| Beacon cadence | Mode dependent, of order 10 seconds |
| Spectrum service | FCC Experimental Radio Service |
| SatNOGS service category | Space Operation (spacecraft health telemetry) |
| Coordination/authorization | FCC call sign `WP2XUX`, file `0244-EX-CN-2025`; effective 2025-09-17 and expiring 2027-10-01 |

## Link framing

Flight software constructs the following radio-interface frame buffer:

| Item | Confirmed software-side value |
| --- | --- |
| Link layer | AX.25 UI frame |
| Destination callsign | `LASP-0` |
| Source callsign | `SUN1-0` |
| AX.25 control | `0x03` |
| AX.25 PID | `0xF0` |
| AX.25 FCS | CRC-16/X-25 over header and information field; transmitted low byte first |

The literal 16-byte header is destination
`98 82 a6 a0 40 40 41`, source `a6 aa 9c 62 40 40 41`, control `03`, and PID
`f0`. The character octets decode to the bit-shifted callsigns `LASP  ` and
`SUN1  `. Both `0x41` SSID octets are intentional and remain unchanged on the
way to the radio, even though setting the address-extension bit in both address
octets is unconventional for a two-address AX.25 frame.

Flight software copies the CCSDS packet immediately after that header, computes
the FCS over every header and payload byte, appends two FCS bytes, and gives the
complete buffer to the radio. The FCS parameters are width 16, reflected
polynomial `0x8408` (the reflected representation of `0x1021`), initial value
`0xffff`, reflected input/output, and final complement `0xffff`. The CRC routine
performs a final byte swap and the caller writes the returned high byte followed
by its low byte; together these operations place the conventional un-swapped
CRC-16/X-25 value low byte first in the transmitted buffer. Flags are not part
of the CRC input.

An RF capture must still establish any radio-added preamble, flags, bit
stuffing, or other physical/link framing. Separately, laboratory validation of
the selected SatNOGS receiver path must establish whether that ground-side path
removes flags, the AX.25 header, or FCS before passing bytes to the telemetry
decoder.

## CCSDS APID 1 packet

All multi-byte APID 1 fields are decoded in big-endian byte order. The
packet begins with a standard six-byte CCSDS Space Packet primary header with
the secondary-header flag set and APID equal to 1.

### Current packet length

CTDB 2.0.5 defines a 252-byte APID 1 packet (2016 bits). The CCSDS length
field is 245, following the standard total-length-minus-seven convention.
Fletcher-32 occupies offsets 248 through 251 and covers bytes 0 through 247.
Byte 247 is consumed opaquely, as are other regions outside the public table.
It participates in checksum validation regardless of its value.

### Secondary time header

| Offset after CCSDS primary header | Size | Meaning |
| --- | --- | --- |
| 0 | 4 bytes | Coarse seconds since `2000-01-01T00:00:00Z`, big endian |
| 4 | 2 bytes | Integer milliseconds after the coarse whole second, big endian; valid range 0-999 |

### Packet checksum

Beacon packets use the following Fletcher-32 variant:

- Compute over every packet byte before the final four checksum bytes.
- Interpret successive input words little endian.
- Initialize both Fletcher accumulators to `0xffff` and reduce modulo `0xffff`.
- Store the resulting 32-bit value big endian.

### Public field table

The machine-readable table is
[`public_beacon_schema.csv`](../suncet_processing_pipeline/satnogs/public_beacon_schema.csv).
It is ordered by CTDB bit offset and includes public names, descriptions,
types, units, conversions, and status maps.

### Decoder and synthetic vector

The generated
[`suncet_apid1.ksy`](../suncet_processing_pipeline/satnogs/suncet_apid1.ksy)
does not yet wrap the packet in AX.25 because laboratory validation must first
establish the actual SatNOGS decoder-input boundary.

The repository also contains a fully synthetic, non-flight
[`252-byte packet`](../suncet_processing_pipeline/satnogs/test_data/suncet_apid1_synthetic_252.hex)
and its
[`expected public values`](../suncet_processing_pipeline/satnogs/test_data/suncet_apid1_synthetic_252_expected.json).

### CSIE image histogram fields

CSIE (Compact Spectral Imager Electronics) serves SunCET's primary instrument,
the extreme-ultraviolet (EUV) imager. Its firmware calculates histogram bins
after subtracting the configurable `ICM_HIST_OFFSET` from each pixel value.
For bin index `i`, offset `O`, and bin width `W`, the corresponding original
pixel-DN range is:

`O + i*W` through `O + (i+1)*W - 1`, inclusive.

The default settings are `O=0` and `W=32`. APID 1 cannot carry the complete
histogram, so it transmits only the first six pixel-count bins:

| Public field | Default pixel-DN range |
| --- | --- |
| `csie_img_hist_0` | 0-31 |
| `csie_img_hist_1` | 32-63 |
| `csie_img_hist_2` | 64-95 |
| `csie_img_hist_3` | 96-127 |
| `csie_img_hist_4` | 128-159 |
| `csie_img_hist_5` | 160-191 |

These ranges must be recomputed if the flight configuration changes either
firmware setting; the beacon values remain counts and still represent bins 0-5.

### Dual-SPS flare fields

Dual-SPS (Dual Sun Position Sensor) is SunCET's secondary instrument.
The public decoder follows CTDB 2.0.5 for these fields:

- `dsps_flare_level` is the flare-trigger threshold in log10 of estimated GOES
  XRS-B flux. The default is -5 (M1); the documented range is -6 (C1) through
  -2 (X100).
- `dsps_flare_magnitude` is an unsigned 8-bit raw value with no engineering
  conversion specified.
- `dsps_flare_phase` is a bit-flag state: 0 not in Sun, 1 filling history, 2 not
  in flare, 4 flare start, 24 declining flare, and 40 rising flare. These are
  exposed using the current CTDB state labels.

## Public sources

- [APL SunCET mission page](https://www.jhuapl.edu/destinations/missions/suncet)
- [LASP SunCET mission page](https://lasp.colorado.edu/missions/suncet/)
- [NASA SunCET selection announcement](https://www.nasa.gov/science-research/heliophysics/nasa-selects-4-cubesats-for-space-weather-tech-development/)
