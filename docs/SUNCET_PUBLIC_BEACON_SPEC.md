# SunCET Public Beacon Specification

Status: **Pre-publication draft — not yet authoritative**

Revision: draft-0.4
Last updated: 2026-08-21

Canonical URL:
<https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md>

The canonical URL is stable across revisions. The status and revision above
determine whether the displayed document is an authoritative release or a
working draft.

## Purpose and scope

This document will provide the public information needed to receive and decode
the globally broadcast SunCET UHF beacon. It covers only the CCSDS APID 1 beacon.
It does not document commanding, private ground-system interfaces, stored-data
playback, science-image transport, or any other spacecraft APID.

Values marked **TBC** must be confirmed against the configured flight radio or
flight software before this document is used as a SatNOGS citation.

## Mission

| Item | Value | Status/source |
| --- | --- | --- |
| Spacecraft | SunCET | Confirmed by APL and LASP public mission pages |
| Expanded name | Sun Coronal Ejection Tracker | Confirmed by the mission PI; the LASP public page should eventually be corrected |
| Form factor | 6U SmallSat/CubeSat | Public LASP mission page |
| Mission status | Future | Public LASP mission page |
| Mission purpose | Extreme-ultraviolet observations of coronal mass-ejection acceleration from the low corona into the extended corona | Public APL and LASP mission pages |
| Lead institutions | Johns Hopkins Applied Physics Laboratory and University of Colorado Boulder Laboratory for Atmospheric and Space Physics | Public APL and LASP mission pages |
| Funding program | NASA Heliophysics | Public LASP mission page |
| Country | United States | Derived from the lead institutions; confirm SatNOGS entry convention |
| Launch/deployment | No earlier than 2027-02-15 | Current mission planning date; update when manifested |
| Expected orbit | 510 km circular, Sun-synchronous, 18:00 mean local time at ascending node | Current mission planning orbit |
| Prime mission | 8 months | Confirmed by the mission PI |
| Primary website | <https://suncet.jhuapl.edu/> | Confirmed by the mission PI |
| Secondary website | <https://lasp.colorado.edu/missions/suncet/> | Use when SatNOGS provides an appropriate secondary-link field |
| Public contact | `james.mason@jhuapl.edu` | Confirmed by the mission PI |
| Public image | `Spacecraft.jpg`, showing the integrated spacecraft with deployed solar arrays | Mission image selected by the PI with permission for unrestricted public use; preserve mission/JHUAPL attribution |

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
| Modulation | GFSK; confirm the exact compatible SatNOGS mode label from an RF recording |
| Symbol/baud rate | 9600 baud nominal; 19200 baud contingency mode if UHF science playback is required |
| Frequency deviation | **TBC** |
| Authorized/declared occupied bandwidth | 19.2 kHz (`19K2`) |
| Pulse shaping/filter | **TBC** |
| Forward-error correction | None in the FCC technical submission; verify against the final programmed flight configuration |
| Interleaving | **TBC** |
| Whitening/scrambling | **TBC** |
| Polarization | RHCP |
| Flight radio and antenna | SpaceQuest TRX-U with GomSpace NanoCom ANT-6F |
| Radio output / licensed ERP | 2.0 W transmitter output; 1.53 W authorized ERP for the space station |
| Beacon cadence | Mode dependent, of order 10 seconds |
| Spectrum service | FCC Experimental Radio Service; map this to the closest current SatNOGS service vocabulary at submission time |
| Coordination/authorization | FCC call sign `WP2XUX`, file `0244-EX-CN-2025`; effective 2025-09-17 and expiring 2027-10-01 |

The values must describe the programmed flight configuration, not merely the
capabilities or defaults of the SpaceQuest TRX-U radio.

The FCC filing describes the spacecraft UHF link at 19200 bit/s, while current
mission planning calls for 9600 baud as the normal beacon rate with 19200 baud
available as a contingency. Both supported configurations must be validated
from flight-equivalent RF recordings; the filing's maximum/configured value
does not override the current operational default.

The current no-earlier-than launch date and eight-month prime mission extend
past the authorization's 2027-10-01 expiration. The mission therefore needs a
renewed or modified authorization before operations continue beyond that date,
and likely before launch if the schedule slips materially.

## Link framing

The processing repository contains the following candidate values from earlier
ground-system integration. They are not yet asserted as flight over-air values:

| Item | Candidate value |
| --- | --- |
| Link layer | AX.25 UI frame |
| Destination callsign | `LASP-0` |
| Source callsign | `SUN1-0` |
| AX.25 control | `0x03` |
| AX.25 PID | `0xF0` |
| AX.25 FCS | Presence, byte order, and SatNOGS stripping behavior **TBC** |

The flight-software constants show the destination character bytes as
`98 82 a6 a0 40 40` (bit-shifted `LASP  `) and source character bytes as
`a6 aa 9c 62 40 40` (bit-shifted `SUN1  `). Both shown SSID octets are `0x41`,
which gives an SSID value of zero but is unusual for a conventional two-address
AX.25 header because both octets have the address-extension bit set and the
reserved bits are not the conventional pair. The flight-software/radio
integration team must confirm whether these are literal radio-interface bytes
or macros adjusted during frame assembly, along with the C/H and reserved bits
and control/PID. Radio documentation or an RF capture must establish what the
radio actually transmits, including flags and FCS. Separately, laboratory
validation of the selected SatNOGS receiver path must establish whether that
ground-side path removes flags or FCS before passing the frame to the telemetry
decoder. Those are distinct questions.

The APID 1 beacon is carried directly in the AX.25 information field. The FSW
2.0.4 user's guide states that only UHF packets larger than the 256-byte AX.25
payload limit are segmented. The APID 72/73 stored-playback segmentation used
elsewhere in SunCET is therefore not part of the ordinary public beacon
protocol. The complete on-air framing must still be verified with an RF capture
from the flight-equivalent transmitter.

## CCSDS APID 1 packet

All multi-byte APID 1 fields are currently decoded in big-endian byte order. The
packet begins with a standard six-byte CCSDS Space Packet primary header with
the secondary-header flag set and APID equal to 1.

The current telemetry definition contains 136 logical rows, including CCSDS
header fields and packed bit fields. The public decoder will expose a reviewed
subset of spacecraft time, power, thermal, mode, ADCS, payload, storage, radio,
and fault/status telemetry. It will not expose command opcode names, command
counters, command status, command arm states, other uplink-related values, or
unrelated APID definitions. Bytes occupied by excluded fields will be consumed
opaquely so later public fields retain their correct offsets.

### Packet-length discrepancy to resolve

- The SunCET Bus CTDB 2.0.1 summary declares 2008 bits, or 251 bytes total.
- Current flight-model UHF test data contains consecutive, checksum-valid APID 1
  packets whose CCSDS length field declares 252 bytes total.
- In the observed packet, an additional byte appears before the final four-byte
  checksum relative to the generated CTDB decoder's offsets.

Flight software must confirm whether this byte is an intentional spare/padding
field and whether 252 bytes is the flight format. The final field table and
Kaitai decoder must follow observed flight framing and the corrected
authoritative definition.

### Secondary time header

| Offset after CCSDS primary header | Size | Meaning |
| --- | --- | --- |
| 0 | 4 bytes | J2000 coarse seconds, big endian |
| 4 | 2 bytes in the current CTDB | Microseconds after the coarse whole second, big endian; serialized width/encoding conflict **TBC** |

The FSW 2.0.4 user's guide defines the epoch as `2000-01-01T00:00:00` and
describes the secondary header as seconds and microseconds elapsed from that
epoch. Flight software has now confirmed the intended interpretation as coarse
seconds since `2000-01-01T00:00:00Z` plus microseconds after that whole second.
This supersedes the pipeline's current binary-fraction interpretation
(`fine / 65536`) and its extra post-J2000 leap-second correction.

One serialization question remains: a literal microsecond-of-second value
ranges from 0 through 999999 and therefore needs at least 20 bits, while the
current CTDB and decoder allocate only 16 bits. Before changing production time
conversion, flight software must identify whether the transmitted field is
actually wider, scaled into 16 bits, truncated, or otherwise encoded. Until
then, the public contract exposes the fine value raw and does not manufacture a
UTC timestamp from it.

### Packet checksum

Observed APID 1 test packets validate using the pipeline's Fletcher-32 variant:

- Compute over every packet byte before the final four checksum bytes.
- Interpret successive input words little endian.
- Initialize both Fletcher accumulators to `0xffff` and reduce modulo `0xffff`.
- Store the resulting 32-bit value big endian.

The FSW 2.0.4 user's guide confirms the checksum coverage but does not document
the word order, accumulator seed, or stored byte order. Repeated successful APID
1 decoding and checksum validation make the observed packets and current
pipeline implementation the working authority for these implementation
details. A sanitized published test vector remains desirable as a regression
and interoperability artifact, but flight-software confirmation is no longer a
blocker for the algorithm itself.

### Public field table

Mission-owner review approved 112 of the 136 logical APID 1 fields for public
decoding. The remaining 24 fields are consumed opaquely and are not exposed:
command opcodes, command counters/status, command arm states, subsystem command
statistics, internal sequence state, the fault-protection response count, and
the stored packet checksum value. The checksum is still validated internally.

The reviewed machine-readable table is
[`public_beacon_schema.csv`](../suncet_processing_pipeline/satnogs/public_beacon_schema.csv).
It is ordered by authoritative CTDB bit offset and includes public names,
descriptions, types, units, conversions, and status maps. NAND read/write
pointers are intentionally public because their progression provides useful
evidence that recording and stored-data playback are functioning.

One published field retains technical-definition debt without reopening the
publication decision:

- `spacecraft_time_fine_raw` remains raw until the 16-bit serialization is
  reconciled with the confirmed microsecond semantics.

### Provisional decoder and synthetic vector

The generated
[`suncet_apid1.ksy`](../suncet_processing_pipeline/satnogs/suncet_apid1.ksy)
is a provisional Kaitai decoder for a bare CCSDS APID 1 packet. It exposes all
112 approved fields, consumes the 24 excluded fields as anonymous gaps, leaves
fine time raw, rejects a non-APID-1 CCSDS primary word, and accepts either the
251- or 252-byte packet form. It deliberately does not guess the unresolved
AX.25 wrapper.

The repository also contains a fully synthetic, non-flight
[`251-byte packet`](../suncet_processing_pipeline/satnogs/test_data/suncet_apid1_synthetic_251.hex)
and its
[`expected public values`](../suncet_processing_pipeline/satnogs/test_data/suncet_apid1_synthetic_251_expected.json).
They provide a safe regression and interoperability fixture, but they do not
replace validation against a flight-equivalent AX.25 frame and RF recording.

### CSIE image histogram fields

The CSIE firmware calculates histogram bins after subtracting the configurable
`ICM_HIST_OFFSET` from each pixel value. For bin index `i`, offset `O`, and bin
width `W`, the corresponding original pixel-DN range is:

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

The Dual-SPS telemetry handbook v1.11 and flight source establish these public
interpretations:

- `dsps_flare_level` is the flare-trigger threshold in log10 of estimated GOES
  XRS-B flux. The default is -5 (M1); the documented range is -6 (C1) through
  -2 (X100).
- `dsps_flare_magnitude` is a signed 8-bit value. Multiplying the raw value by
  0.1 yields the log10 XRS-B flux estimate; equivalently, the estimated linear
  intensity is `10 ** (raw / 10)` in the calibration's XRS-B flux units.
- `dsps_flare_phase` is a bit-flag state: 0 not in Sun, 1 filling history, 2 not
  in flare, 4 flare likely, 24 in-flare decreasing, and 40 in-flare rising.

These findings corrected the private CTDB's flare-magnitude signedness and
scaling. The in-flight GOES conversion coefficients still require calibration,
so the value is an estimate rather than an independently calibrated radiometer
measurement.

## Required validation artifacts

Before promoting this draft to an authoritative revision, retain the following
non-sensitive flight-equivalent artifacts with this specification or in the
SatNOGS decoder test data:

1. A raw AX.25 frame captured from the flight-equivalent RF path.
2. The extracted APID 1 bytes.
3. Independently verified expected raw and engineering values.
4. A checksum calculation with the expected result.
5. Receiver settings and a short representative IQ or audio recording when
   licensing and size permit.

## Public sources

- [APL SunCET mission page](https://www.jhuapl.edu/destinations/missions/suncet)
- [LASP SunCET mission page](https://lasp.colorado.edu/missions/suncet/)
- [NASA SunCET selection announcement](https://www.nasa.gov/science-research/heliophysics/nasa-selects-4-cubesats-for-space-weather-tech-development/)
- SunCET FSW User's Guide, prerelease for FSW 2.0.4 (mission-controlled source;
  stable public URL **TBC**)
