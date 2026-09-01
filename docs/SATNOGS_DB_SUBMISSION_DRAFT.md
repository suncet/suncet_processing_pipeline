# SunCET SatNOGS DB Submission Draft

Last updated: 2026-09-01

## Purpose

This is the offline copy deck for the SunCET spacecraft and nominal UHF
transmitter suggestions. It is not evidence that either suggestion has been
submitted or accepted. Recheck the live SatNOGS form vocabulary before entry.
The satellite suggestion may cite existing public mission pages; it does not
depend on completion of the receiver or telemetry-decoder specification.

## Spacecraft suggestion

| SatNOGS field | Proposed entry |
| --- | --- |
| NORAD ID | Leave blank before identification |
| Followed NORAD ID | Leave blank until there is a justified launch-object candidate |
| Name | `SunCET` |
| Other names | `Sun Coronal Ejection Tracker` |
| Description | SunCET is a 6U NASA Heliophysics CubeSat jointly developed by the Johns Hopkins Applied Physics Laboratory and the University of Colorado Boulder's Laboratory for Atmospheric and Space Physics. Its extreme-ultraviolet imager is designed to observe how coronal mass ejections accelerate from the low solar corona into the extended corona. |
| Owner/operator | Select JHU/APL and LASP/CU Boulder only if exact matching choices exist; otherwise leave this experimental field blank and retain the institutions in the description |
| Status | `Future` |
| Countries of origin | `United States` |
| Website | <https://suncet.jhuapl.edu/> |
| Dashboard URL | Leave blank until the SunCET SatNOGS dashboard exists |
| Launch date | `2027-02-15 00:00 UTC` as a no-earlier-than planning date; update or omit if the form cannot represent NET dates clearly |
| Deploy date | Leave blank until deployment timing is manifested |
| Re-entry date | Leave blank |
| Image | Upload [`assets/suncet_spacecraft.jpg`](assets/suncet_spacecraft.jpg), the resized metadata-free public copy |
| Citation | Public APL, LASP, and NASA mission pages; the beacon specification may be added but is not required for the spacecraft identity record |
| Email when reviewed | Yes |

Stable image URL after this file is merged to `main`:
<https://raw.githubusercontent.com/suncet/suncet_processing_pipeline/main/docs/assets/suncet_spacecraft.jpg>

## Nominal UHF transmitter suggestion

Create this suggestion only after the spacecraft record is accepted.

| SatNOGS field | Proposed entry |
| --- | --- |
| Description | SunCET nominal global health beacon: 401.200 MHz GFSK at 9600 baud, carrying an AX.25 UI frame whose information field contains a CCSDS APID 1 packet. Beacon cadence is spacecraft-mode dependent and is typically of order 10 seconds. |
| Type | `Transmitter`—the public entry intentionally describes only the downlink beacon and publishes no uplink parameters |
| Status | `Inactive` before verified on-orbit reception |
| Downlink frequency | `401200000` Hz |
| Downlink drift frequency | `401200000` Hz initially, representing zero observed correction; update from measured on-orbit drift rather than using licensed tolerance as drift |
| Downlink mode | `GFSK`; this is present in the current SatNOGS vocabulary, while the record remains `Unconfirmed` pending RF validation |
| Baud | `9600` |
| Service | Prefer `Space Research`; confirm with SatNOGS reviewers because the FCC Experimental Radio Service authorization category is not itself a SatNOGS service choice |
| IARU coordination | `N/A` because the link is outside the amateur bands |
| IARU coordination URL | Leave blank |
| ITU notification URLs | Leave blank until a public applicable entry is identified |
| Unconfirmed | Yes before flight-equivalent RF validation and on-orbit reception |
| Citation | Reviewed revision of the [SunCET public beacon specification](SUNCET_PUBLIC_BEACON_SPEC.md) |
| Email when reviewed | Yes |

Do not publish uplink frequency, uplink mode, commanding details, or command
telemetry through this record. Selecting `Transmitter` accurately describes the
public SatNOGS integration boundary even though the physical spacecraft radio
may support private mission uplink functions.

## Contingency 19200-baud mode

Do not combine two baud rates into one nominal transmitter entry. Start with
the planned 9600-baud beacon. If the 19200-baud contingency configuration is
validated and becomes operationally relevant, add a distinct transmitter
record with the same center frequency and mark only the configuration actually
being transmitted as active. This follows the SatNOGS model in which a change
that requires a different receiver configuration is represented separately.

## Submission gates

Before submitting the satellite suggestion:

1. Recheck the current form choices for owner/operator and `Future` status.
2. Confirm or omit the no-earlier-than launch date so it is not presented as a
   firm commitment.
3. Cite the existing public mission pages and use the approved public image.

Before submitting the nominal transmitter suggestion after the satellite
record is accepted:

1. Approve a public DB-facing citation for 401.200 MHz, `GFSK`, and the nominal
   9600-baud beacon configuration.
2. Recheck the current form choices for mode and service.
3. Submit it as `Inactive` and `Unconfirmed`, with the initial drift frequency
   equal to the center frequency until an observed correction exists.

Frequency deviation, pulse shaping, whitening, FEC, interleaving, a complete
RF frame, the APID 1 packet-length discrepancy, and the ground-side decoder
boundary are deferred receiver/decoder validation items. They do not block
either initial DB suggestion.

## SatNOGS form references

- [Satellite suggestion fields](https://wiki.satnogs.org/Satellite_Suggestions)
- [Transmitter suggestion fields](https://wiki.satnogs.org/Transmitter_Suggestions)
