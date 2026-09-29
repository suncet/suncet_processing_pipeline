# SunCET SatNOGS DB Submission Draft

Last updated: 2026-09-29

## Purpose

This is the offline copy deck for the SunCET spacecraft and nominal UHF
transmitter suggestions. It records the submitted spacecraft and transmitter
values, but SatNOGS DB remains authoritative for review
status. Recheck the live SatNOGS form vocabulary before any later entry. The
satellite suggestion may cite existing public mission pages; it does not depend
on completion of the receiver or telemetry-decoder specification.

## Spacecraft suggestion — accepted

Submitted by `jmason86` on 2026-09-01 as
[suggestion 11880](https://db.satnogs.org/satellite-reviewed-suggestions/11880), with
SatNOGS identifier
[`MNRC-9829-4319-5529-8975`](https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975).
The DB history records approval by `fredy` on 2026-09-01 at 21:20. The live
record was verified on 2026-09-25 with `Future` status and temporary NORAD ID
`98244`; this is not an official on-orbit catalog assignment. The table below
preserves the originally submitted values.

| SatNOGS field | Submitted entry |
| --- | --- |
| NORAD ID | Blank before identification |
| Followed NORAD ID | Blank until there is a justified launch-object candidate |
| Name | `SunCET` |
| Other names | `Sun Coronal Ejection Tracker` |
| Description | SunCET is a 6U NASA Heliophysics CubeSat jointly developed by the Johns Hopkins Applied Physics Laboratory and the University of Colorado Boulder's Laboratory for Atmospheric and Space Physics. Its extreme-ultraviolet imager is designed to observe how coronal mass ejections accelerate from the low solar corona through the middle corona. It also contains a soft X-ray quad-diode photometer for characterizing solar flares. |
| Owner/operator | Blank because the live vocabulary had no matching JHU/APL or LASP/CU Boulder choice; both institutions remain in the description |
| Status | `Future` |
| Countries of origin | `United States of America` |
| Website | <https://suncet.jhuapl.edu/> |
| Dashboard URL | Blank until the SunCET SatNOGS dashboard exists |
| Launch date | `2027-03-15 00:00` as submitted; revise if mission planning changes |
| Deploy date | Blank until deployment timing is manifested |
| Re-entry date | Blank |
| Image | [`assets/suncet_spacecraft.jpg`](assets/suncet_spacecraft.jpg), the resized metadata-free public copy |
| Citation | <https://www.jhuapl.edu/destinations/missions/suncet> |
| Email when reviewed | Yes |

Stable image URL after this file is merged to `main`:
<https://raw.githubusercontent.com/suncet/suncet_processing_pipeline/main/docs/assets/suncet_spacecraft.jpg>

## Nominal UHF transmitter suggestion — submitted, awaiting review

Submitted by `jmason86` on 2026-09-29 at 16:51 UTC as
[suggestion 13013](https://db.satnogs.org/transmitter-suggestions/13013),
transmitter UUID `8m9cMNe8eaHd9WAqXucTVa`. SatNOGS confirmed successful storage
and pending moderator review. The saved details match the table below;
review-email notifications are enabled. Submission is complete; acceptance is
not yet recorded.

| SatNOGS field | Submitted entry |
| --- | --- |
| Description | SunCET nominal global health beacon: 401.200 MHz GFSK at 9600 baud, carrying an AX.25 UI frame whose information field contains a CCSDS APID 1 packet. Beacon cadence is spacecraft-mode dependent and is typically of order 10 seconds. |
| Type | `Transmitter`—the public entry intentionally describes only the downlink beacon and publishes no uplink parameters |
| Status | `Inactive` before verified on-orbit reception |
| Downlink frequency | `401200000` Hz |
| Downlink drift frequency | `401200000` Hz initially, representing zero observed correction; update from measured on-orbit drift rather than using licensed tolerance as drift |
| Downlink mode | `GFSK`; this is present in the current SatNOGS vocabulary, while the record remains `Unconfirmed` pending RF validation |
| Baud | `9600` |
| Service | `Space Operation` for spacecraft health telemetry; this SatNOGS category is separate from the FCC Experimental Radio Service authorization |
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

The public citation to paste into the form is:
<https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md#uhf-transmitter>

The reviewed specification revision was published at that URL before submission. Citation
means a public source for the entered radio parameters, not an attached IQ file.
The [official transmitter form guide](https://wiki.satnogs.org/Transmitter_Suggestions#Form_fields)
lists metadata and citation fields; it does not require an IQ upload. The
requested IQ recording supports the separate RF receiver test. The requested
recent UHF packet capture supports the decoder's real-data comparison.

The live [SatNOGS API schema](https://db.satnogs.org/api/schema/?format=json)
was checked on 2026-09-29: both `Space Operation` and `Space Research` are
supported, while `Experimental` is not. `Space Operation` is the proposed
classification for this operational health-beacon transmission. As a relevant
precedent, [COSMO's accepted entry](https://db.satnogs.org/satellite/68460/)
uses `Space Operation` for its TRX-U at 401.200 MHz and 9600 baud. This does not
establish SunCET's legal service classification or imply that COSMO's modulation
and other radio settings apply to SunCET.

## Contingency 19200-baud mode

Do not combine two baud rates into one nominal transmitter entry. Start with
the planned 9600-baud beacon. If the 19200-baud contingency configuration is
validated and becomes operationally relevant, add a distinct transmitter
record with the same center frequency and mark only the configuration actually
being transmitted as active. This follows the SatNOGS model in which a change
that requires a different receiver configuration is represented separately.

## Submission gates

The satellite suggestion has passed both submission and acceptance gates.
The transmitter submission gate passed on 2026-09-29: the citation was
published, live form choices checked, and the inactive/unconfirmed entry
submitted with initial drift frequency equal to the center frequency.
Respond to moderator review before treating the transmitter as accepted.
Keep the accepted spacecraft record's launch date synchronized with mission planning.

Frequency deviation, pulse shaping, whitening, FEC, interleaving, a complete
RF frame, the planned APID 1 beacon revision, and the ground-side decoder
boundary are deferred receiver/decoder validation items. They do not block
either initial DB suggestion.

## SatNOGS form references

- [Satellite suggestion fields](https://wiki.satnogs.org/Satellite_Suggestions)
- [Transmitter suggestion fields](https://wiki.satnogs.org/Transmitter_Suggestions)
