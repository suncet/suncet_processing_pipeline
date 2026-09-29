# SunCET SatNOGS Onboarding and Operations Plan

Last updated: 2026-09-29

## Purpose

This document tracks the work required to establish SunCET in SatNOGS before
launch and operate its public beacon presence after launch. The pre-launch
SatNOGS DB record is accepted with `Future` status. SatNOGS assigns temporary
NORAD ID `98244`; the official on-orbit catalog assignment remains pending.

SunCET will expose only the globally broadcast beacon, CCSDS APID 1, through
SatNOGS. Other spacecraft APIDs may travel through the mission's private ground
system, but they are outside the SatNOGS decoder, dashboard, documentation, and
validation scope.

## System boundaries

SatNOGS consists of several connected services:

- **SatNOGS DB** holds the public spacecraft and transmitter records and the
  decoded telemetry associated with observations.
- **SatNOGS Network** schedules participating ground stations and stores
  observations and received artifacts.
- **SatNOGS Decoders** converts a received SunCET beacon frame into named APID 1
  telemetry values using a Kaitai Struct definition.
- **SatNOGS Dashboard** plots the decoded public beacon telemetry.

This work does not make the Jetson a public server or require inbound access to
it. SatNOGS observations are obtained from the SatNOGS services over outbound
connections. Direct SunCET ground-station ingest remains a separate SOC input.

## Decisions

- Keep the accepted pre-launch SunCET record at `Future`; its temporary SatNOGS
  NORAD value is not an official on-orbit identification.
- Publish only the information required to receive and interpret the public
  beacon. Do not publish private CTDB content, restricted radio documentation,
  credentials, commanding information, or non-beacon packet definitions.
- Decode only complete CCSDS APID 1 beacon packets. Frames containing other
  APIDs are ignored by the SunCET SatNOGS telemetry decoder.
- Use the mission-approved CTDB 2.0.5 definition exclusively. The current public
  decoder supports its 252-byte packet layout and 111 approved public fields;
  legacy CTDB 2.0.1 layouts and automatic layout inference are outside scope.
- Use mission-authored public documentation as the primary citation for the
  satellite, transmitter, and decoder submissions. A private vendor ICD may be
  used to verify facts but is not itself a public citation.
- Reuse an existing `gr-satnogs` demodulator if it correctly receives the
  SunCET waveform. Develop a new flowgraph only if laboratory recordings show
  that an existing mode is insufficient.
- Treat the official on-orbit NORAD assignment and transmitter activation as
  post-launch updates, not pre-launch blockers.
- Treat the initial SatNOGS DB records, receiver validation, and decoder
  submission as separate gates. Detailed waveform parameters and RF captures
  may remain pending while the pre-launch satellite and inactive/unconfirmed
  transmitter records are submitted with cited DB-facing values.
- Keep credentials and SatNOGS account recovery material outside this public
  repository.

## Inputs by gate

SatNOGS DB, receiver validation, and decoder integration need different levels
of detail. Values must come from mission documentation or the configured flight
radio rather than assumptions based on a generic radio model, but not every
receiver detail is required before creating the initial DB records.

### Initial SatNOGS DB records

- The complete proposed pre-launch satellite record includes the official name
  and aliases, concise public description, owners/operators and country,
  website, public image, `Future` status, and public citations. SatNOGS accepts
  a much smaller minimum, and unknown or inapplicable NORAD, owner, deployment,
  and re-entry values may remain blank.
- The complete proposed nominal transmitter record includes the center and
  placeholder initial drift frequencies, `GFSK` mode, 9600 baud, service
  choice, `Inactive` status, `Unconfirmed` flag, short description, and a
  public citation for those values. These are the project-approved submission
  contents, not a claim that every field is mandatory in the live form.
- The launch date may be omitted if the live form cannot represent the current
  no-earlier-than date without implying a firm launch commitment.
- The transmitter record is created only after the satellite suggestion is
  accepted. Its initial drift frequency may equal the published center
  frequency until observations establish a measured correction.

Frequency deviation, occupied bandwidth, pulse shaping, whitening, FEC,
interleaving, detailed AX.25 framing, recorded APID 1 packets, and an RF/IQ
recording are receiver/decoder validation work. They do not block the
inactive/unconfirmed transmitter suggestion.

### Receiver and decoder validation

- Confirm the actual frequency deviation, occupied bandwidth, pulse shaping,
  and polarization where relevant.
- Forward-error correction, interleaving, whitening, and scrambling
- AX.25 address order, callsigns, SSIDs, control/PID values, and FCS behavior
- Encapsulation between AX.25 and the CCSDS APID 1 packet
- Beacon cadence and any mode-dependent changes to cadence or waveform
- Spectrum service and coordination status, including public IARU or ITU
  references where applicable
- One or more representative raw frames and, preferably, an RF recording from
  the flight-equivalent transmitter

Flight software has confirmed its complete radio-interface frame buffer:
literal destination and source address octets representing `LASP-0` and
`SUN1-0`, control `0x03`, PID `0xF0`, the CCSDS packet, and a CRC-16/X-25 FCS.
Both unusual `0x41` SSID octets are literal and intentional. An RF capture must
still establish flags, bit stuffing, and the exact boundary delivered by the
SatNOGS receive path.

#### Radio configuration evidence

The public beacon specification omits unresolved deviation, filter,
interleaving, and whitening rows; these remain receiver-validation tasks here.
It lists FEC as TBC. Obtain the as-built TRX-U hardware/firmware revision and
active downlink configuration for both 9600 and 19200 modes from the radio/FSW
team or vendor. The public
[TRX-U datasheet](https://www.aac-clyde.space/wp-content/uploads/2021/11/TRX-U-datasheet.pdf)
does not establish the settings programmed into SunCET's radio.

| Parameter | Meaning and requested evidence |
| --- | --- |
| Frequency deviation | Frequency excursion above/below the carrier for the two FSK symbols. Obtain the configured deviation in ±Hz for each rate and verify it against IQ after removing carrier offset. |
| Pulse shaping/filter | GFSK smooths symbol transitions with a Gaussian filter. Obtain its BT value (filter bandwidth × symbol duration) and filter implementation/settings. Known transmitted bits and IQ can help check the transition shape; a short arbitrary recording may not uniquely identify BT. |
| Interleaving | Reorders bits or symbols to distribute burst errors. Obtain enabled/disabled state, permutation/depth, reset boundary, and its position relative to FEC and framing. |
| Whitening/scrambling | Reversibly changes repetitive bit patterns; it is not encryption. Obtain enabled/disabled state, algorithm/polynomial, initial state/reset rule, bit order, and which parts of the frame it covers. |
| FEC and processing order | Confirm actual enabled coding and its parameters, plus the order of coding, interleaving, scrambling, NRZI, bit stuffing, framing, and FCS processing. The filing's no-FEC statement is not a readback of flight configuration. |

Neither the nominal baud rate nor the licensed 19.2 kHz bandwidth uniquely
determines these parameters. A spectrum alone cannot establish interleaving or
whitening. Use the configuration and a paired exact transmitted packet to
test candidate receive processing, requiring repeated valid AX.25 FCS and
CCSDS Fletcher-32 results. Publish the confirmed receiver settings when available.

### APID 1 beacon definition for decoder integration

- Complete APID 1 byte layout after the CCSDS primary header
- Packet version and length rules
- Byte order, bit order, signedness, and field widths
- Engineering-unit conversions, scale factors, offsets, and units
- Enumerations, status bits, invalid/fill values, and reserved fields
- Time representation and its conversion to UTC, if a beacon timestamp exists
- Checksums or CRCs above the AX.25 FCS, if present
- A raw packet paired with independently verified expected field values

The public APID 1 definition should be a deliberately reviewed export. It must
not be generated by publishing the complete private CTDB. These details are
needed for the telemetry decoder, not for the initial satellite or transmitter
DB suggestions.

## Roadmap and status

### 0. Establish accounts and ownership — in progress

- The primary maintainer already has the SatNOGS username `jmason86` and a
  Raspberry Pi SatNOGS ground station. The personal account email is
  deliberately not stored in this public repository. Integration or operation
  of that particular station is outside the current SunCET onboarding scope.
- Confirm that the existing account can sign in to SatNOGS DB, Network,
  Dashboard, the Libre Space Community forum, and GitLab.
- Sign in to SatNOGS DB, Network, Dashboard, the Libre Space Community forum,
  and GitLab using the intended mission-maintainer identity.
- Record at least one backup SunCET maintainer so the mission entry and decoder
  are not dependent on a single personal account.
- Revisit mission-operated ground-station integration only if it later provides
  a clear commissioning or operations benefit. It is not a prerequisite for
  the satellite, transmitter, decoder, or dashboard work.

### 1. Produce the public communications specification — in progress

- A public working draft now exists as the
  [SunCET public beacon specification](SUNCET_PUBLIC_BEACON_SPEC.md). It contains
  cited public mission facts, confirmed software-side framing values, and
  explicit RF-validation items.
- The canonical public location is the specification's rendered page in the
  mission's public GitHub repository:
  <https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md>.
  Publication at this stable URL does not make a draft revision authoritative;
  its status and revision remain explicit inside the document.
- A dependency-free mission-side APID 1 beacon contract now validates the
  stable CCSDS envelope and known Fletcher-32 algorithm, combines coarse
  seconds with the validated 0-999 millisecond fine field, rejects non-beacon
  APIDs, and accepts only the exact 252-byte CTDB 2.0.5 envelope. The validated
  decoder entrypoint applies this checksum/envelope contract before calling the
  generated Kaitai field parser. The KSY alone does not validate Fletcher-32.
- A local field-review exporter creates an offset-preserving APID 1 worksheet
  from the private CTDB. Every ordinary field begins as `REVIEW`, likely
  command/uplink-related fields begin as `OMIT`, and no field is automatically
  approved or copied into a public schema. Private review outputs are ignored
  by Git.
- Mission-owner approval selects CTDB 2.0.5 as the authoritative definition for
  this work. The public interface now contains 111 approved fields. The old
  `csie_meta_nand_sci_write_ptr` field is absent from that export and has been
  removed; command/uplink and other excluded fields remain opaque. No legacy
  decoder compatibility or additional definition confirmation is required.
- The reviewed public field table is stored in the repository as
  [`public_beacon_schema.csv`](../suncet_processing_pipeline/satnogs/public_beacon_schema.csv).
  It preserves authoritative bit offsets while containing only approved public
  names, descriptions, units, conversions, and status maps.
- The public schema has been reconciled to CTDB 2.0.5 offsets, types, and
  engineering definitions. Packet provenance must identify this version:
  historical layouts can also be 252 bytes, so length and checksum alone do
  not establish that a packet has the supported field layout. The UHF packets
  requested from the mission team, recorded the previous week, remain the next
  recorded-data comparison; they are validation evidence, not a request for
  another CTDB definition.
- The 16-bit fine-time encoding has been resolved empirically as integer
  milliseconds after the coarse second. Production timestamp conversion, the
  public schema, and the decoder now use `coarse + fine / 1000` and constrain
  the wire value to 0-999.
- CTDB 2.0.5 defines the current engineering units and state maps. Dual-SPS
  flare magnitude is an unsigned 8-bit raw value with no engineering
  conversion; its threshold remains log10 estimated XRS-B flux. Phase labels
  include `FLARE_START`, `DECLINING_FLARE`, and `RISING_FLARE`. The current
  schema also preserves the configurable CSIE histogram formula, default offset
  0 and width 32, six default beacon ranges, and truncation after bin 5.
- Publish a reviewed DB-facing revision that clearly distinguishes confirmed
  values from receiver/decoder items still marked `TBC`.
- Include a revision identifier and effective date.
- Add enough information for an independent observer to demodulate a frame and
  interpret every exposed beacon field as receiver validation is completed.
- Review the document for export, licensing, security, and spectrum-coordination
  concerns before publication.
- Publish reviewed revisions at the canonical GitHub URL suitable for SatNOGS
  citations.

Current receiver/decoder findings to track after the DB-facing review; these do
not block the initial SatNOGS DB suggestions:

- CTDB 2.0.5 defines the supported 252-byte layout. The earlier 251-byte CTDB
  2.0.1 export and its compiler-aligned 252-byte counterpart are historical
  evidence only and are not accepted decoder contracts. Compare the current
  decoder with the requested UHF packet sample before operational acceptance.
- The FCC authorization and technical submission resolve the center frequency,
  19.2 kHz emission bandwidth, GFSK modulation, RHCP polarization, no-FEC filing
  configuration, 2 W transmitter output, 1.53 W authorized ERP, experimental
  service, and license identifiers. Frequency deviation, filtering, whitening,
  interleaving, and exact programmed flight settings still need confirmation or
  RF measurement.
- The FCC filing describes 19200 bit/s, while the current mission operations
  plan uses 9600 baud nominally and retains 19200 baud as a contingency. Both
  modes need flight-equivalent receiver validation.
- Flight software has confirmed the literal 16-byte AX.25 header assembled as
  destination, source, control, and PID; the direct CCSDS payload; and the FCS
  calculation and append operation. The `0x41` octets are not modified later.
  Ground-side removal of flags or FCS remains a separate receiver-path question
  for SatNOGS laboratory validation.
- The private APID 1 definition includes a command-opcode name map. Do not copy
  that map into the public decoder. The mission has also excluded command
  counters/status, arm states, and other uplink-related beacon values from the
  public decoder.
- The FSW 2.0.4 prerelease user's guide confirms the mission time epoch,
  Fletcher-32 coverage, AX.25-plus-CCSDS layering, and the 256-byte threshold for
  segmentation. Separate flight-source confirmation resolves the literal AX.25
  address octets and FCS. CTDB 2.0.5 is the mission-approved field definition;
  recorded-packet comparison remains pending. The successfully validated
  pipeline Fletcher-32 implementation is the working authority for its word
  order, seed, and stored byte order. Neither legacy compiler padding nor a new
  FSW definition is a current blocker.

**Gate:** The satellite suggestion may use the public mission pages directly.
The transmitter suggestion requires a reviewed public citation for its
DB-facing frequency, mode, nominal baud rate, status, and service description.
Unresolved receiver and decoder parameters may remain explicitly `TBC`.

### 2. Create the pre-launch SatNOGS DB record — complete

- An offline [SatNOGS DB submission draft](SATNOGS_DB_SUBMISSION_DRAFT.md) now
  contains proposed values for every spacecraft field and the nominal 9600-baud
  transmitter. It also records the submission gates and the policy for a
  separate 19200-baud contingency record.
- A resized, metadata-free public spacecraft image is stored at
  [`assets/suncet_spacecraft.jpg`](assets/suncet_spacecraft.jpg) so the accepted
  record does not depend on a private local file or leak phone/GPS metadata.
- The spacecraft suggestion was submitted by `jmason86` on 2026-09-01 as
  [suggestion 11880](https://db.satnogs.org/satellite-reviewed-suggestions/11880)
  and approved by `fredy` on 2026-09-01 at 21:20 as displayed in the DB history.
- The live record retains SatNOGS identifier
  [`MNRC-9829-4319-5529-8975`](https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975).
  Its acceptance and `Future` status were verified on 2026-09-25. The displayed
  temporary NORAD ID `98244` is not an official on-orbit catalog assignment.
- The submitted record leaves NORAD and owner/operator blank, uses `Future`
  status, identifies the United States of America as the country of origin, and
  includes the public image, mission website, APL citation, and the submitted
  2027-03-15 launch-planning date.
- Keep the record synchronized with reviewed mission-planning changes.

**Gate passed:** The accepted spacecraft record is live; spacecraft review no
longer blocks the transmitter suggestion.

### 3. Add the UHF transmitter record — pending

- The live spacecraft page showed no approved transmitters on 2026-09-25.
- The live SatNOGS API vocabulary was checked on 2026-09-29. Use `Space
  Operation` for the APID 1 health-beacon submission; `Experimental` is not a
  supported choice. This is the proposed SatNOGS transmission category, separate
  from the FCC authorization. The [submission draft](SATNOGS_DB_SUBMISSION_DRAFT.md)
  records the source and rationale.
- The [submission draft](SATNOGS_DB_SUBMISSION_DRAFT.md) contains the nominal
  transmitter metadata. Publish the reviewed citation at the canonical
  [public beacon specification URL](https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md),
  then submit it against the accepted SunCET DB record. RF/IQ data and recorded
  UHF packet comparison are not prerequisites for this submission.
- Enter the cited DB-facing frequency, mode, nominal baud rate, placeholder
  drift, service, coordination references, and source citation.
- Mark pre-launch or unverified facts appropriately; do not mark the transmitter
  active merely because it is planned.
- Check that `GFSK` and the selected service match the current SatNOGS form
  vocabulary. Retain `Unconfirmed` until laboratory or on-orbit validation.

**Gate:** The accepted inactive/unconfirmed record contains the project-approved
DB-facing values and public citations. Complete receiver configuration is a
separate phase-4 gate.

### 4. Validate reception with flight-representative data — pending

- The requested flight-equivalent IQ recording is pending. Use it for
  demodulator and receiver-boundary validation; a decoded UHF packet file can
  validate the packet parser but does not replace this RF evidence.
- Confirm the exact center frequency, modulation, rate, coding, framing, and
  beacon cadence against the public specification.
- Run the recording through the most appropriate existing `gr-satnogs`
  flowgraph.
- Verify the captured AX.25 FCS against the confirmed algorithm and determine
  exactly what bytes SatNOGS passes to the Kaitai decoder.
- Compare the resulting CCSDS APID 1 packet byte-for-byte with an independently
  captured reference packet.
- If no existing receiver is adequate, document the mismatch and contribute
  the smallest required `gr-satnogs` flowgraph and client support.

Retain the validation artifacts with the test results: a flight-equivalent raw
AX.25 frame, extracted APID 1 bytes, independently verified raw and engineering
values, the expected checksum calculation, receiver settings, and a short
representative IQ/audio recording where sharing rights and size permit. These
are acceptance evidence for reception/decoding, not initial transmitter-form
attachments. Keep action tracking here rather than in the public specification.

**Gate:** A repeatable laboratory test produces an intact APID 1 packet using
the same reception path expected in the SatNOGS Network.

### 5. Implement the APID 1 Kaitai decoder — in progress

- The generated [`suncet_apid1.ksy`](../suncet_processing_pipeline/satnogs/suncet_apid1.ksy)
  now targets only CTDB 2.0.5, exposing its 111 approved public fields and
  engineering conversions while consuming excluded regions opaquely. It
  enforces the exact 252-byte packet size, APID 1 primary word, declared length,
  end-of-input, and fine-time range. There is no legacy compatibility path.
- The parser deliberately starts at the bare CCSDS packet until RF testing
  establishes whether the selected SatNOGS receive path retains the known
  AX.25 header or FCS. Use
  [`decoder.py`](../suncet_processing_pipeline/satnogs/decoder.py) as the
  validated entrypoint: it applies the Python Fletcher-32/envelope contract
  before invoking Kaitai. The KSY alone only consumes checksum bytes.
- The official Kaitai compiler 0.11 and Python runtime 0.11 now compile and
  execute the generated parser successfully locally. Passing local checks
  compare all 111 synthetic public-field values and independent engineering
  expectations, and exercise malformed packets, checksum corruption, and
  boundary values. CI is configured to rebuild with the SHA-256-pinned official
  compiler ZIP, compare generated source, run the parser tests, and decode a
  fixture from the installed wheel. This records local verification and CI
  configuration, not a completed remote CI run. Reproduce the source check with
  `python -m suncet_processing_pipeline.satnogs.build_decoder --check --compiler PATH`.
- The fixture is deliberately synthetic. The requested recent UHF telemetry
  remains the recorded-flight comparison gate. The separate
  pending IQ sample establishes the RF receiver path; neither sample blocks
  local implementation or preparation for upstream review.
- Use the current `satnogs-decoders` repository conventions and a comparable
  AX.25/CCSDS mission such as CIRBE as a structural reference.
- Parse the required link and CCSDS framing, then accept only APID 1 for public
  telemetry extraction.
- Define meaningful field names, types, enumerations, units, scale factors, and
  documentation references for every exposed beacon field.
- Do not include unrelated APID definitions or private CTDB material.
- Submit the decoder to the upstream `satnogs-decoders` project and respond to
  maintainer review once its actual input boundary and recorded-frame evidence
  are ready; preserve end-to-end checksum validation in that integration.

**Gate:** Upstream tests pass and the accepted decoder converts an observed
SunCET APID 1 frame into correct engineering values while rejecting or ignoring
non-beacon packets safely.

### 6. Build the SunCET dashboard — local plan ready, implementation pending

- The [dashboard plan](SATNOGS_DASHBOARD_PLAN.md) maps approved public aliases to
  identity/time, power, thermal, ADCS, payload/radio, and fault-status panels.
  It defines units/enums, no-data and stale-data behavior, raw spacecraft time
  versus reception UTC, and a confidential editor-access request template.
  It is a local design, not a live or recorded-telemetry-validated dashboard.
- Sign in to the SatNOGS Dashboard once through Libre Space Foundation SSO.
- Open the confidential `satnogs-ops` request containing the SunCET satellite,
  maintainer team, and account email to obtain editor access.
- Create a compact operations dashboard from APID 1 fields.
- Group panels into spacecraft identity/time, power, thermal, ADCS, radio, and
  fault/status categories as supported by the actual beacon definition.
- Show units, enumerated states, last-observation time, and data gaps clearly.
- Avoid implying that beacon sampling is continuous or simultaneous when it is
  not.
- Validate datasource field paths, decoder ingestion, and the time policy
  against recorded observations before calling the dashboard operational.

**Gate:** A maintainer can diagnose basic spacecraft health from a decoded
beacon without consulting raw bytes.

### 7. Prepare Network operations and community coordination — pending

- Introduce SunCET and its public documentation to the SatNOGS community.
- A few weeks before launch, create or participate in the appropriate launch
  thread and provide expected deployment orbit, timing, frequency, modulation,
  and identification information.
- Decide who monitors observations and reception reports during commissioning.
- Define how SatNOGS observation IDs, timestamps, raw frames, station metadata,
  and decoder versions enter the SunCET operations record.
- If SunCET operates its own SatNOGS station, register it, keep it online, and
  validate scheduling and uploads before launch.
- Track FCC authorization renewal or modification as a pre-launch operations
  gate. The current authorization expires 2027-10-01, before the end of an
  eight-month prime mission beginning at the current 2027-03-15 no-earlier-than
  launch date and potentially before launch if the schedule slips.

### 8. Complete the post-launch identity transition — blocked until launch

- Add the best justified post-deployment TLE candidate or followed catalog ID.
- Work through the launch thread to correlate receptions using deployment
  timing, Doppler, onboard position/time, or exclusion evidence.
- Do not claim an identification without recording its evidence.
- Replace the temporary/pre-launch catalog reference with the official NORAD ID
  when assigned.
- Set satellite status to `Alive` after positive identification.
- Mark the transmitter active only after verified reception.
- Confirm that network auto-scheduling, decoding, and dashboard ingestion work
  with the accepted orbital elements.

### 9. Establish routine SatNOGS operations — future

- Monitor decoder errors, unexpected packet versions, frequency drift, data
  gaps, and stale dashboard panels.
- Review SatNOGS observations around every relevant pass during commissioning.
- Preserve the observation ID and decoder/public-specification revision for any
  SatNOGS data used operationally or scientifically.
- Define and implement a checksum-verified, idempotent archival transfer for
  every SunCET SatNOGS observation package retrieved by the SOC. Confirm the
  source API and artifact boundary rather than assuming SatNOGS DB is the
  binary-data endpoint: DB primarily contains satellite/transmitter metadata,
  while Network contains observations. Record which raw frames, decoded APID 1
  telemetry, station/observation metadata, and available RF products are
  retained, then upload the reviewed package under the shared raw archive's
  dedicated `satnogs/` namespace. Preserve observation IDs and decoder/public
  specification revisions, verify content checksums, and never delete or alter
  the SatNOGS source.
- Update the public specification, decoder, DB record, and dashboard together
  when the beacon format or transmitter behavior changes.
- Record the explicit CTDB/decoder version with retained packets. Current
  implementation scope is CTDB 2.0.5 only; legacy decoding would require a
  separately authorized workstream.
- Periodically verify that both primary and backup maintainers retain access.

## Deliverables

| Deliverable | Completion evidence |
| --- | --- |
| Public SunCET communications specification | Stable public URL and reviewed revision |
| SatNOGS satellite record | Complete: [`MNRC-9829-4319-5529-8975`](https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975), accepted with status `Future`; verified 2026-09-25 |
| SatNOGS transmitter record | Accepted record with cited flight parameters |
| Receiver validation package | RF recording, raw APID 1 frame, and reproducible procedure |
| APID 1 decoder | Merged `satnogs-decoders` contribution and passing test vectors |
| SunCET dashboard | Public panels populated from decoded observations |
| Launch coordination record | Libre Space Community launch thread |
| On-orbit identity | Evidence-backed NORAD assignment and `Alive` status |
| Operations procedure | Named maintainers, monitoring cadence, and update process |

## Immediate next action

Publish the reviewed beacon specification at its canonical citation URL, then
submit the prepared nominal 9600-baud transmitter as inactive and unconfirmed
against the accepted spacecraft record. Compare the locally tested CTDB 2.0.5
decoder with the already-requested UHF packets when they arrive. Use the
separately requested IQ sample for demodulation and
receiver-boundary validation before upstream integration. Sign in to the
dashboard and request editor access using the prepared template. The mission
owner confirmed the public launch wording on 2026-09-29: no earlier than
2027-03-15, manifested on a SpaceX Falcon 9 launch. No new CTDB/FSW definition
or RF sample is needed to submit the transmitter record.

## Current mission answers pending

No further mission-team answer currently blocks the initial satellite or
transmitter DB suggestions. The following input remains necessary for receiver
and decoder validation:

1. **Recorded packet comparison:** the requested recent UHF telemetry is pending.
   Compare its CTDB 2.0.5 decoded values against independent
   expectations. The mission has approved the available CTDB 2.0.5 definitions;
   another export or definition confirmation is not a prerequisite.
2. **Flight-equivalent RF/IQ sample:** the requested recording and paired raw
   frame are pending. They establish frequency deviation, pulse shaping, line
   coding, whitening/scrambling, interleaving, both supported baud modes, and
   the exact over-air and decoder-input boundaries. This is separate from the
   packet-parser comparison and transmitter registration.
3. **UTC conversion policy:** fine time is confirmed as integer milliseconds,
   but the epoch/time-scale and leap-second policy for authoritative UTC display
   remains unresolved. Preserve raw epoch seconds and reception time separately
   while this is settled; it does not block the transmitter record.

No further answer is currently needed about the legacy APID 1 compiler
padding, the flight-software AX.25 buffer or FCS, modulation, polarization,
filed FEC, licensed bandwidth/power, spectrum service, license identity,
Fletcher-32 implementation, fine-time serialization, or the maintainer's
personal SatNOGS station. Those are now resolved or deliberately outside scope.
Public-field policy, ADCS units, Command Loss Timer semantics, Dual-SPS flare
definitions, and CSIE histogram definitions are covered by the approved CTDB
2.0.5 interface and its 111 public aliases. A backup maintainer and the eventual
NORAD ID remain later governance/on-orbit items rather than current blockers.

## References

- [SatNOGS Satellite Operator Guide](https://wiki.satnogs.org/Satellite_Operator_Guide)
- [SatNOGS Satellite Suggestions](https://wiki.satnogs.org/Satellite_Suggestions)
- [SatNOGS Transmitter Suggestions](https://wiki.satnogs.org/Transmitter_Suggestions)
- [Adding a new SatNOGS data decoder](https://wiki.satnogs.org/Adding_a_new_data_decoder)
- [SatNOGS Decoders](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders)
- [SatNOGS Dashboard](https://wiki.satnogs.org/Dashboard)
- [SatNOGS Network operation](https://wiki.satnogs.org/Operation)
- [Libre Space Community launch discussions](https://community.libre.space/c/satellites-observations/launches/26)
