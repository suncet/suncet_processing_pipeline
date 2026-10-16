# SunCET SatNOGS Onboarding and Operations Plan

Last updated: 2026-10-05

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
Both unusual `0x41` SSID octets are literal and intentional. The September 30
flight-unit IQ capture now confirms flags, bit stuffing, and captured FCS.
The released SatNOGS GFSK/AX.25 receiver now passes a local integration replay:
it retains the header and removes flags/FCS, delivering 268-byte beacon frames.

The [2026-09-23 Hydra capture audit](SATNOGS_UHF_VALIDATION_20260923.md) found
this literal header immediately before 544 checksum-valid beacons. Most saved
beacons are bare CCSDS; at 516 header-bearing boundaries the next header follows
the packet immediately, with no FCS bytes. This supports the software framing
description but does not establish the SatNOGS receiver's output boundary.
On 2026-09-29, the mission owner accepted these headers as valid evidence for
implementation. Address order/callsigns, literal SSID octets, control/PID, and
direct CCSDS encapsulation are therefore closed as header-layout questions.
The reason Hydra retained them need not be resolved before decoder work.

#### Radio configuration evidence

The [September 30 IQ audit](SATNOGS_IQ_VALIDATION_20260930.md) establishes the
9600-baud receive chain with G3RUH descrambling, NRZI, and HDLC, without FEC or
deinterleaving. Measured deviation is approximately ±2.4 kHz; Gaussian BT and
the complete burst preamble remain unconfirmed. The public transmitter table
still lists FEC as TBC because the capture does not establish every supported
configuration. Obtain the as-built TRX-U hardware/firmware revision and
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
  not establish that a packet has the supported field layout. The
  [2026-09-23 recorded Hydra comparison](SATNOGS_UHF_VALIDATION_20260923.md)
  passes for all 111 public fields across 7,352 distinct checksum-valid
  beacons. Thirty-four checksum-valid version packets report FSW 2.0.5.
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
  evidence only and are not accepted decoder contracts. The current decoder's
  local recorded-packet comparison and released-receiver IQ replay pass;
  local upstream-decoder/DB-function integration also passes. Upstream
  acceptance/deployment and hosted ingestion remain before operational acceptance.
- The FCC authorization and technical submission resolve the center frequency,
  19.2 kHz emission bandwidth, GFSK modulation, RHCP polarization, no-FEC filing
  configuration, 2 W transmitter output, 1.53 W authorized ERP, experimental
  service, and license identifiers. The September 30 flight-unit recording now
  verifies the 9600-baud descrambling/line-coding/framing chain and supplies a
  deviation estimate. Gaussian BT and configuration readback remain open.
- The FCC filing describes 19200 bit/s, while the current mission operations
  plan uses 9600 baud nominally and retains 19200 baud as a contingency. The
  flight-unit 9600-baud offline test and released SatNOGS receiver replay pass;
  19200 baud and live Network reception still need validation.
- Flight software has confirmed the literal 16-byte AX.25 header assembled as
  destination, source, control, and PID; the direct CCSDS payload; and the FCS
  calculation and append operation. The `0x41` octets are not modified later.
  The local SatNOGS replay establishes that its GFSK/AX.25 receiver removes
  flags and FCS while preserving the complete literal header.
- The private APID 1 definition includes a command-opcode name map. Do not copy
  that map into the public decoder. The mission has also excluded command
  counters/status, arm states, and other uplink-related beacon values from the
  public decoder.
- The FSW 2.0.4 prerelease user's guide confirms the mission time epoch,
  Fletcher-32 coverage, AX.25-plus-CCSDS layering, and the 256-byte threshold for
  segmentation. Separate flight-source confirmation resolves the literal AX.25
  address octets and FCS. CTDB 2.0.5 is the mission-approved field definition;
  the recorded Hydra packet comparison passes. The successfully validated
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

### 3. Add the UHF transmitter record — submitted, awaiting review

- Submitted [suggestion 13013](https://db.satnogs.org/transmitter-suggestions/13013)
  as `jmason86` on 2026-09-29 at 16:51 UTC. SatNOGS confirmed successful
  storage and pending moderator review. Transmitter UUID:
  `8m9cMNe8eaHd9WAqXucTVa`. Review-email notifications are enabled.
- The live SatNOGS API vocabulary was checked on 2026-09-29. Use `Space
  Operation` for the APID 1 health-beacon submission; `Experimental` is not a
  supported choice. This is the proposed SatNOGS transmission category, separate
  from the FCC authorization. The [submission draft](SATNOGS_DB_SUBMISSION_DRAFT.md)
  records the source and rationale.
- The [submission record](SATNOGS_DB_SUBMISSION_DRAFT.md) contains the saved
  nominal transmitter metadata. The reviewed citation is published at the canonical
  [public beacon specification URL](https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md),
  and was verified before submission. The saved entry uses 401.200 MHz, GFSK,
  9600 baud, Space Operation, inactive/unconfirmed status, and zero initial
  drift correction. No uplink parameters were submitted.
- Respond to moderator feedback; submission does not establish acceptance.
  Retain `Unconfirmed` until the relevant parameters are validated.

**Gate:** The accepted inactive/unconfirmed record contains the project-approved
DB-facing values and public citations. Complete receiver configuration is a
separate phase-4 gate.

### 4. Validate reception with flight-representative data — nominal local receiver replay passes

- The [September 30 flight-unit IQ audit](SATNOGS_IQ_VALIDATION_20260930.md)
  recovers ten distinct beacons, each with valid CRC-16/X-25 and Fletcher-32,
  plus ten other CRC-valid frames excluded from public analysis. The mission
  owner confirmed actual flight hardware and FSW 2.0.5 on October 5.
- The receive-only reconstruction establishes 9600 baud, G3RUH scrambling,
  NRZI, HDLC flags/bit stuffing, literal AX.25 header, direct CCSDS payload,
  and the full-FCS adapter. Deviation is approximately ±2.4 kHz; Gaussian BT,
  complete burst preamble, and 19200-baud operation remain open.
- Confirm the exact center frequency, modulation, rate, coding, framing, and
  beacon cadence against the public specification.
- The [released-receiver replay](../benchmarks/satnogs_receiver_replay/README.md)
  passes with `satnogs-flowgraphs` 2.7 and `gr-satnogs` v3.1.0.1. The GFSK mode
  selects `satnogs_fsk_ax25.py`; its standard settings recover all 20 CRC-valid
  frames, including all ten beacons. No custom flowgraph or tuning is needed
  for this recording. Disabling descrambling produces zero valid frames.
- Captured AX.25 FCS passes independent verification. Both the reconstructed
  GNU Radio HDLC path and the upstream receiver retain the header and remove
  flags/FCS, giving 268 bytes to
  `decode_public_ax25_beacon(..., fcs_present=False)`. PMT serialization and the
  subscriber's JSON/base64 representation preserve the bytes.
- All 20 frames match the separate receive reconstruction and independent
  HDLC recovery byte-for-byte. No paired independent recording from this
  session was supplied; this is agreement between decoders of the same IQ.
- Live Network reception and the 19200-baud contingency mode remain untested.
  The local replay does not establish a deployed station's software version.

Retain the validation artifacts with the test results: a flight-equivalent raw
AX.25 frame, extracted APID 1 bytes, independently verified raw and engineering
values, the expected checksum calculation, receiver settings, and a short
representative IQ/audio recording where sharing rights and size permit. These
are acceptance evidence for reception/decoding, not initial transmitter-form
attachments. Keep action tracking here rather than in the public specification.
Keep this mixed IQ recording and all raw packets private. Even an APID 1-only
crop retains excluded command/status bytes; it is not a sanitized public
artifact. The aggregate audit and existing synthetic fixture can support
upstream preparation without disclosing captured bytes.

**Gate:** Passed locally for nominal 9600 baud: a repeatable laboratory replay
produces intact APID 1 packets through the released SatNOGS receive chain.

### 5. Implement the APID 1 Kaitai decoder — submitted, awaiting upstream review

- The generated [`suncet_apid1.ksy`](../suncet_processing_pipeline/satnogs/suncet_apid1.ksy)
  now targets only CTDB 2.0.5, exposing its 111 approved public fields and
  engineering conversions while consuming excluded regions opaquely. It
  enforces the exact 252-byte packet size, APID 1 primary word, declared length,
  end-of-input, and fine-time range. There is no legacy compatibility path.
- The Kaitai core starts at the bare CCSDS packet. A tested local
  [`AX.25 adapter`](../suncet_processing_pipeline/satnogs/ax25.py) now validates
  the accepted literal header and extracts the beacon. Its caller explicitly
  selects 268-byte header-plus-packet input or 270-byte input with an FCS;
  the latter additionally validates CRC-16/X-25. The released SatNOGS
  GFSK/AX.25 receiver replay establishes the 268-byte input boundary. Use
  [`decoder.py`](../suncet_processing_pipeline/satnogs/decoder.py) as the
  validated entrypoint: it applies the Python Fletcher-32/envelope contract
  before invoking Kaitai. The KSY alone only consumes checksum bytes.
- The adapter passes the recorded 544 valid header-bearing beacons with
  unchanged public values and rejects the 278 checksum-invalid candidates.
  All 126 local SatNOGS tests pass in the `suncet` environment, including
  independent FCS vectors, corrupted header/payload/checksum bytes, and wrong
  input boundaries. The full-FCS path additionally passes all ten beacons
  recovered from the September 30 flight-unit IQ capture. The expanded local
  SatNOGS suite passes 144 tests as of October 5.
- The official Kaitai compiler 0.11 and Python runtime 0.11 now compile and
  execute the generated parser successfully locally. Passing local checks
  compare all 111 synthetic public-field values and independent engineering
  expectations, and exercise malformed packets, checksum corruption, and
  boundary values. CI is configured to rebuild with the SHA-256-pinned official
  compiler ZIP, compare generated source, run the parser tests, and decode a
  fixture from the installed wheel. The published core passed remote CI for
  commit `78049a4`; the new AX.25 adapter has local validation only so far.
  Reproduce the source check with
  `python -m suncet_processing_pipeline.satnogs.build_decoder --check --compiler PATH`.
- The committed fixture is deliberately synthetic. Local validation against
  the supplied 2026-09-23 Hydra capture now passes: 14,270 valid occurrences,
  7,352 distinct packets, and zero mismatches across all 111 approved fields
  against the generated private CTDB decoder. The raw captures remain private;
  the [aggregate audit](SATNOGS_UHF_VALIDATION_20260923.md) records the evidence.
  The September 30 IQ sample passes both offline receiver reconstruction and
  the released SatNOGS receiver replay. Live Network ingestion remains pending.
- The [upstream contribution bundle](../suncet_processing_pipeline/satnogs/upstream/README.md)
  now supplies a pure Kaitai 0.10 wrapper for the verified 268-byte receiver
  input. It enforces Fletcher-32 and the full CCSDS envelope on construction,
  including through the upstream DB entrypoint. Anonymous excluded regions
  and checksum buffers are hidden from the raw-structure API.
- [Integration checks completed 2026-10-05](SATNOGS_DECODER_INTEGRATION_20261005.md):
  all 164 schemas compile to Python/JavaScript; 293 upstream Python tests,
  JavaScript checks, five static checks, and installed-wheel decoding pass.
  The actual DB ingestion functions, with in-memory I/O substitutes, accept
  ten flight beacons with all 111 public values matching and reject ten other
  frames. Upstream state values use stable integer codes supported by Grafana;
  nonfinite sensor readings are omitted without losing the rest of the point.
  Hosted ingestion and deployment remain unverified.
- Mission-maintainer review and DCO sign-off were approved on 2026-10-06 under
  the current upstream
  [contribution guide](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/blob/master/CONTRIBUTING.md).
  The six-file contribution is submitted as
  [merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542),
  targeting upstream `master`. Published commit `f31782b6` includes the approved
  sign-off and exactly matches the reviewed local contribution tree.
  Follow-up commit `6248b1c6` corrects one test's class reference for pylint;
  [remote CI #2918408670](https://gitlab.com/jmason86/satnogs-decoders/-/pipelines/2918408670)
  passes in full. The MR awaits one eligible upstream approval.
- Parse the required link and CCSDS framing, then accept only APID 1 for public
  telemetry extraction.
- Define meaningful field names, types, enumerations, units, scale factors, and
  documentation references for every exposed beacon field.
- Do not include unrelated APID definitions or private CTDB material.
- Respond to upstream review of merge request !542. After release/deployment,
  request DB association of decoder `suncet` with the SunCET satellite entry.

**Gate:** Upstream tests pass and the accepted decoder converts an observed
SunCET APID 1 frame into correct engineering values while rejecting or ignoring
non-beacon packets safely.

### 6. Build the SunCET dashboard — Scratchpad draft created, ingestion pending

- The [dashboard plan](SATNOGS_DASHBOARD_PLAN.md) maps approved public aliases to
  identity/time, power, thermal, ADCS, payload/radio, and fault-status panels.
  It defines units/enums, no-data and stale-data behavior, raw spacecraft time
  versus reception UTC, and the remaining acceptance checks.
- Editor access and membership in the SunCET team were confirmed by the mission
  owner on 2026-09-29. Scratchpad creation/editing was verified.
- [suncet-dev](https://dashboard.satnogs.org/d/ffzqpb9arppfkb/suncet-dev) now exists
  in Scratchpad with 38 telemetry panels covering 82 approved public fields.
  Its section layout follows the read-only AEPEX reference; no other dashboard
  was edited. The generator and saved export are in
  `suncet_processing_pipeline/satnogs/dashboard/`.
- The production datasource is `SatNOGS DB` (`000000001`). A field-key query for
  provisional measurement `98244` returned no frames. All telemetry queries
  remain disabled, and reception timestamp/age remains explicitly unavailable.
- Group panels into spacecraft identity/time, power, thermal, ADCS, radio, and
  fault/status categories as supported by the actual beacon definition.
- Show units, enumerated states, last-observation time, and data gaps clearly.
- Avoid implying that beacon sampling is continuous or simultaneous when it is
  not.
- Local DB-function validation confirms all 82 saved query aliases and the
  Canvas numeric state mappings. The code writes flat engineering fields,
  uses the satellite NORAD number as measurement, and formats reception
  timestamps to whole UTC seconds. Live field presence and Grafana query/
  display behavior still need verification after decoder deployment.
- Validate hosted ingestion and displayed recorded observations, including
  provenance/time semantics, before calling the dashboard operational.

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

The public specification is published and transmitter suggestion 13013 awaits
moderator review. Respond to review feedback. The local CTDB 2.0.5
recorded-data comparison, released receiver replay, and local upstream decoder/
DB-function integration now pass. Mission-maintainer review and DCO sign-off
are complete; [decoder merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542)
is open. Respond to upstream review. After acceptance and deployment,
request the SunCET DB decoder association. No new nominal-mode IQ sample is
needed for those steps, and recorded raw frames must remain private.
The Scratchpad `suncet-dev` dashboard's 82 field aliases and numeric state
mappings pass the local contract check. Verify actual hosted fields, reception
timestamps, and displayed values before enabling queries or promoting it to
public Telemetry. The mission
owner confirmed the public launch wording on 2026-09-29: no earlier than
2027-03-15, manifested on a SpaceX Falcon 9 launch. No new CTDB/FSW definition
or RF sample is needed to submit the transmitter record.

## Current mission answers pending

No further mission-team answer currently blocks the initial satellite or
transmitter DB suggestions. The following input remains necessary for receiver
and decoder validation:

1. **Recorded packet comparison — complete locally:** all 111 approved fields
   agree with the separate generated CTDB 2.0.5 decoder on the supplied Hydra
   capture. No additional packet sample is needed for this local comparison.
   The local upstream decoder/DB-function check now passes on the flight IQ;
   hosted ingestion and dashboard display still require an end-to-end check.
2. **Flight-unit RF/IQ sample — received; nominal offline test passes:** the
   September 30 sample resolves the recorded 9600-baud line coding,
   descrambling, HDLC, and FCS questions, and supports approximately ±2.4 kHz
   deviation. Remaining inputs are Gaussian BT/configuration readback, a
   19200-baud sample if offered, and a paired independent packet record if
   available. Released SatNOGS receiver replay and the local upstream-decoder
   integration pass; live Network reception remains unverified.
3. **UTC conversion policy:** fine time is confirmed as integer milliseconds,
   but the epoch/time-scale and leap-second policy for authoritative UTC display
   remains unresolved. Preserve raw epoch seconds and reception time separately
   while this is settled; it does not block the transmitter record. The Hydra
   capture also includes 4,043 valid beacon occurrences with coarse time below
   one day, apparently before clock setting. Do not treat those values as
   actual observation times.

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
