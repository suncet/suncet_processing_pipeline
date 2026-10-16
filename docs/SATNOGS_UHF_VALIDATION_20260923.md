# Recorded Hydra beacon validation — 2026-09-23 CPT

Inspected 2026-09-29 in the `suncet` conda environment. The recorded-data
comparison for the local CTDB 2.0.5 public beacon decoder passes. These Hydra
files also retain some AX.25 headers, but do not establish the RF waveform or
the input boundary of the SatNOGS receiver.

## Inputs and provenance

The mission supplied seven `*/ccsds_*` files below
`test_data/2026-09-23_cpt/uhf_rundirs`. Inputs were read without modification.
File sizes, SHA-256 hashes, and aggregate results are retained in the
[audit report](assets/satnogs_uhf_validation_20260923.json). Captured packets,
private decoded fields, and complete CTDB definitions are not included here.

Thirty-four checksum-valid version packets in the same capture report FSW
2.0.5. The comparison uses the configured private bus CTDB 2.0.5 generated
decoder and the public Kaitai decoder. This supplies version evidence for the
capture; packet length alone would not distinguish historical layouts.

## Packet and field results

| Check | Result |
| --- | --- |
| Current APID 1 header/length candidates | 14,548 |
| Strictly valid 252-byte beacon occurrences | 14,270 |
| Byte-distinct valid beacons | 7,352 |
| Rejected candidates | 278; all fail Fletcher-32 |
| Approved public fields compared | 111 |
| Distinct-packet field comparisons against generated CTDB decoder | 816,072; zero mismatches |
| Field comparisons against regenerated, checksum-enforced CSV | 1,583,970; zero mismatches |
| Valid fine-time range | 27–969 milliseconds |

Of the distinct valid packets, 6,918 appear twice and 434 appear once. These
are recorded copies, not evidence of separate RF receptions or retransmissions.
The public decoder required no changes. Its public field selection continues
to omit commanding information.

The initial CSV, associated with processing run
`20260929T184155.160251Z_736e71f1`, used checksum bypass. It had 14,311 beacon
rows: 14,033 valid occurrences, 278 checksum-invalid candidates, and 237 valid
occurrences missing relative to the binary scan. A concurrent reprocessing
run regenerated these products with checksum enforcement during this audit.
The regenerated CSV contains exactly all 14,270 valid occurrences and no
invalid or excess occurrences. An independent in-memory strict packetizer run
also recovers exactly that set. Do not use the initial bypass CSV as the
recorded-data reference.

## What framing survived

The first four binary files consist entirely of concatenated bare 252-byte
CCSDS beacons. The last three contain mixed records. Across those files, 544
valid beacons are immediately preceded by the documented 16-byte downlink
AX.25 header:

```text
98 82 a6 a0 40 40 41 a6 aa 9c 62 40 40 41 03 f0
```

This matches destination `LASP-0`, source `SUN1-0`, the literal `0x41` SSID
octets, control `0x03`, and PID `0xF0`. The CCSDS packet starts immediately
after the header. A further 13,726 valid occurrences have no such immediately
preceding header.

For 516 of the 544 valid header-bearing copies, the next AX.25 header begins
immediately after the 252-byte CCSDS packet. There is no room for an FCS at
those boundaries. Testing the two following bytes as a little-endian
CRC-16/X-25 over header plus packet produced zero matches across all 822
header-bearing APID 1 candidates, including rejected candidates. This is
evidence about the saved representation, not a finding of bad on-air FCS.

All 278 rejected beacon candidates occur after that same AX.25 header. Some
overlap the beginning of another valid packet; the files must not be treated
as a uniformly aligned sequence of complete link-layer frames. The audit does
not establish whether damaged/partial copies originated in reception, logging,
or another processing step.

There is no IQ recording in this directory. Flags, bit stuffing, preamble,
frequency deviation, Gaussian filtering, whitening, FEC, and the actual
SatNOGS demodulator output boundary remain unverified. The retained headers
support the software framing description but do not complete RF validation.

The mission owner accepted these headers as valid framing evidence on
2026-09-29. This closes the implementation questions about address order,
callsigns/SSID octets, control/PID, and direct CCSDS encapsulation. The reason
Hydra retained the headers remains uncertain and does not block that decision.
The new local `decode_public_ax25_beacon(..., fcs_present=False)` adapter
accepts all 544 valid header-bearing occurrences, with 60,384 public values
matching the bare decoder, and rejects the 278 checksum-invalid candidates.
The explicit full-FCS mode has independent synthetic tests; the capture does
not validate that mode. The complete local SatNOGS suite passes 126 tests.

## Time and dashboard implications

There are 4,043 valid occurrences with coarse time below one day after the
defined epoch. The capture later contains coarse times consistent with the
2026 test period. The early values appear to precede setting the spacecraft
clock; they must not be presented as actual 2026 reception timestamps.
Checksum validity alone does not establish that a spacecraft timestamp is
usable as observation time. Preserve the raw coarse/fine values and use
verified ground reception metadata for reception age.

This audit completes the local recorded-packet comparison. It does not
establish SatNOGS ingestion field names, timestamp semantics, or displayed
values. The `suncet-dev` Scratchpad queries remain disabled pending that
end-to-end check. Upstream integration and the separate IQ-based receiver
validation remain open.

## Comparison procedure

1. Read each binary and scan every occurrence of primary word `0x0801` and
   length word `0x00f5`, allowing any two sequence bytes. Take 252 bytes from
   each candidate start; validate with `parse_beacon_packet`.
2. Count occurrences by BLAKE2b hash with a 16-byte digest. Decode each
   distinct valid packet with `decode_public_beacon` and separately with the
   private generated `BEACON` class loaded by `import_bus_decoder_bundle`.
3. Compare only the 111 approved aliases to their CTDB source fields. Derive
   primary-header metadata separately with `_packet_header_metadata`. Compare
   state labels exactly and numbers with relative and absolute tolerances of
   `1e-10`; paired NaNs agree. All fields match.
4. Join the checksum-enforced beacon CSV by packet hash, comparing both field
   values and occurrence multiplicities. All fields and multiplicities match.
5. Independently call `packetize_checksum_valid_ccsds` for APID 1 with expected
   length 252, checksum bypass disabled, and playback extraction disabled.
   Its recovered packet multiset matches the direct scan exactly.
6. At candidate offsets, check the literal preceding AX.25 header, subsequent
   record boundary, and possible following CRC-16/X-25. Validate the captured
   version packets separately to corroborate firmware provenance.

This compares separate parser implementations against the same authoritative
CTDB definitions. It does not independently calibrate the hardware sensors or
verify receiver performance.
