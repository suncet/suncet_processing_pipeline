# Flight-unit IQ validation — 2026-09-30

Inspected 2026-10-05 using the `suncet` conda environment. **Offline reception
passes for the recorded 9600-baud mode:** ten distinct APID 1 beacons pass both
AX.25 CRC-16/X-25 and the SunCET Fletcher-32 contract. Ten additional CRC-valid
frames were excluded from the public-beacon analysis. No recording or packet
bytes were uploaded or added to repository artifacts.

The mission owner confirmed that this recording came from the actual flight
unit running FSW 2.0.5. The owner believes the flowgraph was unchanged during
recording; its saved settings are corroborating evidence, not a configuration
readback or calibrated frequency reference.

## Inputs and reproducibility

Input: `test_data/2026-09-30_uhf_beacon_iq/SunCET_UHF1790796500.bin`.
The [aggregate report](assets/satnogs_iq_validation_20260930.json) records its
SHA-256, size, receiver settings, framing counts, and waveform estimates.

| Input property | Result |
| --- | --- |
| File size | 74,712,960 bytes |
| Sample representation | Little-endian complex float32, interleaved I/Q |
| Complex samples | 9,339,120 |
| Sample rate used | 307,200 samples/s |
| Duration at that rate | 30.40078125 s |
| IQ SHA-256 | `30ff20c8c860630d4168d81ea1fcb8b4d966029de5b0a66c2c360ca25673ccbe` |

The supplied `suncet_uhf_demodulator` repository contains:

- `LASP_UHF_SDR_2_1_4_SUNCET.grc`, SHA-256
  `cb62aa05e829a60ac95da395e7f39c3d32db9d61c017eaa31cb4eaad21afa5f5`.
- `LASP_UHF_SDR_2_1_4b_SUNCET.py`, SHA-256
  `241e4c3f0265a9cabd2267926905d2968de85977f2913a5be9b6bce0cd55f95c`.

The GRC connects its optional complex file sink after the +100 kHz digital
mixer and before receive filtering/resampling. Its USRP tuning is 401.3 MHz,
corresponding to a nominal 401.2 MHz signal after translation. The file-sink
blocks are disabled in the supplied saved GRC, and absent from its exported
Python, so neither file is a complete record of the capture session. Successful
baseband decoding without an additional frequency translation supports this
interpretation of the recording point.

The supplied application includes transmit/UHD/network blocks and missing
out-of-tree `ais`, `lithium`, and `tr` modules. It was read but not executed.
The new [receive-only audit](../suncet_processing_pipeline/satnogs/iq_audit.py)
uses installed GNU Radio 3.10.5.1, NumPy, and SciPy:

```shell
conda activate suncet
python -m suncet_processing_pipeline.satnogs.iq_audit \
  /path/to/SunCET_UHF1790796500.bin \
  --output /path/to/aggregate-report.json
```

It filters with a 257-tap Hamming FIR at 9 kHz and resamples 5/16 to 96 ksample/s.
It then uses the supplied GFSK branch's 10 samples/symbol, sensitivity π/10,
M&M gain 0.175, and relative timing limit 0.005, followed by descrambling,
differential decoding, inversion, and HDLC. GNU Radio 3.10 uses `symbol_sync_ff`
inside `gfsk_demod`; this is a receive reconstruction, not bit-identical
execution of the supplied GNU Radio 3.8 application. There are no hardware,
transmit, or network blocks in the audit tool.

## Results

| Check | Result |
| --- | --- |
| GNU Radio CRC-valid frames | 20 |
| Independent flag/bit-stuff/FCS recovery | Same 20 frames, byte-for-byte after removing FCS |
| Current public APID 1 beacons | 10 distinct; all pass Fletcher-32 |
| Other CRC-valid frames | 10 excluded; no decoded content or identifiers exported |
| Public fields returned | 111 per beacon |
| Full beacon frame | 270 bytes: 16 header + 252 CCSDS + 2 FCS |
| GNU Radio PDU boundary | 268 bytes: header retained; flags and FCS removed |
| Recovered opening / closing flag runs | 20 / 5 consecutive `0x7E` flags for every beacon |
| Stuffed zeros within beacon frames | 9–13 per frame |
| Beacon spacing | Approximately 3 seconds in this recording |

Both unusual literal `0x41` address octets survive on air. The full-FCS
`decode_public_ax25_beacon(..., fcs_present=True)` path now has captured-RF
validation. GNU Radio's stripped result uses `fcs_present=False` and yields
identical beacon bytes. Both paths apply the strict packet contract and return
only the 111 approved fields. Generic AX.25 address-extension parsing must not
replace the documented literal-header contract.

The bitstream confirms:

- Self-synchronizing G3RUH descrambling: `b[n] XOR b[n-12] XOR b[n-17]`, matching
  GNU Radio mask `0x21`, seed `0`, length parameter `16`.
- NRZI decoding with no transition representing 1.
- HDLC `0x7E` flags, zero insertion after five consecutive ones, and LSB-first
  byte transmission.
- Little-endian stored CRC-16/X-25 over header and information bytes.
- Direct CCSDS encapsulation, with no FEC or deinterleaving operation needed
  to recover these beacons.

The audit independently reproduces GNU Radio's descrambler and NRZI bits in
NumPy, then independently finds flags, removes stuffed zeros, packs bytes, and
checks every CRC. Omitting either descrambling or NRZI decoding gives zero
CRC-valid frames. Swapping their order recovers the same 20 frames after the
startup transient. That agrees with the processing order supported by the
[gr-satellites AX.25 deframer](https://github.com/daniestevez/gr-satellites/blob/main/python/components/deframers/ax25_deframer.py).
GNU Radio's [HDLC implementation](https://github.com/gnuradio/gnuradio/blob/main/gr-digital/lib/hdlc_deframer_bp_impl.cc)
checks CRC and removes the final two bytes before emitting its PDU; the two
independent recovered frame sets also verify that behavior in this audit.

The SunCET SatNOGS test suite passes **144 tests**, including new checks for
HDLC stuffing, aborts, byte/FCS preservation, corrupt data, and rejection of
non-public or internally invalid packets even when their link CRC is valid.

## Waveform estimate and remaining limits

The median positive/negative frequency-plateau separation gives approximately
**±2.4 kHz deviation** (per-beacon estimates 2,381.6–2,393.1 Hz; median
2,389.7 Hz). The median baseband carrier offset is approximately −113 Hz.
These are estimates after receive filtering, assuming the declared sample
rate. They do not establish calibrated absolute RF center frequency.

The supplied `Fdeviation = 4800` variable controls receive sensitivity; it does
not establish the flight transmitter's deviation. Its transmit-side Gaussian
`BT = 0.4` and modulation index `0.5` likewise do not prove the downlink's
Gaussian BT. Gaussian shaping parameters and the complete burst preamble
before the recovered flags remain unconfirmed. The 9600-baud result does not
validate the contingency 19200-baud mode.

The released SatNOGS receiver now passes the local replay described below.
The remaining integration task is to preserve the SunCET literal header and
Fletcher-32 contract in `satnogs-decoders`, then verify ingestion fields and
timestamps before enabling the dashboard. No paired independent byte recording from this September 30
session was supplied. Checksums and the two deframers give strong reception
consistency evidence, but are not an independent capture comparison or a
receiver sensitivity/packet-loss characterization.

## Released SatNOGS receiver replay — passed 2026-10-05

The [reproducible harness](../benchmarks/satnogs_receiver_replay/README.md) uses
`satnogs-flowgraphs` **2.7**, commit
`75eb7300fdf38ca3e6455d8bed687fc10d680054`, and `gr-satnogs` **v3.1.0.1**,
commit `4defcfc445c1f94450c48ad348689e7986ebe6f7`, with GNU Radio 3.10.5.1
in the `suncet` environment. The release dispatcher maps **GFSK** to
`satnogs_fsk_ax25.py`. The harness preserves that flowgraph's demodulator
blocks, parameters, and connections and compiles the unmodified upstream
AX.25/frame-decoder implementation. This is a local replay of released receiver
blocks; it does not establish which software version any live station runs.

Only the offline boundaries change: a vector source supplies the recorded IQ,
the upstream dummy Doppler source supplies zero offset instead of rigctl, and
an in-memory message sink replaces the station output. Upstream filtering and
resampling convert 307.2 to 57.6 ksample/s; the standard quadrature gain is 1.2,
G3RUH descrambling and CRC checking are enabled, and one-bit error correction
is disabled. No custom demodulator tuning was required.

The [aggregate replay result](assets/satnogs_receiver_replay_20260930.json)
records:

| Check | Result |
| --- | --- |
| Upstream receiver CRC-valid frames | 20 |
| Public beacons | 10 distinct, all passing Fletcher-32 and the 111-field contract |
| Other frames | 10 excluded from public decoding |
| Comparison with separate receive reconstruction and independent HDLC recovery | All 20 frames identical byte-for-byte |
| Beacon output boundary | 268 bytes: literal 16-byte header + 252-byte CCSDS packet; no flags/FCS |
| PMT serialization and subscriber JSON/base64 conversion | Preserve every frame byte in memory |
| Negative control: G3RUH descrambling disabled | Zero CRC-valid frames |

Use `decode_public_ax25_beacon(frame, fcs_present=False)` for this receiver.
Its AX.25 decoder does not walk the unusual SunCET address-extension bits, so
the literal `0x41` octets pass through intact. It validates and strips the link
FCS; the mission adapter still validates the internal Fletcher-32 checksum.

The nominal receiver-selection and local input-boundary questions are closed
for this recording. Live Network reception, upstream telemetry-decoder
deployment, field/time ingestion, and dashboard activation remain separate.
The PMT metadata `time` in an offline replay is processing time, not a verified
September 30 acquisition timestamp; it must not be ingested as flight data.
No raw or decoded data was uploaded.

The subsequent [October 5 decoder integration check](SATNOGS_DECODER_INTEGRATION_20261005.md)
also passes this recording through a compiled upstream Kaitai 0.10 decoder and
the actual DB ingestion functions with in-memory I/O substitutes. All ten
beacons yield 111 matching public values, and all ten other frames are rejected.
The prepared wrapper preserves the literal header and checks Fletcher-32 on
construction. Upstream submission/deployment and hosted ingestion remain open.

## Privacy and sharing boundary

The original IQ is a mixed-packet recording and remains private. Intermediate
symbol streams and raw extracted beacons stay in ignored local working files.
The tracked report contains only aggregate receiver/framing evidence and
approximate relative frame positions; it contains no waveform samples, packet
bytes, or decoded telemetry values.

**A beacon-only IQ crop would still encode all APID 1 bytes**, including the
opaque commanding/status regions excluded from the public decoder. Selecting
APID 1 alone is therefore not sufficient to produce a sanitized public test
vector. Use the existing synthetic public fixture for upstream examples until
a separately reviewed sharing approach is agreed. Approximate symbol-clock
positions in this audit must not be used as approved IQ crop boundaries.
