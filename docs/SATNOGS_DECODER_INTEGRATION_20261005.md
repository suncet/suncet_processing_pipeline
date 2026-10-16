# SunCET upstream decoder and DB integration validation

Validated 2026-10-05. **Local integration passes; upstream review, deployment,
and hosted dashboard ingestion remain pending.** Submitted 2026-10-06 as
[merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542)
after mission-maintainer review and DCO approval. Published commit `f31782b6`
matches the reviewed local contribution tree exactly.

The [contribution bundle](../suncet_processing_pipeline/satnogs/upstream/README.md)
adds a pure Kaitai 0.10 SunCET decoder to the existing upstream discovery and
field-extraction mechanism. The [aggregate report](assets/satnogs_decoder_integration_20261005.json)
contains no packet bytes or recorded telemetry values.

## Results

| Check | Result |
| --- | --- |
| Kaitai 0.10 compilation | All 164 upstream KSY files compile to Python and JavaScript |
| Python regression suite | 293 pass: 286 SunCET cases and 7 existing upstream tests |
| JavaScript runtime 0.10.0 | Public values, every protected byte, invalid boundaries/envelopes, nonfinite data, and unsigned checksum checks pass |
| Upstream static checks | yamllint, flake8, isort, yapf, pylint pass |
| Distribution | Source distribution and wheel build; separately installed wheel discovers `suncet` and decodes all 111 fixture values |
| Mission regression suite | 144 SatNOGS tests pass in conda `suncet` |
| Flight IQ → released receiver → compiled decoder → DB functions | 20 received frames; 10 APID 1 beacons accepted; 10 other frames rejected |
| Public values | All 111 fields in each accepted beacon match the mission decoder |
| Saved dashboard mapping | All 82 queried aliases exist in all ten decoded points; four Canvas state panels map numeric codes correctly |

Python checks used 3.10 on macOS, not the upstream Python 3.9/Linux CI image.
The first remote [pipeline #2918373215](https://gitlab.com/jmason86/satnogs-decoders/-/pipelines/2918373215)
passed compilation, sign-off, both builds, dependency checks, and the test suite.
Only pylint failed: `tests/test_suncet.py:38` accessed the dynamically registered
`decoder.Suncet` attribute, which CI's static analysis could not infer. The local
Python 3.10 pylint run did not reproduce that diagnostic.

Follow-up commit `6248b1c6e926496c7d827f43588b12979e440e4f` uses the already
imported `Suncet` class for that assertion. Runtime discovery remains covered by
the existing `decode_frame_to_fields` call; no decoder behavior or checks were
changed. All 293 tests and pylint pass locally in conda `suncet`, and the fetched
remote tree exactly matches the corrected local files. Replacement
[pipeline #2918408670](https://gitlab.com/jmason86/satnogs-decoders/-/pipelines/2918408670)
passed on 2026-10-06, including the Python 3.9/Linux static, build, dependency,
and test jobs. The merge request now awaits one eligible upstream approval.
Actual server deployment remains pending.

## Implementation

The upstream parser accepts exactly 268 bytes: the literal 16-byte flight
AX.25 header plus the 252-byte CCSDS beacon. It validates Fletcher-32 on all
124 little-endian packet words, then validates the CCSDS header, APID, packet
length, and fine time. These checks run on construction, including when DB
calls `Suncet.from_bytes` directly. Corrupt excluded bytes therefore fail too.

The 111 public aliases and engineering conversions are unchanged. Two upstream
representations differ from the mission's standalone Python API:

- State fields always return integer wire codes, including unknown codes.
  This avoids mixing enum strings and unknown integers in one Influx field.
  The existing dashboard supports those numeric codes and supplies the labels
  and colors. One-bit state fields are integer 0/1 rather than Python booleans.
- Nonfinite floating-point readings become NaN. The upstream DB removes NaN
  fields but does not remove infinity. Synthetic NaN and both infinities each
  result in 110 retained fields, while the rest of the point remains usable.

Excluded packet regions and Fletcher working buffers are anonymous attributes;
the existing raw-structure serializer skips them. The checksum remains enforced
without exposing the words used to compute it. Only a synthetic packet is
included for public regression tests. APID 1 filtering does not make captured
raw bytes suitable for publication.

## DB and dashboard contract

The harness executes the unchanged `decode_demoddata`, `create_point`,
`flatten_data`, and `remove_nan_fields` functions extracted from the pinned
SatNOGS DB source. In-memory substitutes replace file reads, model updates,
and the Influx writer. No Django service, Influx server, or hosted service is
started or contacted. Reception metadata in this test is synthetic.

- Decoder registration name: `suncet`; class: `Suncet`.
- Measurement: the satellite's NORAD number. `98244` is the current proposed
  SunCET measurement because its SatNOGS catalog number is temporary.
- Fields: flat approved aliases, already in engineering units; no second scale
  conversion in Grafana. State fields are numeric.
- Time: `demoddata.timestamp`, formatted by DB to whole UTC seconds. This is
  reception metadata, not reconstructed spacecraft clock time.
- Tags: `satellite`, `sat_id`, `decoder`, `station`, `observer`, `source`, and
  `version`. This code does not add a separate observation-ID tag.

The saved hosted dashboard (UID `ffzqpb9arppfkb`, version 5) uses 82 of these
aliases in 38 telemetry panels. Its state mappings already accept numeric
codes; no visual or query edit was necessary. Targets remain disabled. This
check establishes source-level compatibility, not field presence in the live
Influx database or successful query/render behavior.

After release and DB association, confirm the actual measurement, field types,
decoder/version/source tags, timestamp behavior, packet-coherent latest
summaries, and missing/nonfinite/unknown-state behavior in Grafana. Add reception
age and verify an end-to-end displayed observation before enabling operational
queries. Do not upload the mixed recording or raw APID 1 packets for that check.

## Reproduction and provenance

Use the [local validation harness](../benchmarks/satnogs_decoder_integration/README.md).
It runs the upstream decoder and DB functions with Kaitai runtime 0.10. The
separate mission reference runs with 0.11 through local memory pipes, keeping
the two runtime contracts independent.

- `satnogs-decoders`: `813bbd0c549eb68fa20851c5407b16fbb245efbe` plus the six
  contribution files.
- `satnogs-db`: `e6ea806331a3d075c88eab1890a41ab8bcb93f7d`.
- Official Kaitai 0.10 compiler ZIP SHA-256:
  `3d11d6cc46d058afb4680fda2e7195f645ca03b2843501d652a529646e55d16b`.
- IQ input SHA-256:
  `30ff20c8c860630d4168d81ea1fcb8b4d966029de5b0a66c2c360ca25673ccbe`.
- Receiver build pins and DB-source hash are preserved in the aggregate report.

The mission owner confirmed the September 30 recording came from the actual
flight unit running FSW/CTDB 2.0.5, with no known flowgraph changes. The
[earlier IQ audit](SATNOGS_IQ_VALIDATION_20260930.md) supplies its provenance and
receiver-validation limits. Live Network reception, 19200-baud validation, and
authoritative spacecraft UTC conversion remain separate work.
