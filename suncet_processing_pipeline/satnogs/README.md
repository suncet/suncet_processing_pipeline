# SunCET SatNOGS decoder artifacts

This directory implements the mission-approved public APID 1 interface for
**CTDB 2.0.5 only**: 111 public fields in a 252-byte bare CCSDS packet. No legacy
layout is retained. Use captures known to follow 2.0.5; packet length cannot
distinguish a historical layout of the same size.

The public CSV contains reviewed fields and current offsets, types, and
conversions. Excluded regions are anonymous gaps in the Kaitai definition and
are never returned by the public entrypoint:

```python
from suncet_processing_pipeline.satnogs.decoder import decode_public_beacon

values = decode_public_beacon(packet_bytes)
```

This entrypoint checks the exact length, CCSDS header, integer fine time
(0–999 milliseconds), and Fletcher-32 **before** constructing the compiled
Kaitai parser. It returns approved engineering values and uppercase state
labels; unknown states remain integers. Raw floating-point NaN/infinity remain
nonfinite values and should be displayed as missing by downstream dashboards.
It does not invent a spacecraft-time-to-UTC or leap-second policy. Calling the
generated parser directly performs structural checks but does not check Fletcher-32.

## AX.25 input

The mission has accepted the literal header found in the 2026-09-23 Hydra
capture as framing evidence. The local adapter handles one complete record:

```python
from suncet_processing_pipeline.satnogs.ax25 import decode_public_ax25_beacon

values = decode_public_ax25_beacon(frame_bytes, fcs_present=False)
```

Declare the receiver boundary explicitly:

- `fcs_present=False`: exactly 268 bytes, the literal 16-byte SunCET header
  followed by the 252-byte CCSDS beacon. This path passed all 544 valid
  header-bearing occurrences in the Hydra capture and rejected all 278
  checksum-invalid candidates. It also passes all ten flight-unit beacons
  from the released SatNOGS GFSK/AX.25 receiver's local IQ replay.
- `fcs_present=True`: exactly 270 bytes, including the two little-endian
  CRC-16/X-25 bytes. The adapter checks the FCS over header plus packet. This
  path passes all ten flight-unit beacons recovered with retained FCS from the
  [2026-09-30 IQ recording](../../docs/SATNOGS_IQ_VALIDATION_20260930.md),
  in addition to its synthetic tests.

Both modes enforce the literal address/control/PID bytes and the complete
CCSDS Fletcher-32 contract, then return the same 111 approved fields. Flags,
KISS transport, extra bytes, and mixed Hydra streams are not accepted as a
single frame. Call the existing bare-packet entrypoint if a receiver removes
the AX.25 header. No receiver mode is selected automatically.

## Reproduce the artifacts

Use the `suncet` conda environment. The Python runtime is pinned to
`kaitaistruct==0.11`. Install the official
[Kaitai compiler 0.11](https://github.com/kaitai-io/kaitai_struct_compiler/releases/tag/0.11)
(Java required) and run:

```shell
conda activate suncet
python -m suncet_processing_pipeline.satnogs.build_decoder --compiler /path/to/kaitai-struct-compiler
python -m suncet_processing_pipeline.satnogs.build_decoder --check --compiler /path/to/kaitai-struct-compiler
python -m pytest suncet_processing_pipeline/tests/satnogs*_test.py
```

The command generates the KSY, compiles `generated_suncet_apid1.py`, and builds
the synthetic packet and expected values. It normalizes trailing whitespace
in compiler output. `--check` compares all four artifacts without writing. CI
downloads the official 0.11 ZIP, verifies SHA-256
`ff89389d9dc9e770d78a24af328763cb1f8e7b31ce7766c9edf10669a060f2a2`,
checks generation, exercises the compiled decoder, and verifies decoding from
the installed wheel. Compiler changes require an explicit toolchain update.

The synthetic fixture covers every approved field and contains no captured
telemetry or private CTDB values. Separate manually packed tests check offsets,
signed/scaled values, enums, packed fields, opaque gaps, and invalid packets.

## Remaining validation

The [recorded Hydra comparison](../../docs/SATNOGS_UHF_VALIDATION_20260923.md)
passes for all approved values. Header layout and direct CCSDS encapsulation
are accepted for implementation. The supplied flight-unit IQ capture also
passes the [offline 9600-baud receive audit](../../docs/SATNOGS_IQ_VALIDATION_20260930.md):
ten public beacons pass both checksums. The reconstructed GNU Radio HDLC path
removes flags and FCS and retains the 16-byte address/control/PID header, so it
uses `fcs_present=False`. An independent deframer retaining FCS passes the
`True` path on the same frames. A subsequent
[released SatNOGS receiver replay](../../benchmarks/satnogs_receiver_replay/README.md)
also recovers all ten beacons byte-for-byte with standard settings. Its output
is 268 bytes, so use `fcs_present=False`. Receiver selection and the local
output boundary are established for this recording; live Network reception
and hosted telemetry ingestion remain untested. Local upstream decoder and
DB-function integration now pass as described below.

The receive-only `iq_audit.py` recipe uses GNU Radio, NumPy and SciPy from the
`suncet` environment and writes aggregate evidence only:

```shell
python -m suncet_processing_pipeline.satnogs.iq_audit /path/to/recording.bin \
  --output /path/to/aggregate-report.json
```

The 307.2 ksample/s complex-float32 input is already at baseband. The tool never
opens radio hardware or sockets. Raw IQ, intermediate bits, and extracted raw
beacons must remain private: filtering for APID 1 alone does not remove its
opaque, non-public fields. No capture bytes or decoded telemetry are written
to the aggregate report.

## Upstream contribution

The [prepared contribution](upstream/README.md) targets SatNOGS's Kaitai 0.10
Python and JavaScript toolchain. Its separate `ksy/suncet.ksy` accepts the
verified 268-byte receiver frame and validates Fletcher-32 plus the CCSDS
envelope directly during construction. Anonymous excluded regions and checksum
working buffers remain hidden from the standard full-structure API.

All 111 field aliases and engineering conversions are preserved. For upstream
DB storage, enum states always use integer codes (including unknown codes),
and nonfinite floating-point values become NaN for DB omission. These choices
avoid Influx field-type conflicts and infinity-related write failures. The
mission's standalone API above retains its existing representation.

```shell
python -m suncet_processing_pipeline.satnogs.build_upstream
python -m suncet_processing_pipeline.satnogs.build_upstream --check
```

The [October 5 integration record](../../docs/SATNOGS_DECODER_INTEGRATION_20261005.md)
documents 293 passing upstream Python tests, JavaScript/static/package checks,
and private receiver-to-DB-function validation with all ten flight beacons and
111 public fields matching. All 82 saved dashboard aliases match the resulting
points. After mission-maintainer review and DCO approval, the six-file
contribution was submitted on 2026-10-06 as
[merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542).
Published commit `f31782b6` matches the reviewed local contribution tree.
Follow-up `6248b1c6` fixes a test-only pylint reference; remote CI passes.
Upstream acceptance, deployment, DB association, and hosted ingestion
verification remain.

References: [upstream CI](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/blob/master/.gitlab-ci.yml),
[decoder entrypoint](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/blob/master/satnogsdecoders/decode_frame.py),
[field extraction](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/blob/master/satnogsdecoders/decoder/__init__.py).

See the [onboarding plan](../../docs/SATNOGS_ONBOARDING_PLAN.md) and
[dashboard plan](../../docs/SATNOGS_DASHBOARD_PLAN.md) for upstream integration,
editor access, and the remaining operational gates.
