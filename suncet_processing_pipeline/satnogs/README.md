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

Recent recorded UHF telemetry has been requested for the final independent
raw/engineering-value comparison. The requested IQ capture is a separate RF
reception test. The flight-software AX.25 header and FCS are known, but an RF
test must establish the selected receiver's output boundary before adding a
link-layer adapter. Current input is exactly one bare CCSDS APID 1 packet.

See the [onboarding plan](../../docs/SATNOGS_ONBOARDING_PLAN.md) and
[dashboard plan](../../docs/SATNOGS_DASHBOARD_PLAN.md) for upstream integration,
editor access, and the remaining operational gates.
