# SunCET SatNOGS decoder artifacts

This directory contains the mission-owned public APID 1 schema, provisional
Kaitai decoder, and non-flight interoperability fixture.

Regenerate the tracked KSY and fixture after an approved schema change:

```shell
python -m suncet_processing_pipeline.satnogs.kaitai_generator
python -m suncet_processing_pipeline.satnogs.synthetic_fixture
```

The generated decoder currently starts at the bare CCSDS packet. The
flight-software AX.25 header and FCS are confirmed, but do not wrap the Kaitai
input until an RF frame establishes the output boundary of the selected SatNOGS
receiver path. Fine time is an integer millisecond field constrained to 0-999.
Retain both candidate packet lengths until the flight packet length is resolved.

The files under `test_data` are constructed entirely from deterministic public
values. They contain no captured flight/model data and no private CTDB fields.
