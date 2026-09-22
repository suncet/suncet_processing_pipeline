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
FSW confirmed that the current 252-byte compiled layout contains one alignment
byte before Fletcher-32, while CTDB 2.0.1 exports 251 bytes without that byte.
Retain both forms until the planned beacon revision has an authoritative export
and flight-equivalent test packet.

The files under `test_data` are constructed entirely from deterministic public
values. They contain no captured flight/model data and no private CTDB fields.
