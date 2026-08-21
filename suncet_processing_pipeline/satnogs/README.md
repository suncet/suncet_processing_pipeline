# SunCET SatNOGS decoder artifacts

This directory contains the mission-owned public APID 1 schema, provisional
Kaitai decoder, and non-flight interoperability fixture.

Regenerate the tracked KSY and fixture after an approved schema change:

```shell
python -m suncet_processing_pipeline.satnogs.kaitai_generator
python -m suncet_processing_pipeline.satnogs.synthetic_fixture
```

The generated decoder currently starts at the bare CCSDS packet. Do not add an
AX.25 wrapper by assumption: wait for the confirmed flight frame and the output
boundary of the selected SatNOGS receiver path. Likewise, retain raw fine time
and both candidate packet lengths until their flight definitions are resolved.

The files under `test_data` are constructed entirely from deterministic public
values. They contain no captured flight/model data and no private CTDB fields.
