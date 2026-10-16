# Local SunCET upstream decoder integration check

`validate.py` runs the private September 30 recording through the released
SatNOGS receiver and proposed upstream decoder, then executes the actual
SatNOGS DB ingestion functions with in-memory file/database substitutes.
It compares all public values with the mission decoder and validates the saved
dashboard's aliases and numeric Canvas state mappings. Only aggregate results
are written. No HTTP requests, DB writes, or packet dumps are performed.

First prepare the [receiver replay build](../satnogs_receiver_replay/README.md).
Clone `satnogs-decoders` at `813bbd0c549eb68fa20851c5407b16fbb245efbe` into
`tmp/satnogs-decoders`, and `satnogs-db` at
`e6ea806331a3d075c88eab1890a41ab8bcb93f7d` into `tmp/satnogs-db`. Copy the six
[contribution files](../../suncet_processing_pipeline/satnogs/upstream/README.md)
into the decoder checkout and compile all its KSY files with Kaitai 0.10.

Keep the upstream runtime isolated from conda `suncet`'s installed 0.11:

```shell
conda activate suncet
python -m pip install --target tmp/satnogs-toolchain/python-deps kaitaistruct==0.10
PYTHONPATH="$PWD/tmp/satnogs-toolchain/python-deps" \
  python benchmarks/satnogs_decoder_integration/validate.py \
  /private/path/to/SunCET_UHF1790796500.bin \
  --output docs/assets/satnogs_decoder_integration_20261005.json
```

The child mission-reference process removes this runtime override and uses the
conda environment's 0.11. Frames and values pass only through local memory
pipes. The output includes aggregate counts, booleans, public identifiers, and
source hashes; it does not expose excluded telemetry. GNU Radio may need local
sandbox permission to allocate its temporary shared-memory buffers.

Assertions require the known recording's 20 receiver frames and 10 public
beacons, matching 111 public values, safe unknown/nonfinite handling, and all
82 saved dashboard aliases. A different recording/layout requires an explicit
validation-plan update. This is a source-level integration check; it does not
exercise a real Influx server or hosted Grafana queries.

Results and upstream test/build checks are recorded in the
[integration report](../../docs/SATNOGS_DECODER_INTEGRATION_20261005.md).
