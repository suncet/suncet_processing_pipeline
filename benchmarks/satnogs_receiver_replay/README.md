# Offline SatNOGS receiver replay

This validation harness replays private 307.2 ksample/s complex-float32 IQ through
the released SatNOGS GFSK/AX.25 receive chain, then through the mission's strict
public beacon adapter. Run with the **suncet conda environment** on macOS. It is
an integration experiment, not a dependency of production telemetry processing.

The September 30 flight-unit recording passes: **20 CRC-valid frames, ten
public beacons, all ten byte-identical to the independent receive audit**. Every
beacon returns 111 approved fields. Disabling descrambling produces zero frames.
See the [validation record](../../docs/SATNOGS_IQ_VALIDATION_20260930.md) and
[aggregate result](../../docs/assets/satnogs_receiver_replay_20260930.json).

## Pinned sources

- [satnogs-flowgraphs 2.7](https://gitlab.com/librespacefoundation/satnogs/satnogs-flowgraphs/-/tree/2.7),
  commit `75eb7300fdf38ca3e6455d8bed687fc10d680054`.
- [gr-satnogs v3.1.0.1](https://gitlab.com/librespacefoundation/satnogs/gr-satnogs/-/tree/v3.1.0.1),
  commit `4defcfc445c1f94450c48ad348689e7986ebe6f7`.
- Its pinned CRCpp submodule, commit `6da4f42ec110c22d10973f4c16927700f1b52bf1`.

`prepare.py` extracts the bit-slicer's receive ancestors from the release's
`generic/fsk_ax25.grc`, preserving DSP block parameters and connections. Only
its input/output seams are replaced with hierarchical pads. The release
mode map selects this flowgraph for GFSK. A small binding compiles unmodified
upstream AX.25, frame-decoder, metadata, and Doppler C++ sources against the
installed GNU Radio 3.10.5.1. No installed conda files are changed.

The harness also runs the upstream Python Doppler/filter/resampling hierarchy.
The recorded input is already frequency-translated; the upstream dummy Doppler
source supplies zero offset instead of connecting to rigctl. An unused Hamlib
import is removed from that Python file in memory. This is an offline replay of
the actual receiver blocks, not execution of a live station/client/dispatcher.
There are no radio, transmitter, socket, uploader, or public-data sinks.

For the local Apple Clang 21 / Boost 1.78 combination, two copied enum headers
receive explicit `int` underlying types to satisfy the compiler. They retain
their numeric values. Those build-only overlays, downloaded dependencies,
generated sources, and binaries all live under ignored `tmp/`. The binding uses
the installed GNU Radio pybind11 ABI, `v4_libcpp`; other platforms/environments
may require their normal upstream build instead of this macOS helper.

## Reproduce

From the repository root, create an isolated source/build directory. These
commands download public code and headers only; they do not upload recordings.

```sh
conda activate suncet
mkdir -p tmp/satnogs-receiver/include/nlohmann
git clone --depth 1 --branch v3.1.0.1 https://gitlab.com/librespacefoundation/satnogs/gr-satnogs.git tmp/satnogs-receiver/gr-satnogs
git -C tmp/satnogs-receiver/gr-satnogs submodule update --init --depth 1 CRCpp
git clone --depth 1 --branch 2.7 https://gitlab.com/librespacefoundation/satnogs/satnogs-flowgraphs.git tmp/satnogs-receiver/satnogs-flowgraphs
python -m pip install --no-deps --target tmp/satnogs-receiver/build-deps pybind11==2.10.4
curl -fL https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp -o tmp/satnogs-receiver/include/nlohmann/json.hpp
python benchmarks/satnogs_receiver_replay/prepare.py
python benchmarks/satnogs_receiver_replay/replay.py /private/path/to/recording.bin --output tmp/satnogs-receiver/result.json
```

If those checkouts already exist, reuse them with the pinned commits available;
do not run the clone commands again. The helper uses `git show` for the pinned
flowgraph, so it need not change that checkout's current branch. GNU Radio needs
its normal writable temporary shared-memory directory.

The replay runs the standard 9600-baud receiver, a no-descrambling negative
control, and the separate mission receive reconstruction with independent HDLC
and FCS recovery. It compares full frame byte multisets, requires at least one
valid public beacon, and fails on any discrepancy. It also exercises PMT
serialization and the release subscriber's JSON/base64 conversion in memory.
The 16-byte SunCET header is retained; flags and FCS are removed, so the adapter
uses `fcs_present=False` on each 268-byte beacon frame.

Only aggregate counts, settings, and provenance are saved. Decoded frames and
telemetry values remain in memory. The original IQ must remain private: even
APID 1 contains opaque command/status bytes omitted from the public decoder.

Live Network reception, `satnogs-decoders` deployment, timestamp/field ingestion,
dashboard activation, and the 19200-baud contingency mode are separate checks.
