# SunCET public beacon

`ksy/suncet.ksy` decodes CTDB 2.0.5 APID 1 telemetry. It exports the 111 fields
defined in the mission's
[public beacon specification](https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md).
SunCET's SatNOGS DB identifier is `MNRC-9829-4319-5529-8975`.

## Receiver input

Supply exactly 268 bytes: the SunCET AX.25 address/control/PID header followed
by one 252-byte CCSDS beacon. The receiver must already have checked and removed
HDLC flags and the two-byte FCS. The literal header is:

```text
98 82 a6 a0 40 40 41 a6 aa 9c 62 40 40 41 03 f0
```

Both address-ending `41` octets are intentional flight values. The decoder
matches this fixed header rather than walking generic AX.25 extension bits.
Construction validates Fletcher-32 across the CCSDS packet, the APID/type/
version/secondary-header word, packet length, and 0–999 millisecond fine time.
Bare CCSDS packets, retained FCS, concatenated frames, and other APIDs fail.

Fletcher-32 uses 124 little-endian 16-bit words, initial accumulators 65535,
modulus 65535, and a final big-endian 32-bit checksum. Its working buffers and
excluded packet regions have anonymous attributes so the standard raw-structure
serializer omits them. No command fields or other APID definitions are exposed.

The field API returns engineering values and **integer state codes**, including
unknown codes. Labels and units are in the public specification. Nonfinite
floating-point readings become NaN for omission by SatNOGS DB. Spacecraft time
remains separate coarse seconds and fine milliseconds; no UTC conversion is
performed here.

## Synthetic checks

`synthetic.hex` is a complete, artificial receiver frame. Its excluded bytes
contain no recorded data. `synthetic_expected.json` lists the public values.
Compile with the repository's Kaitai 0.10 toolchain, then run from its root:

```shell
ksc --target python --outdir satnogsdecoders/decoder ksy/*.ksy
ksc --target javascript --outdir js ksy/*.ksy
python -m pytest tests/test_suncet.py
node contrib/suncet/check_javascript.js
```

Use Python `kaitaistruct==0.10` and JavaScript `kaitai-struct@0.10.0` runtimes.
The JS checker requires Node 18 or later. The Python tests are discovered by
the existing upstream test job; the JavaScript check is an additional manual
check. It covers all public values, corruption at every input byte, invalid
envelopes, nonfinite readings, and unsigned Fletcher values above 2^31.

The mission locally replayed its September 30 flight-unit recording through
the released SatNOGS 9600-baud GFSK/AX.25 receiver with G3RUH descrambling and
CRC checking. All ten beacons passed this decoder with matching public values;
ten other frames were rejected. The original recording and extracted packets
are private and are not included in this contribution.
