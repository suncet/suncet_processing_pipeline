# Add SunCET APID 1 beacon decoder

Submitted 2026-10-06 as [upstream merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542).
The text below records the submitted description.

---

Add the SunCET CTDB 2.0.5 public beacon decoder and a synthetic regression
fixture. Input is one 268-byte frame from the GFSK/AX.25 receiver: the literal
16-byte SunCET header followed by the 252-byte CCSDS packet, with flags and FCS
already removed. The parser enforces APID 1, the complete packet envelope, and
Fletcher-32 during construction, including through the existing DB entrypoint.

The decoder exposes 111 reviewed public engineering fields. Excluded regions
and checksum buffers are anonymous and omitted by the raw-structure API.
State values retain integer codes for stable Influx field types, and nonfinite
floating-point readings become NaN for DB omission. No private definitions or
recorded packets are included.

Validation on macOS with Python 3.10 in the mission's `suncet` environment:

- All 164 schemas compile to Python and JavaScript with Kaitai 0.10.
- 293 upstream Python tests pass, including 286 new SunCET cases; all five
  repository static checks pass.
- Additional JavaScript checks pass with runtime 0.10.0, including unsigned
  checksums above 2^31. The built Python wheel decodes the synthetic fixture.
- Private local receiver-to-DB-function replay accepts ten flight-unit beacons,
  with all 111 public values matching the mission decoder, and rejects ten
  other frames. This used in-memory DB substitutes; it was not a hosted upload.

[Public beacon specification](https://github.com/suncet/suncet_processing_pipeline/blob/main/docs/SUNCET_PUBLIC_BEACON_SPEC.md)
· [SunCET DB entry](https://db.satnogs.org/satellite/MNRC-9829-4319-5529-8975)

---

Mission-maintainer review and DCO sign-off were authorized on 2026-10-06 as
`James Paul Mason <jmason86@gmail.com>`. Published commit
`f31782b66352319bc2fb9426d5867d052b497218` includes that sign-off. Its complete
tree matches the reviewed local commit `0edb422`; only commit metadata differs.
The contribution contains exactly the six allowlisted public files.
The first CI run failed only pylint's inference of `decoder.Suncet` in a test;
all other jobs passed. Follow-up commit `6248b1c6` uses the test's direct class
import and retains the approved sign-off. Replacement
[pipeline #2918408670](https://gitlab.com/jmason86/satnogs-decoders/-/pipelines/2918408670)
passed on 2026-10-06. The merge request awaits one eligible upstream approval.
After the decoder is merged and deployed, request its association with the
SunCET DB entry; that association is separate from this source contribution.
