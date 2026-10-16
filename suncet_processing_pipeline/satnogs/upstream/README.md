# SunCET contribution prepared for satnogs-decoders

Prepared and locally validated on 2026-10-05 against upstream commit
`813bbd0c549eb68fa20851c5407b16fbb245efbe`. Submitted 2026-10-06 as
[merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542).
**Awaiting upstream review; not deployed.**

Copy only these paths into the matching upstream checkout:

- `ksy/suncet.ksy`
- `tests/test_suncet.py`
- `contrib/suncet/README.md`
- `contrib/suncet/synthetic.hex`
- `contrib/suncet/synthetic_expected.json`
- `contrib/suncet/check_javascript.js`

This top-level README is a local review guide, not a replacement for upstream's
README. The local `conftest.py` keeps these upstream-only tests out of the
mission test discovery and is also not copied. Compiled Python/JavaScript,
private CTDB files, and recorded packets are
not contribution files. The only packet fixture is synthetic ASCII hex.

## What to review

The KSY accepts exactly one 268-byte receiver frame: the literal 16-byte SunCET
AX.25 header followed by a 252-byte APID 1 beacon. It validates Fletcher-32 and
the full CCSDS envelope when constructed, so SatNOGS's existing decoder and DB
entrypoints enforce the same contract without custom Python changes. Other
APIDs and malformed frames are rejected.

Its 111 `:field` aliases correspond exactly to the reviewed public schema.
Excluded bytes and checksum working buffers have anonymous compiler-generated
names, which the existing full-structure serializer skips. This keeps command
and other excluded values out of both public extraction APIs. Raw input bytes
remain private even when they contain only APID 1.

Engineering values match the mission decoder. Upstream state fields always use
integer wire codes to avoid an InfluxDB string/integer conflict when an unknown
state appears. Existing Grafana mappings accept those codes. Floating-point
NaN and infinities become NaN, which SatNOGS DB omits without losing the rest
of a point. The mission's standalone decoder API remains unchanged.

Review the [synthetic regression tests](tests/test_suncet.py),
[JS checks](contrib/suncet/check_javascript.js), and
[integration record](../../../docs/SATNOGS_DECODER_INTEGRATION_20261005.md).
The [merge-request draft](../../../docs/SATNOGS_DECODER_MR_DRAFT.md) describes
the proposed change for upstream reviewers.

## Reproduce generated contribution files

From the mission repository, with the `suncet` conda environment:

```shell
python -m suncet_processing_pipeline.satnogs.build_upstream
python -m suncet_processing_pipeline.satnogs.build_upstream --check
```

These commands use only the public schema and synthetic fixture. Tests,
documentation, and the JS checker are handwritten. Compilation and execution
instructions are in [contrib/suncet/README.md](contrib/suncet/README.md).

## Submission step

The mission maintainer reviewed the contribution and authorized DCO sign-off
on 2026-10-06 as `James Paul Mason <jmason86@gmail.com>`. Upstream's
[CONTRIBUTING.md](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/blob/813bbd0c549eb68fa20851c5407b16fbb245efbe/CONTRIBUTING.md)
requires these certifications. Local commit
`0edb422fe2293f2abcc4db4dbdd69bbde666849c` contains exactly the six reviewed files
and the authorized sign-off. The public
[fork](https://gitlab.com/jmason86/satnogs-decoders) contains the published
`codex/suncet-beacon-decoder` branch at
`f31782b66352319bc2fb9426d5867d052b497218`, created through GitLab Web IDE with
the same sign-off. Fetching that branch confirmed its complete tree matches
the reviewed local commit. [Merge request !542](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders/-/merge_requests/542)
targets the official repository's `master` branch.
Upstream master was rechecked on October 6 and is unchanged; all 293 upstream
tests still pass, including the maintainer's test-docstring edit.

Follow-up commit `6248b1c6e926496c7d827f43588b12979e440e4f` corrects one test
reference from the dynamically registered `decoder.Suncet` to the directly
imported `Suncet` class after the first remote pylint run flagged it. All other
jobs passed in that run. The corrected six-file bundle matches the fetched
remote tree; 293 local tests and pylint pass. Replacement
[pipeline #2918408670](https://gitlab.com/jmason86/satnogs-decoders/-/pipelines/2918408670)
also passes in full; upstream approval remains pending. The
[integration record](../../../docs/SATNOGS_DECODER_INTEGRATION_20261005.md)
records the initial failure and correction.

After acceptance and deployment, a SatNOGS DB maintainer must associate decoder
`suncet` with satellite `MNRC-9829-4319-5529-8975`. Grafana editor/team access
does not establish permission to perform that DB registration. Validate the
live measurement and fields before enabling dashboard targets.
