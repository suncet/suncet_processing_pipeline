# SunCET development dashboard

Created 2026-09-29 in SatNOGS Grafana 11.6.0:
[suncet-dev](https://dashboard.satnogs.org/d/ffzqpb9arppfkb/suncet-dev),
in **Scratchpad** (folder UID `28WO65umk`). Dashboard UID `ffzqpb9arppfkb`,
instance ID `496`. The saved export is version 5 (reviewed visuals published 2026-09-29).

The mission owner reports membership in the SunCET team. Sign-in, creation,
and editing in Scratchpad were verified. Team membership and folder ACLs were
not changed. AEPEX was read/exported as a layout reference; neither AEPEX nor
any other existing dashboard was edited.

## Artifacts

- `build_online_dashboard.py` creates `suncet-dev.json`, a reproducible **new-dashboard
  import template** with null ID/UID. It extends the original panel definitions
  in `build_dashboard.py`. Run it using the `suncet` conda environment.
- `suncet-dev.saved.json` is the actual Grafana export of the saved SunCET
  dashboard. It preserves the existing SunCET identity for subsequent work.
- The public CSV is the only telemetry schema input. The current layout covers
  82 approved aliases in 38 telemetry panels, 3 text/image panels, and 9 section
  rows when the optional spacecraft image is supplied.

```shell
conda activate suncet
python -m suncet_processing_pipeline.satnogs.dashboard.build_online_dashboard \
  --image /path/to/spacecraft-render.png \
  --output suncet_processing_pipeline/satnogs/dashboard/suncet-dev.json
```

Do not re-import the new-dashboard template to update the existing dashboard;
that creates a second dashboard. Edit the SunCET UID above and export the saved
result again. Keep the generator synchronized with layout changes. Do not
reuse any reference dashboard's ID, UID, queries, or thresholds.

## Data connection status

The observed production datasource is **SatNOGS DB**, InfluxDB UID `000000001`.
AEPEX uses a NORAD-number measurement and source metadata such as `pid` and
`observer`. A focused `SHOW FIELD KEYS FROM "98244"` query through Grafana
returned HTTP 200 with no frames on 2026-09-29. Thus no SunCET field names have
been verified in that measurement. `98244` is SunCET's temporary SatNOGS NORAD
identifier; the eventual ingestion measurement may differ.

A recorded-data preview was investigated on 2026-09-29. The account's panel
source picker offers `SatNOGS DB`, `SatNOGS DB development`, and the built-in
Dashboard/Grafana/Mixed sources, but no TestData source for inline CSV content.
The mission owner prefers to skip a hosted preview if it needs a workaround.
No capture was uploaded and the temporary panel was discarded. A configured
preview source or an agreed development-DB ingestion path would be needed
before putting these recorded values into the hosted dashboard.

Every hosted target is deliberately `hide: true` (disabled). Its InfluxQL is a staged
mapping from approved public aliases to a proposed `98244` measurement, not a
claim that those fields have been ingested. No synthetic data or mock source is
connected. The reception section is an explicit pending-data note, not a live
freshness calculation. Empty panels were verified in Grafana.

Queries assume the approved engineering conversions; there is no second
scaling pass. State mappings support the public string labels
and numeric codes. Unknown values remain visible numerically rather than being
mapped to a healthy state. Inferred units are named in panel descriptions.
Latest summaries and histogram use one row, preserving a common packet time.
Time-series panels draw points without connecting gaps or carrying values
forward. The saved dashboard has visual voltage bands but no alert rules or auto-refresh.
Native Canvas state panels query aliases of the actual source enumeration;
missing or unknown data leaves every choice dimmed. The reported-value line
shows the raw unknown code or **No data**. The histogram query selects one
packet and transforms its six fields into bin/count rows.

The [October 5 integration check](../../../docs/SATNOGS_DECODER_INTEGRATION_20261005.md)
confirms all 82 query aliases against actual flight beacons passed through the
proposed upstream decoder and unchanged DB ingestion functions, with in-memory
I/O substitutes. The upstream representation consistently uses numeric state
codes; the existing Canvas mappings accept them. DB uses flat aliases, omits
NaN readings, and formats reception time to whole UTC seconds. Live Influx
field presence and Grafana query/render behavior remain to be verified after
decoder acceptance, deployment, and SunCET DB association.

## Local recorded preview

A separate [local Grafana preview](http://127.0.0.1:3107/d/suncet-recorded-local/suncet-dev-c2b7-recorded-preview)
was created on 2026-09-29 with Grafana 11.6.0, matching SatNOGS. It uses the same
initial layout, expanded to 39 telemetry panels with 82 public fields from 5,241 distinct, checksum-valid
2026-09-23 CPT beacons. The reviewed visuals have also been applied to the hosted
dashboard; recorded values and the extra earlier-packet histogram remain local.
Hosted datasource queries remain disabled pending ingestion verification.

Use **Edit**, adjust panels, then **Save dashboard**. Saved edits persist in the
Docker volume `suncet-grafana-preview-data`. This instance listens only on
`127.0.0.1:3107` and allows local editing without sign-in. Leave Docker Desktop
running while using it. From the repository root:

```shell
docker compose -f tmp/suncet-grafana-preview/compose.yaml stop
docker compose -f tmp/suncet-grafana-preview/compose.yaml start
```

The built-in TestData datasource supplies typed, embedded Grafana DataFrames
(`raw_frame` scenario); these contain decoded public values, not raw packet
bytes. Numeric clock/uptime fields retain numeric types. Non-finite values
become missing values, and ordinary recorded outliers remain visible.

The time axis uses the reported spacecraft clock against the declared epoch,
with provisional UTC formatting, not reception timestamps. The default range
covers 19:04–23:36 on September 23. Deduplication removes 6,918 repeated packet
occurrences; 2,111 distinct packets outside the capture day, including unset
startup clocks, are omitted. All 278 checksum-invalid candidates are rejected.
Summary cards, tables, gauges, state blocks, and the main histogram show the final recorded packet and
do **not** follow the time picker. A separately dated histogram example shows
the most recent earlier packet with nonzero bins (23:11:07.850 spacecraft time);
the final packet's bins are all zero. This preview does not validate SatNOGS
ingestion, observation timestamps, freshness, or RF reception.

`build_local_preview.py` regenerates the snapshot from captures without
modifying them or making network requests:

```shell
conda activate suncet
python -m suncet_processing_pipeline.satnogs.dashboard.build_local_preview \
  /path/to/2026-09-23_cpt/uhf_rundirs \
  --date 2026-09-23 --output tmp/suncet-grafana-preview \
  --image /path/to/spacecraft-render.png
```

The optional spacecraft image is embedded in the local snapshot. The
`preview_panels.py` module adds native Grafana visualizations: voltage gauges
alongside the time series; all-state mode blocks with a highlighted current
choice; five EPS power-enable blocks (including ADCS); and vertical CSIE bins.
The histogram plots transmitted counts directly using a Bar chart, avoiding a
second histogram calculation on already binned values. Its DN labels assume
offset 0 and width 32. Axes read **EUV image intensity [DN]** and **Pixel count
(3,008,000 total)**; the full image is 1504 × 2000 pixels, and the six
transmitted bins need not sum to that total. Counts are not normalized.

The mission owner selected gauge display bands on 2026-09-29: red below
13.2 V, yellow from 13.2 V to 14 V, and green at 14 V and above, for both
batteries and the EPS bus. The display scale is 12–16 V so the threshold
labels remain readable; these are visual
bands, with no alert rules configured.

Mode and UHF-health blocks use the built-in Canvas panel so the background
and text colors can be bound separately to the selected-state indicators.
Inactive labels are dimmed. CDH choices are ordered PHOENIX, SAFE, DOWNLINK,
SCIENCE; their active colors are red, yellow, green, green. ADCS mode highlights SUN_POINT yellow and
FINE_REF_POINT green. Sun-pointing choices are ordered FINE_REF_POINT, ON_SUN,
SUN_POINT, CONVERGING, SEARCHING, SEARCH_INIT, NOT_ACTV, WAITING: the first two
are green, the next three yellow, and the remaining three red when selected.
WAITING was retained because it is a valid reported state. UHF highlights
ALIVE green and OFF/DEAD red. These colors follow mission-owner display
preferences. Each indicator still reflects the fixed final preview packet.

The Sun pointing error plot defaults to a 0–10° vertical axis. Its table legend
shows the actual maximum within the selected time range, including excursions
above 10°. This changes the visible scale only; original samples are retained.

The snapshot, saved export, audit report, Compose file, and local admin
credentials are in ignored `tmp/suncet-grafana-preview/`. Keep recorded values
and credentials out of commits. Rebuilding files does not update Grafana;
export any user layout edits before replacing its saved dashboard. The local
UID is `suncet-recorded-local`, separate from the hosted SunCET UID.

## Before enabling queries or promoting to Telemetry

1. Confirm the actual SunCET measurement, field aliases/types, engineering-value
   conversion stage, station/source metadata, and source timestamp meaning in
   the hosted datasource. Their source-level contracts now pass locally.
2. Validate the first CTDB 2.0.5 recorded packet through checksum validation,
   ingestion, and displayed values. Resolve non-finite values (including Inf),
   unknown codes, and source failures without substituting zero or healthy states.
3. Implement reception timestamp/age and reviewed freshness handling, distinguish
   source failure from empty data, and validate packet-coherent summaries.
4. Check queries against actual numeric/string field types, timestamp ordering,
   missing fields, and observation gaps; bound query volume for normal ranges.
5. Review operational limits and spacecraft-time display policy. Only then enable
   verified targets and move the dashboard into the public Telemetry workflow.

The remaining end-to-end and edge-case acceptance checks are in
[`docs/SATNOGS_DASHBOARD_PLAN.md`](../../../docs/SATNOGS_DASHBOARD_PLAN.md).
The [2026-09-23 Hydra audit](../../../docs/SATNOGS_UHF_VALIDATION_20260923.md)
passes the local recorded-packet decoder comparison. The October 5 DB-function
integration also passes. Hosted ingestion and display validation remain open;
some recorded spacecraft times appear to precede
clock setting and must not be substituted for reception time.
