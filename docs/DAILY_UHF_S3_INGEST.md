# Daily UHF S3 collection

The UHF collector uses the existing `[aws]` and `[uhf]` sections in the Jetson's
private `~/.config/suncet/aws_ingest.ini`. Bucket/account identifiers and AWS
credentials remain host-local. It collects the configured `uhf/` delivery
prefix once daily at **00:00 UTC**, with no randomized delay. It does not ingest
X-band, process telemetry into science products, publish files, or alter S3.

## Collection and recovery

`python -m suncet_processing_pipeline.sync_s3 sync` enumerates every available
object version under the delivery prefix, including noncurrent versions.
Explicit pagination follows both S3 key and version markers. Delete markers
never delete local files. Zero-byte folder markers are ignored. Full daily
enumeration avoids timestamp-watermark gaps and catches deliveries after an
outage, provided those versions still exist in the delivery bucket. Recovery
after lifecycle expiration requires the separate archive restore workflow.

New versions use the existing atomic ingest implementation. Each GET specifies
the exact version and an ETag precondition; returned size, ETag, and version
must match the listing before local finalization. The complete downloaded file
is hashed with SHA-256. Each receipt records bucket, key, version, S3 checksum
metadata, local digest, and local path in a private mode-0600 JSON file.

Local files are stored below the configured UHF destination as
`objects/<identity SHA-256>/<original basename>`. The identity includes bucket,
full key, version ID, ETag, modification time, and size, so different keys with
the same basename, new versions, and replacements of a `null` version do not
overwrite each other. The existing single-object manual ingest layout is
unchanged. These are raw inputs, not public distribution products.

The private index and receipts live in `~/.local/state/suncet/aws_uhf_sync/`.
On repeat collection, the local file's size and complete SHA-256 are checked
against its receipt, without downloading it again. Missing files are recovered;
different existing content causes a visible conflict instead of an overwrite.
Each object is committed independently. A partial failure preserves completed
downloads but never updates `last_success`. A restarted run deduplicates those
objects and continues. A lock prevents overlapping collector runs.

## Limits and credentials

Before listing and each download, the existing SOC storage preflight checks
the configured mounts, volume identity, byte/inode reserves, and planned work
size. Both warnings and critical results stop unattended ingest for review.
The workload target must match `suncet_data`.

Defaults are 1,000 listing pages, 50 GiB downloaded per run, three AWS standard
retry attempts, and an eight-hour systemd timeout. A limit hit is a failure,
not a successful partial inventory. Already completed downloads remain, so a
later run can continue through the backlog. A single object exceeding 50 GiB
requires an operator to review and raise the limit. No automatic immediate
service restart is configured.

The collector performs only read operations using the existing source-IP-restricted
AWS ingest identity. That identity also has a separately approved, narrow
`satnogs/` archive-write permission for the SatNOGS collector; this UHF collector
never invokes it and has no write permission to UHF archive keys. The
current setup retains the existing host credential mechanism; migration to
short-lived workload credentials remains an operations improvement. AWS
replication independently preserves delivery objects in the raw archive.

## Operation

Installed service: `suncet-uhf-s3.service`; timer: `suncet-uhf-s3.timer`.
Its wrapper selects a separate pinned Python runtime, leaving the manual
processing release and SatNOGS collector runtimes unchanged. It uses the
existing user-local AWS CLI. The unit supplies `suncet_data=/srv/suncet/data`
and `suncet_ctdb=/srv/suncet/ctdb` explicitly. User lingering and
`Persistent=true` allow logged-out operation and recovery of missed timer runs.

On the Jetson:

```sh
systemctl --user start suncet-uhf-s3.service
systemctl --user list-timers suncet-uhf-s3.timer
journalctl --user -u suncet-uhf-s3.service --since yesterday
~/.local/libexec/suncet-uhf-s3-python -m suncet_processing_pipeline.sync_s3 status
```

For a direct CLI sync outside systemd, first export the two managed-root
environment variables. Private AWS error details are retained in
`~/.local/state/suncet/aws_uhf_sync/last_failure.json`; the journal contains only
the generic failure message and aggregate success counts. That diagnostic may
refer to an earlier failure after a successful retry; compare its timestamp
with `status`. No pager/email alert integration is currently installed.

Disable with `systemctl --user disable --now suncet-uhf-s3.timer`. This removes
neither local data nor S3 objects. Keep the source receipt database and backing
files together when migrating the collector.

## Test-data boundary

The 2026-10-22 integration test uses a 2,520-byte binary containing the ten
validated 252-byte CCSDS APID 1 packets recovered from the 2026-09-30 flight-unit
ground recording. AX.25 headers and FCS bytes were removed only after validating
both the 268-byte and 270-byte recovered representations. No other APIDs or IQ
samples are included. This is actual ground-test telemetry, not a synthetic
decoder fixture, and it remains private.

Its SHA-256 is
`3d423700cf4ee36e1ffe903e31cb294ffbcb5e310b95c959f02fdaeb9f85d57c`.
Test keys use an explicit `_test/` prefix and `TEST_` filename. Exact cloud keys,
versions, and upload/verification receipts
belong in the private deployment record, not the public repository. Test copies
must not be represented as operational flight receptions or uploaded to the
public SatNOGS DB without the separate agreed test procedure.

## Validation status

Implementation regression tests cover pagination, version identity, no-op
repeats, recovery of missing files, conflicting local content, interrupted runs,
storage failures, and transfer limits. The collector, manual ingest, replication
monitor, and storage-preflight regression suite passed all 39 tests.

On 2026-10-22 the collector was installed on the Jetson in the isolated runtime
`~/.local/share/suncet/uhf-s3/runtime-28720f1e9a99`. Its wheel SHA-256 is
`28720f1e9a997b476468003c4801392f8346b6bd79d556f1a317c2e5b7ecde32`.
The actual systemd service passed storage checks and authenticated listing of
the empty delivery prefix. The timer is enabled and active, with its next run
at 2026-10-23 00:00 UTC. User lingering is enabled for logged-out execution.

The real APID 1 test binary was then uploaded to the delivery test prefix using
the operator's existing AWS console session. The systemd collector downloaded
one version (2,520 bytes), and the Jetson's independently calculated SHA-256
matched the original. A second service run reported one already-present version,
zero downloads, and zero transferred bytes.

A separate direct upload to the archive's test prefix succeeded with
`DEEP_ARCHIVE` storage class and a server-verified full-object SHA-256 matching
the original. The existing delivery-bucket replication also completed; its
separate archive replica reports `REPLICA`, `DEEP_ARCHIVE`, and the same full-object
CRC64NVME checksum as the source. These checks verify archive upload and
replication, not an archive restore. The test uploads used the operator's console
session. The daily UHF collector remains download-only; the later SatNOGS
archive permission is restricted to the separate `satnogs/` prefix.
