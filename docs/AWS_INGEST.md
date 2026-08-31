# AWS Delivery Ingest

## Scope and security boundary

The SOC Jetson pulls raw X-band and UHF pass files from two LASP-owned AWS S3
delivery buckets. AWS replication independently preserves each new object in
deep archive, and bucket lifecycle rules expire ordinary delivery copies. The
Jetson must never upload, alter, or delete an S3 object.

The repository contains no AWS account number, bucket name, access key, or
secret. The Jetson identity has no console password and is restricted to:

- locate and list the two delivery buckets;
- read objects and object versions from those buckets;
- requests originating at the approved SOC public IPv4 address.

It has no access to the archive bucket, no account-wide bucket listing, and no
write or delete action. The source-IP condition is defense in depth: if the SOC
egress address changes, both this policy and any external network allowlist
must be reviewed. A long-lived access key is acceptable for the initial manual
workflow, but replace it with short-lived external-workload credentials before
unattended production ingest.

## Host-local configuration

The AWS CLI profile is stored in the normal mode-`0600` files below
`$HOME/.aws/`. Its exact values must not be copied into an issue, log, or Git.
The installed user-local AWS CLI must be on `PATH`, or pass its absolute path
with `--aws-cli`.

Create `$HOME/.config/suncet/aws_ingest.ini` with mode `0600`:

```ini
[aws]
profile = suncet-ingest
region = APPROVED_AWS_REGION

[xband]
bucket = X_BAND_DELIVERY_BUCKET
prefix =
destination = telemetry/incoming/xband

[uhf]
bucket = UHF_DELIVERY_BUCKET
prefix = uhf/
destination = telemetry/incoming/uhf
```

The UHF prefix must match the actual LASP uploader policy. An empty prefix is
valid only when the identity is authorized to list and read the entire delivery
bucket.

## Manual procedure

First confirm the source and review its object key, timestamp, and size:

```sh
python -m suncet_processing_pipeline.ingest_s3 list xband
```

Preview one exact pull. This does not contact S3 or write a file:

```sh
python -m suncet_processing_pipeline.ingest_s3 pull xband OBJECT_KEY
```

Then execute the reviewed pull:

```sh
python -m suncet_processing_pipeline.ingest_s3 \
  pull xband OBJECT_KEY --execute
```

Use `uhf` in place of `xband` for LASP ground-network delivery. Add
`--version-id VERSION_ID` when recovering a specific S3 version.

For every execution, the command:

1. rejects keys outside the configured source prefix;
2. downloads to a process-specific `.partial` file on the NVMe data volume;
3. verifies the reported content length when S3 supplies it;
4. calculates a complete SHA-256 digest;
5. atomically renames new content into its final location;
6. treats identical existing content as idempotent and refuses to overwrite
   different content with the same filename;
7. records source, key, version, ETag, S3 checksums, size, local path, and
   SHA-256 in a private mode-`0600` receipt below
   `$HOME/.local/state/suncet/aws_ingest/` by default.

The command never deletes the source object. S3 lifecycle remains the only
delivery-deletion authority.

Keep receipts outside `suncet_data`: they contain exact private bucket and key
identifiers that do not belong in the public data tree. An alternate private
location can be selected with `--state-directory`; the command refuses a state
directory at or below `suncet_data`.

Before relying on lifecycle cleanup, inspect every current and noncurrent
source version in the full lifecycle risk window:

```sh
python -m suncet_processing_pipeline.aws_replication_monitor \
  xband uhf --pending-hours 24 --retention-days 37
```

This deliberately has no recent-object count limit. A failed status, stale
pending/unknown status, or truncated version listing fails closed. Source-side
status monitoring is still not a substitute for an independent destination
inventory and restore drill.

## Archive custody status and future evolution

The source-neutral raw archive cutover was completed on 2026-08-31. Because S3
bucket names cannot be renamed, a replacement bucket was created with
versioning, default server-side encryption, blocked public access,
bucket-owner-enforced object ownership, and a lifecycle transition to
`DEEP_ARCHIVE`. Both delivery-bucket replication rules were switched to the
replacement and independently validated with live objects: each source reached
`COMPLETED`, and each destination object reported `REPLICA` in
`DEEP_ARCHIVE`. The replication roles and the LASP software principal no longer
have access to the legacy archive.

A post-cutover read-only audit confirmed these controls. Verification of the
first 30-day source expirations and an independent archive inventory/restore
drill remain pending.

The small historical inventory remains in the legacy bucket under read-only
retention. Restoring and copying already archived objects would add delay and
cost without improving custody, so the legacy bucket must not be deleted unless
a later reviewed migration proves a complete inventory and checksum-equivalent
replacement. Exact resource names remain in the private AWS inventory.

The remaining archive evolution task is:

- Add an idempotent SatNOGS archival path after operational retrieval is
  designed. Resolve the real source boundary first: SatNOGS DB primarily holds
  satellite/transmitter metadata, while observation artifacts are associated
  with SatNOGS Network. Store selected raw frames, decoded public APID 1 data,
  observation/station metadata, and any retained RF products under a dedicated
  `satnogs/` namespace with source IDs, decoder/public-specification revisions,
  checksums, and transfer receipts. This archival copy must never modify or
  delete the upstream SatNOGS record.

Exact AWS resource names remain in the private host/account inventory and must
not be added to this public repository.

## Operational checklist

- Confirm `curl -4 https://checkip.amazonaws.com` still reports the approved
  SOC egress address before diagnosing an access failure.
- Keep the credential and config files at mode `0600`.
- Review the list output and pull one exact object; do not mirror an entire
  bucket as an ad hoc shortcut.
- Preserve ingest receipts with the corresponding processing run.
- Rotate the access key after suspected exposure and on the approved operations
  schedule; never create a second key as an undocumented fallback.
- Before scheduling ingest, define polling, duplicate/version handling,
  backpressure, disk thresholds, alerting, credential rotation, and migration
  to temporary credentials.
