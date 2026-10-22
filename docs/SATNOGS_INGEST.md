# Daily SatNOGS telemetry collection

The SOC collector downloads SunCET telemetry from SatNOGS DB once daily. The
public SatNOGS dashboard remains the timely monitoring view. Collection is
outbound HTTPS only; no incoming Jetson service or webhook is needed. After
each API page is durably saved, it immediately archives pending SatNOGS frames
and reception receipts to private S3 Glacier Deep Archive.

## Status (2026-10-22)

- Immediate private S3 archival is deployed and enabled in the pinned runtime
  `/home/james/.local/share/suncet/satnogs/runtime-75af2455b945` (Python 3.14.7,
  Kaitai runtime 0.11). Its wheel SHA-256 is
  `75af2455b94562ef8e9213d93f3eacea17f01f613b8e02ddfdba934b21f8ca0c`.
  The processing release and UHF collector runtime are unchanged.
- All 170 targeted archive, collector, decoder and UHF collector tests passed
  in the local `suncet` conda environment. Seven IAM simulations verified the
  intended scope, including denial of writes outside `satnogs/`, other source
  IPs, unconditional writes, non-Deep-Archive storage and deletions. The applied
  policy was read back and matched the explicitly approved policy exactly.
- An isolated synthetic test on the Jetson uploaded one frame binary and its
  reception manifest under a clearly labeled `satnogs/_test/` prefix. Both
  objects passed full-object SHA-256, size, Deep Archive and version checks.
  Repeating collection archived zero additional receipts. A conditional retry
  against the existing frame verified it successfully without overwriting it.
  Synthetic receipts remain in a separate test database; no test frame was
  submitted to SatNOGS or inserted into the operational collection database.
- A real authenticated systemd service run succeeded with archive enabled at
  `2026-10-22T18:05:13.881765Z`: zero production receipts and zero pending archive
  transfers. The production telemetry API still has no SunCET frames. Deck has
  been asked about an approved test upload; live frame retrieval/decoding and
  end-to-end archival of an actual SatNOGS reception remain to be verified when
  data becomes available. No archive restore was requested or tested.
- Both collection timers remain enabled at midnight UTC, without randomized
  delay. The next run is `2026-10-23T00:00:00Z`. User lingering is enabled, and
  the private NVMe collection directory is mode 0700. The UHF collector remains
  download-only and relies on AWS replication for its archive copies.

## Data boundary

The collector uses `GET https://db.satnogs.org/api/telemetry/` filtered by stable
satellite ID `MNRC-9829-4319-5529-8975`, `start`, and `end`. DB provides raw frames
and reception metadata; its `decoded` field may be an InfluxDB marker rather
than engineering values. The collector decodes locally with the reviewed CTDB
2.0.5 public decoder. It accepts the tested 268-byte AX.25 representation without
FCS, validates the CCSDS/Fletcher contract, and exposes only the 111 public fields.
Other boundaries, other APIDs, and invalid checksums are retained as rejected
receipts without decoded values. It does not guess a framing boundary.

All retrievals are private. Exact JSON response bytes, request URLs, receipt
metadata, and original frame hex are stored in SQLite under
`/srv/suncet/satnogs-private/telemetry.sqlite3`, outside the public data tree.
Raw beacon bytes can include opaque fields excluded from public decoding.
Never publish this directory or include it in public product mirroring.

Tables:

- `pages`: SHA-256 of exact response bytes, response body, retrieval time, URL.
- `receipts`: SHA-256 of canonical full API record, original JSON, raw frame
  SHA-256, retrieval time, decoded public JSON or rejection, decoder source hash.
- `state`: last complete collection window end and last full reconciliation.
- `archive_objects`: verified bucket/key, SHA-256, size, version ID and verification time.
- `archived_receipts`: completed receipt digests for each destination bucket/prefix.

Identical API records are idempotent. Different station receptions and changed
upstream metadata remain separate receipts. This is a provenance archive, not
a count of unique spacecraft packets. Consumers should use frame hashes to
combine repeated receptions while preserving their provenance. Existing
receipts retain their original decoder revision; a future decoder upgrade needs
an explicit reprocessing migration.

No upload to SatNOGS, remote deletion, IQ download, main telemetry-store merge,
or downstream science processing is performed. The private S3 copy is made only
from this collector's own SatNOGS receipt database. It never scans the UHF S3
download tree or copies UHF deliveries into the archive; AWS replication already
handles those deliveries.

## Immediate Deep Archive custody

The host-local `[archive]` section enables archival to a configured bucket under
`satnogs/`. Each distinct raw frame is stored unchanged as
`satnogs/frames/<frame SHA-256>.bin`; these are the original DB frame bytes,
including AX.25 framing where present, not silently stripped CCSDS packets.
Each reception has a separate `satnogs/receipts/<record SHA-256>.json` containing
the original API record, source/station/observation metadata as supplied by DB,
retrieval time, frame hash/key, decoded public values or rejection, and decoder
revision. Identical bytes from multiple stations share one binary while retaining
each reception. Invalid-but-hex frames are also preserved privately; malformed
hex remains in its JSON receipt without an invented binary. Exact API page bytes
remain in the local SQLite archive.

Uploads use `DEEP_ARCHIVE`, full-object SHA-256, and conditional creation
(`If-None-Match: *`). A subsequent HEAD verifies checksum, byte count, storage
class and non-null version ID before committing the local transfer record.
Retries after an interrupted local commit verify the existing object instead of
overwriting it. Previously verified objects are skipped from local transfer
receipts; ongoing independent cloud inventory/restore auditing is separate.

Pending receipts are retried before the next API request, including receipts
retrieved before archival was enabled. Each new page is archived before fetching
the following page. An upload failure leaves the local data intact, returns a
failure, and prevents advancing the daily checkpoint. The next scheduled or
manual run resumes. Successful empty DB pulls do not test archive-write access.
`status` includes `archive_enabled`, `archived_receipts`, and `archive_pending`
once the database exists. A configuration without `[archive]` explicitly remains
local-only for development or rollback.

The permission template is
[`operations/satnogs_archive_iam.example.json`](../operations/satnogs_archive_iam.example.json).
It grants only conditional Deep Archive creation and checksum-read access under
`satnogs/*`, restricted to the existing SOC egress IP. It grants no deletions,
restores or writes to other prefixes. The literal `*` with `StringEquals` on
`s3:if-none-match` is intentional, not a wildcard policy match. Keep actual
bucket/account identifiers in private configuration. No administrator credentials
are installed on the Jetson. The existing archive uses SSE-S3; a future change to
KMS needs a separate permission review.

## Daily behavior and recovery

The timer runs at midnight (00:00 UTC), with no randomized delay and a
one-second scheduling accuracy setting. `Persistent=true`
catches a missed activation when the user's systemd manager next starts. Runs
are locked so manual and scheduled collection cannot overlap.

The first run starts at 2026-10-01 UTC. Ordinary runs start seven days before
the last successful window end, including any outage interval. Every 30 days,
the collector scans from the configured initial date to catch older delayed
uploads. There is no claim that an overlap alone catches arbitrarily late data.
`--full` triggers this scan manually. Adjust `initial_start` if an approved test
frame uses an earlier reception timestamp.

Each page commits durably before the next download. A failed or interrupted run
retains committed receipts, but never advances the checkpoint. Replaying the
window safely deduplicates those receipts. A run succeeds only after pending
archive transfers finish. Empty results are a successful pull.
An explicit `--since` pull does not advance the daily checkpoint.

Requests are spaced at least 11 seconds apart, with a 30-second timeout and at
most three attempts for network errors or selected HTTP 5xx failures. HTTP
401/403/429 stops the run immediately; there is no immediate automatic restart.
Inspect authentication or throttling before manually retrying. Redirects and
pagination URLs outside the exact production telemetry endpoint are refused.
Limits are 4 MiB per page, 1,000 pages per run, and four hours per service run.
The page budget may need review as the mission archive grows; hitting a limit
fails visibly with the checkpoint unchanged rather than silently losing data.

The collector refuses a missing data mount, less than the configured free-space
reserve (150 GiB), or a non-private collection directory. Failures return exit
code 1 and appear in the systemd journal. There is no email/pager integration
yet; check timer failures and `last_success` during routine operations.

## Jetson installation

For a new installation, deploy a reviewed wheel containing this module into a pinned runtime. Do not
modify the existing release environment in place or point the service at a
mutable development checkout. Installation below assumes that runtime already
exists; replace `REVIEWED_RUNTIME` with its absolute prefix.

```sh
suncet_checkout=/home/james/src/suncet_processing_pipeline
install -d -m 700 "$HOME/.config/suncet" "$HOME/.local/libexec"
install -m 600 "$suncet_checkout/operations/satnogs.example.ini" "$HOME/.config/suncet/satnogs.ini"
cat > "$HOME/.local/libexec/suncet-satnogs-python" <<'SH'
#!/bin/sh
exec REVIEWED_RUNTIME/bin/python "$@"
SH
chmod 700 "$HOME/.local/libexec/suncet-satnogs-python"
```

Review paths and the storage reserve in the configuration. Store the existing
SatNOGS DB API token in `$HOME/.config/suncet/satnogs.token` (one line, mode 0600,
owned by the service user). Enter it using a local editor or a hidden prompt;
never put it in Git, chat, a command-line argument, or logs. The collector only
uses GET, although the service's token itself may not be read-only scoped.

The current collector-only runtime installs the hashed pipeline wheel with
`--no-deps` plus the pinned `kaitaistruct==0.11` wheel. It is not a replacement
for the full science-processing environment. Keep its dependency wheels and
SHA-256 manifest with the deployment record. The installed runtime selector is
a shell wrapper that executes its absolute Python path.

Run from outside the source checkout to verify the installed package:

```sh
cd "$HOME"
"$HOME/.local/libexec/suncet-satnogs-python" -m suncet_processing_pipeline.satnogs.collect sync
"$HOME/.local/libexec/suncet-satnogs-python" -m suncet_processing_pipeline.satnogs.collect status
```

After a successful authenticated pull, install the user units from this repo:

```sh
install -d "$HOME/.config/systemd/user"
install -m 644 "$suncet_checkout/operations/systemd/suncet-satnogs.service" "$suncet_checkout/operations/systemd/suncet-satnogs.timer" "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now suncet-satnogs.timer
systemctl --user list-timers suncet-satnogs.timer
```

For collection while logged out or after reboot, ensure the operator's user
manager has lingering enabled (`loginctl show-user "$USER" -p Linger`). If it
does not, an administrator can enable it with `sudo loginctl enable-linger james`.
Verify this rather than assuming a user timer runs without a login.

## Manual sync and monitoring

```sh
systemctl --user start suncet-satnogs.service
journalctl --user -u suncet-satnogs.service --since yesterday
"$HOME/.local/libexec/suncet-satnogs-python" -m suncet_processing_pipeline.satnogs.collect status
```

Use `sync --full` for a full reconciliation or
`sync --since 2026-10-01T00:00:00Z` for a one-off historical pull. A nonzero
`rejected` count warrants inspecting the private receipt archive before using
its telemetry. A successful pull does not imply any valid beacon was received.
Disable scheduling with `systemctl --user disable --now suncet-satnogs.timer`;
this preserves all local receipts and upstream data.

## References

- [API authentication](https://community.libre.space/t/satnogs-db-telemetry-api-endpoint/5341)
- [Time-window filtering](https://community.libre.space/t/question-about-api-telemetry/6938)
- [Maintainer rate-limit guidance](https://community.libre.space/t/we-were-polling-the-db-api-badly-fixed-and-asking-what-rate-is-acceptable/15212)
- [Test uploads belong in the development service unless otherwise coordinated](https://community.libre.space/t/satnogs-db-post-request-telemetry-api/5897)
