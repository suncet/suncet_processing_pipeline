# One-way public-data copies with rclone

## Safety boundary

Dropbox is an exchange/publication path for selected public SunCET data. It is
not the processing queue, the only archive, or the authority for an incomplete
product. The initial workflow uses `rclone copy`, never `sync` or `bisync`, so a
deletion on either system is not propagated to the other.

The repository wrapper adds four further controls:

- every invocation is a dry run unless `--execute` is explicit;
- local paths must remain below the declared `suncet_data` root, which keeps the
  sibling private `suncet_ctdb` tree out of scope;
- the task file, token-bearing rclone configuration, filter, publication
  manifest, and operational state must all remain outside `suncet_data`;
- `--immutable` refuses to replace different content at an existing path;
- pulls compare size and content checksum instead of modification time, so an
  independently installed identical CSV is accepted while a same-name content
  change still fails closed;
- filters are copied into a private invocation snapshot before rclone starts;
- every executed copy is followed by `rclone check --one-way`, with a private
  log and mode-`0600` JSON receipt below
  `$HOME/.local/state/suncet/rclone/` by default.

A push has an additional gate: the operator first freezes and reviews a private
publication manifest containing the exact relative path, size, and SHA-256 of
every allowed source file. The wrapper verifies that exact tree before and
after transfer and gives rclone an invocation-local `--files-from-raw` list.
This prevents an unreviewed file added elsewhere in a broad directory from
joining the transfer. Stop writers and place products atomically in their final
publication directories before freezing the manifest; a changed source makes
the operation fail and requires a new manifest.

The reviewed filters are:

- `operations/rclone/pull_metadata.filter`: top-level, explicitly versioned
  FITS and NetCDF/Zarr metadata CSV exports from Dropbox to the SOC;
- `operations/rclone/push_products.filter`: finalized Levels 0.5–4, trends, and
  processing manifests from the SOC to Dropbox.

The pull filter deliberately excludes synthetic data, test data, obsolete
definitions, example files, workbooks, and mutable unversioned files. The push
filter deliberately excludes raw Level 0, processing-run inputs,
scratch, staging, transfer logs, and private definitions. Expand either filter
only after explicit review.

## Scheduled metadata and completion-triggered products

The operator authorized automatic Dropbox copies, including Level 0.5, on
2026-11-06. This replaces per-run manual approval for the installed automatic
workflow; the manual dry-run/manifest commands below remain available.
Dropbox copies do not publish anything to the LASP website.

- `suncet-dropbox-metadata.timer` pulls eligible versioned metadata at
  **00:00 UTC**, with `Persistent=true`. Downloading a new metadata definition
  does not activate it in processing configuration.
- `ProcessingRunProvenance` checks the private task configuration's
  `[automation] enabled = true` option. On successful completion it freezes
  changed products from the approved `level0_5/`, `level1/` through `level4/`,
  and `trends/` roots. Failed runs and no-op runs enqueue nothing. Only files
  under those standard product directories participate; standalone writers
  that do not use run provenance must use `processing_completion` explicitly.
- Snapshots are private copies under `$suncet_data/.dropbox-outbox/<id>/`.
  A minimal completion manifest accompanies the products; full private run
  provenance, raw input paths, invocation/configuration, and CTDB details are
  not automatically published. The snapshot is hashed and an exact manifest
  is frozen before a pending record is written atomically.
- `suncet-dropbox-products.path` starts the worker when the pending directory
  changes. A separate 15-minute recovery timer retries queued uploads after
  a failure or reboot; an empty queue makes no Dropbox requests. Transfers
  preserve the usual relative product paths. The worker retains the existing
  immutable-copy, content-verification, and exact-manifest checks. Different
  content at an existing remote path is a conflict, not an overwrite.
- After verified delivery the pending record is archived locally and its
  staging copy is removed. Original products remain. Interrupted/failed
  transfers retain their snapshot for retry. Participating processing runs
  are serialized while computing the changed-product set; a concurrent run
  fails visibly instead of mixing output ownership.

The Jetson uses an isolated Dropbox worker runtime. Its active `suncet` conda
environment imports the manual processing checkout; that checkout received
only the completion hook and the new automation module, with a checksum-guarded
backup. Older separately pinned processing releases do not acquire the hook
automatically. Deployment records remain private. Worker services run storage
preflight before copying. Queue records, frozen manifests, and copy receipts
are under `~/.local/state/suncet/rclone/`, outside the data tree.

Level 0 is excluded: AWS replicates LASP deliveries to the raw archive, and
the SatNOGS collector immediately archives its receipts/frames. Keep local raw
inputs for processing and verify archival before any later local cleanup.
The 2026-11-06 audit found all 11 UHF source objects marked `COMPLETED` and all
60 collected SatNOGS receipts recorded as archived; this was not a Glacier
restore test. The remaining 29-byte X-band setup fixture was preserved under
`level0/unknown_ground_station/xband/`, and the empty legacy `telemetry/` tree
was removed.

```sh
systemctl --user list-timers 'suncet-dropbox-*'
systemctl --user status suncet-dropbox-products.path
systemctl --user start suncet-dropbox-products.service  # retry pending copies
journalctl --user -u suncet-dropbox-metadata.service -u suncet-dropbox-products.service
```

To stop automatic copies, disable the metadata and products timers and the
products path unit. Set `[automation] enabled = false` in the private task
configuration as well to stop new processing runs from generating snapshots.
Do not remove pending snapshots or stale operation locks without checking
whether a writer or transfer is still active.

Deployment validation on 2026-11-06: 26 targeted tests passed in a clean source
export. The real metadata service copied the `v1.0.3dev` and `v1.0.4dev`
NetCDF/Zarr definition exports and passed the one-way check, without changing
the active processing definition. A synthetic successful run created
`level0_5/_tests/dropbox_automation_20261106/TEST_completion_trigger.json`;
the path unit automatically uploaded it and its minimal completion manifest.
Both copy and verification returned zero, the queued snapshot was retired,
and independent Dropbox read-back matched the complete source bytes. This
test contains no spacecraft telemetry and measures transfer correctness,
not large-product throughput.

## One-time host setup

The checksum-verified native ARM64 `rclone` 1.75.0 binary is installed on the
Jetson at `/home/james/.local/bin/rclone`. The initial deployment uses the
project owner's paid Dropbox account because a separate free account cannot
hold the shared data tree. This is an explicitly accepted compromise: the
Dropbox token has broader account access than the task filters. Keep it mode
`0600`, never print or copy it, retain SSH-only administration, and revoke the
connected app promptly if the Jetson or credential may be compromised. A
dedicated paid/team identity remains the preferred long-term boundary.

Run `rclone config` interactively. Its credential-bearing configuration is
normally `$HOME/.config/rclone/rclone.conf`; set mode `0600`. Do not copy it,
its token, or its browser callback into Git, a processing manifest, or a support
ticket.

Copy `operations/rclone/rclone_public.example.ini` to
`$HOME/.config/suncet/rclone_public.ini`, set mode `0600`, and replace the
remote root and repository path with their reviewed host-local values:

```ini
[rclone]
config = /home/james/.config/rclone/rclone.conf
executable = /home/james/.local/bin/rclone
transfers = 4
checkers = 8
timeout_seconds = 21600
state_directory = /home/james/.local/state/suncet/rclone

[task:pull-metadata]
direction = pull
remote = SUNCET_PUBLIC_REMOTE:PUBLIC_DATA_ROOT/metadata
local = metadata
filter_file = /home/james/src/suncet_processing_pipeline/operations/rclone/pull_metadata.filter

[task:push-products]
direction = push
remote = SUNCET_PUBLIC_REMOTE:PUBLIC_DATA_ROOT
local = .
filter_file = /home/james/src/suncet_processing_pipeline/operations/rclone/push_products.filter
```

Neither this INI, the rclone credential file, the tracked filter, private
publication manifests, nor operational state belongs below `suncet_data`.
The wrapper enforces that boundary and refuses either private configuration
file when group or other users can read it.

## Metadata release discipline

The live Google Sheet is the review/development authority, but is never read by
the Jetson processing pipeline or by the host setup scripts. After review,
export both schema tabs as CSVs with a new SemVer-style mission version in each
filename. Development versions use `MAJOR.MINOR.PATCHdev`; approved major
releases use `MAJOR.MINOR.PATCH` without `dev`. Never change the content of a
previously published filename.

Dropbox distributes these immutable exports to development systems. The
Jetson, which has no Dropbox desktop client, receives them only through the
explicit `pull-metadata` rclone task. Its remote is rooted directly at the
Dropbox `metadata` directory and its local destination is rooted directly at
`$suncet_data/metadata`; its strict filter then admits only the stable and
development FITS and NetCDF/Zarr filename families. This defense in depth keeps
synthetic and test trees outside even the task's remote namespace. The task
copies eligible versions but does not activate a new definition. Activation
remains a reviewed code and configuration change, and each processing run takes
its own checksum-guarded metadata snapshot. Obsolete definitions, examples,
workbooks, synthetic data, and test data never participate in the Jetson pull.

## Manual pull and push

Before a pull, run the storage preflight in the operator runbook. Preview the
reviewed task:

```sh
python -m suncet_processing_pipeline.rclone_public_data pull-metadata
```

Read the complete rclone log and confirm that every proposed path matches the
pull filter. Then repeat the identical named task explicitly:

```sh
python -m suncet_processing_pipeline.rclone_public_data \
  pull-metadata --execute
```

For finalized products, stop their writers and atomically place the complete,
reviewed products under the included Level 1–4 or trends subtree. Freeze the
exact allowlist into a timestamped private manifest and inspect its file list,
sizes, hashes, filter hash, and tree hash:

```sh
manifest="$HOME/.local/state/suncet/publication_manifests/$(date -u +%Y%m%dT%H%M%SZ)_push-products.json"
install -d -m 700 "$(dirname "$manifest")"
python -m suncet_processing_pipeline.rclone_public_data \
  push-products --create-manifest "$manifest"
less "$manifest"
```

Use that same reviewed manifest for the dry run and execution:

```sh
python -m suncet_processing_pipeline.rclone_public_data \
  push-products --manifest "$manifest"
python -m suncet_processing_pipeline.rclone_public_data \
  push-products --manifest "$manifest" --execute
```

An execution succeeds only when both `copy` and the subsequent one-way check
return success. Retain the JSON receipt with the processing/publication record.
The receipt records the invocation filter snapshot, frozen publication tree,
pre/post stability checks, exact commands, return codes, and failures. A source
or manifest change invalidates the attempt even when rclone itself returned
success.

## Conflicts and recovery

- A differing file at the same destination path is an error by design. Give a
  corrected product a reviewed revision/version name; do not weaken
  `--immutable` during an ordinary retry.
- A failed or interrupted operation may be repeated. `copy` does not delete
  destination-only content, and the verification pass reports anything still
  missing or different.
- Do not reuse a publication manifest after any included file, allowlist root,
  or filter changes. Freeze and review a new timestamped manifest.
- Never point ordinary `rclone` commands directly at the complete data root to
  bypass a filter. Never use `sync`, `bisync`, `delete`, or `purge` in this
  workflow.
- Keep scheduled jobs disabled until manual copies, alerting, conflict
  ownership, credential rotation, and recovery are accepted operationally.
- The wrapper permits only one public-data operation at a time. If an abrupt
  host failure leaves `.operation.lock` in the private state directory, first
  prove that no rclone/publication process remains and preserve the failed run
  receipt before removing that one exact lock file.
