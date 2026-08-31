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
- `operations/rclone/push_products.filter`: finalized Levels 1–4, trends, and
  processing manifests from the SOC to Dropbox.

The pull filter deliberately excludes synthetic data, test data, obsolete
definitions, example files, workbooks, and mutable unversioned files. The push
filter deliberately excludes raw ingest, Level 0.5, processing-run inputs,
scratch, staging, transfer logs, and private definitions. Expand either filter
only after explicit review.

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
