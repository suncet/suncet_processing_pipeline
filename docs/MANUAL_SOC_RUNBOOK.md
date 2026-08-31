# Manual SunCET SOC operator runbook

## Scope

This is the controlled manual workflow for `suncet-soc`. It covers readiness,
raw AWS ingest, CTDB integrity, processing, review, public-data copies, and
recovery. It does not authorize unattended ingest, automatic publication, AWS
resource changes, or deletion of upstream data.

Exit codes used by the preflight and monitoring tools are `0` for healthy, `1`
for a warning requiring review, and `2` for a critical stop condition.

## One-time storage policy

Copy `operations/soc_operations.example.ini` to
`$HOME/.config/suncet/soc_operations.ini`, review it, and set mode `0600`:

```ini
[storage:data]
path = ${suncet_data}
expected_mountpoint = /srv/suncet
expected_uuid = REPLACE_WITH_SUNCET_NVME_UUID
expected_source = /dev/nvme0n1p1
expected_fstype = ext4
warning_free_gib = 300
critical_free_gib = 150
warning_free_percent = 20
critical_free_percent = 10
warning_free_inodes = 1000000
critical_free_inodes = 250000
warning_free_inode_percent = 5
critical_free_inode_percent = 2
work_multiplier = 3
accepts_workload = true
require_writable = true
write_probe = true

[storage:system]
path = /
expected_mountpoint = /
warning_free_gib = 10
critical_free_gib = 5
warning_free_percent = 15
critical_free_percent = 8
warning_free_inodes = 100000
critical_free_inodes = 50000
warning_free_inode_percent = 5
critical_free_inode_percent = 2
work_multiplier = 1
accepts_workload = false
# The unprivileged SOC operator is not expected to write directly to `/`.
require_writable = false
write_probe = false
```

Populate `expected_uuid` from `findmnt -no UUID --target /srv/suncet`; do not
leave the example placeholder in the active policy. The preflight checks the
mountpoint, source device, filesystem type, UUID, read/write state, a private
create/fsync/remove probe on the workload volume, byte headroom, and inode
headroom. A nonzero planned input is accepted only when exactly one target is
declared as the workload target, and its expansion multiplier must be at least
one.

For the data volume, the larger of the byte and percentage thresholds wins.
The initial workload reserve is three times the compressed input size to allow
for raw staging, intermediate products, and output. These are conservative
commissioning values, not flight-derived constants. Revisit them after several
representative passes have recorded actual peak disk growth; reducing a stop
threshold requires an operations review.

## Before every pass or processing run

1. Confirm that the repository checkout and environment are the reviewed
   deployment, and record the commit and Jetson power mode:

   ```sh
   cd "$HOME"
   suncet_runtime_prefix=/home/james/.local/share/mamba/envs/suncet-release-4fbd7b9
   suncet_python="$suncet_runtime_prefix/bin/python"
   suncet_aws_cli=/home/james/.local/bin/aws
   if [ ! -x "$suncet_python" ] || [ ! -x "$suncet_aws_cli" ]; then
     printf 'STOP: reviewed Python or AWS CLI is unavailable\n' >&2
     false
   else
     (
       set -e
       "$suncet_python" --version
       cd /tmp
       "$suncet_python" -c \
         'from importlib.metadata import version; import suncet_processing_pipeline as p; print(version("suncet"), p.__file__)'
       git -C /home/james/src/suncet_processing_pipeline status --short
       git -C /home/james/src/suncet_processing_pipeline rev-parse HEAD
       nvpmodel -q
     )
   fi
   ```

   Stop immediately if the block prints `STOP` or any command fails. Running the
   import check from `/tmp` prevents the source checkout from shadowing the
   installed wheel, while `git -C` records the checkout without changing the
   shell's working directory.

   The package path must be below the reviewed runtime prefix, not the source
   checkout or an older environment. Do not claim a reproducible production
   run from a dirty checkout. Update the exact prefix only as part of a reviewed
   release deployment; do not silently point it at a mutable development
   environment.

2. Confirm the two managed roots and the NVMe mount. The roots must be siblings,
   never nested:

   ```sh
   printf 'suncet_data=%s\nsuncet_ctdb=%s\n' "$suncet_data" "$suncet_ctdb"
   findmnt /srv/suncet
   ```

3. Run the storage preflight. When the selected delivery object size is known,
   supply it so the expansion reserve participates in backpressure:

   ```sh
   "$suncet_python" -m suncet_processing_pipeline.soc_preflight
   "$suncet_python" -m suncet_processing_pipeline.soc_preflight \
     --planned-input-bytes OBJECT_SIZE_BYTES
   ```

   Stop on `CRITICAL`. On `WARNING`, do not start another pass until an operator
   has identified safe, recoverable space. Never solve pressure with an ad hoc
   recursive deletion.

4. Confirm the SOC public egress address still matches the address approved for
   the AWS policy and LASP SFTP allowlist:

   ```sh
   curl -4 https://checkip.amazonaws.com
   ```

   If it changed, stop AWS/SFTP diagnosis and update the external allowlists
   through their owners. Do not broaden a source-IP policy casually.

5. Verify the active private CTDB snapshot before decoding:

   ```sh
   "$suncet_python" -m suncet_processing_pipeline.ctdb_snapshot verify \
     --manifest "$suncet_ctdb/.suncet_ctdb_snapshot.json"
   ```

   Any missing, unexpected, or mismatched file is a stop condition. Use the
   quiescent exact-verification refresh procedure below; do not patch generated
   decoders directly on the SOC.

## Discover, preserve, and ingest a delivery

Check both configured sources as appropriate. The simple listing shows the
current objects; the version-aware monitor supplies the immutable version ID
and simultaneously checks replication custody:

```sh
"$suncet_python" -m suncet_processing_pipeline.ingest_s3 \
  --aws-cli "$suncet_aws_cli" list xband
"$suncet_python" -m suncet_processing_pipeline.ingest_s3 \
  --aws-cli "$suncet_aws_cli" list uhf
"$suncet_python" -m suncet_processing_pipeline.aws_replication_monitor \
  xband uhf --aws-cli "$suncet_aws_cli" \
  --pending-hours 24 --retention-days 37
```

Do not proceed if the monitor is nonzero. Review the exact key, version,
timestamp, size, and `COMPLETED` state, then copy the selected source, key, and
version ID exactly. Preview the operation without contacting or writing S3,
rerun the storage preflight with its byte size, and execute the exact versioned
pull:

```sh
source_name=SOURCE
object_key='OBJECT_KEY'
object_version='VERSION_ID_FROM_MONITOR'
"$suncet_python" -m suncet_processing_pipeline.ingest_s3 \
  --aws-cli "$suncet_aws_cli" \
  pull "$source_name" "$object_key" --version-id "$object_version"
"$suncet_python" -m suncet_processing_pipeline.ingest_s3 \
  --aws-cli "$suncet_aws_cli" \
  pull "$source_name" "$object_key" --version-id "$object_version" --execute
```

Confirm the final local path, SHA-256, S3 version/ETag/checksum information, and
mode-`0600` JSON receipt under
`$HOME/.local/state/suncet/aws_ingest/`. A repeated pull of
identical content should report `already_present`; different content at an
existing filename is a conflict and must not be overwritten.

The monitor uses the version listing rather than only current keys and inspects
every object version inside the complete current-plus-noncurrent lifecycle risk
window. `FAILED`, stale `PENDING`, stale/unknown status, truncated coverage, or
an unexpectedly empty source is a critical stop or AWS-owner review condition.
This is a read-only source-bucket check; it does not prove archive inventory by
itself and never changes replication or lifecycle. Lifecycle remains the only
delivery-deletion authority.

## Process and review

1. Create a named run and copy or reference only the verified local ingest.
2. Run Level 0.5 with an explicit configuration and input mode. Preserve its
   processing manifest even when the run fails.
3. Before advancing a level, review packet/decode counts, image inventory,
   warnings, and expected APIDs. Inspect representative FITS data and headers.
4. Confirm that the final manifest records the reviewed Git revision, sanitized
   resolved paths and CTDB versions, input hashes, success/failure, duration,
   and output hashes.
5. Record elapsed time, peak memory, peak data-volume growth, Jetson power mode,
   and any thermal anomaly while the manual workflow is being commissioned.

Production Level 1 and Level 2 processing remains gated on approved,
versioned calibration assets. A provisional synthetic Level 2 handoff must not
be relabeled as a science product.

## Copy or publish reviewed products

- Use the dry-run-first public Dropbox tasks in
  `docs/RCLONE_PUBLIC_DATA.md`. Review the complete filter-controlled path list
  before adding `--execute`.
- Use `suncet_processing_pipeline.publish_sftp` and
  `docs/LASP_SFTP_PUBLICATION.md` only for an explicitly approved public product
  set. Review the dry run before adding `--execute`.
- Do not publish private CTDB material, raw delivery credentials, transfer logs,
  staging files, or incomplete products.

## CTDB refresh and exact verification

On the authoritative development host, stop CTDB writers and any process that
generates or refreshes packet definitions. Activate the reviewed development
environment there, confirm `python` resolves inside it, and create a timestamped
private manifest from the quiescent tree. Do not use the Jetson-only
`$suncet_python` path for this step:

```sh
command -v python
python -m suncet_processing_pipeline.ctdb_snapshot snapshot \
  --root "$suncet_ctdb" \
  --output PRIVATE_CTDB_MANIFEST_PATH
```

The snapshot ignores `.DS_Store`, `__pycache__`, `*.pyc`, and `*.pyo`, which are
host caches rather than authoritative definitions. It rejects other symlinks or
special entries, detects files that change while being hashed, uses an
exclusive writer lock, and writes the manifest mode `0600`. Keep it outside
publicly synchronized `suncet_data`.

The Jetson intentionally keeps only one CTDB tree. Refresh that active tree
only while all decoders and CTDB writers are stopped and the authoritative
source is quiescent. First preview a checksum-aware in-place copy whose
`--delete-delay` output makes every obsolete destination file visible. Review
the complete itemized dry run, especially every deletion, before running the
identical command without `--dry-run`:

```sh
rsync --archive --checksum --delete-delay --itemize-changes --dry-run \
  --exclude='.DS_Store' --exclude='__pycache__/' \
  --exclude='*.pyc' --exclude='*.pyo' \
  "$suncet_ctdb/" \
  james@suncet-soc:/srv/suncet/ctdb/

rsync --archive --checksum --delete-delay --itemize-changes \
  --exclude='.DS_Store' --exclude='__pycache__/' \
  --exclude='*.pyc' --exclude='*.pyo' \
  "$suncet_ctdb/" \
  james@suncet-soc:/srv/suncet/ctdb/

if scp PRIVATE_CTDB_MANIFEST_PATH \
    james@suncet-soc:/srv/suncet/ctdb/.suncet_ctdb_snapshot.json.incoming; then
  ssh james@suncet-soc \
    'chmod 600 /srv/suncet/ctdb/.suncet_ctdb_snapshot.json.incoming && mv -f /srv/suncet/ctdb/.suncet_ctdb_snapshot.json.incoming /srv/suncet/ctdb/.suncet_ctdb_snapshot.json'
else
  printf 'STOP: CTDB manifest transfer failed; the active manifest was not replaced\n' >&2
  false
fi
```

Because this one-copy procedure does not retain an on-Jetson rollback tree, an
interrupted transfer leaves the CTDB unavailable for processing until the same
copy is rerun and exact verification passes. The authoritative off-host tree
and its manifest are the recovery source. Do not resume decoding after a
partial or unreviewed refresh.

From the development host, explicitly run the locked release verifier on the
Jetson before resuming:

```sh
ssh james@suncet-soc \
  '/home/james/.local/share/mamba/envs/suncet-release-4fbd7b9/bin/python -m suncet_processing_pipeline.ctdb_snapshot verify --root /srv/suncet/ctdb --manifest /srv/suncet/ctdb/.suncet_ctdb_snapshot.json'
```

Exact verification, the focused Level 0.5 tests, and a representative decode
must all pass before processing resumes. If the dry run shows an unexpected
deletion, stop and resolve the source-of-truth question rather than editing the
Jetson tree manually.

The verifier rejects symlinks and detects missing, added, resized, or
content-modified files. A snapshot contains private relative filenames and
hashes; it is not a public product.

An interrupted snapshot can leave a hidden `.<manifest-name>.lock` beside its
requested output. Do not remove it merely to make the command run: first prove
that no snapshot writer remains, preserve any partial file for diagnosis, and
then remove only that exact stale lock before retrying.

## Failure and recovery rules

- **Low disk or missing NVMe:** stop ingest/processing. Preserve the operating
  system and SSH recovery path on eMMC; do not let work fall through onto an
  unmounted `/srv/suncet` directory.
- **Interrupted AWS pull:** rerun the same exact object/version. The ingest tool
  uses process-specific partial files and atomic finalization.
- **Checksum or filename conflict:** quarantine and investigate. Do not
  overwrite, delete upstream content, or reinterpret the conflict as a retry.
- **Processing failure:** preserve its manifest and partial outputs until the
  cause is understood. Use a new named run for a materially changed input,
  configuration, CTDB, calibration set, or code revision.
- **Replication warning:** do not delete the delivery copy. Escalate to the AWS
  owner and prove archive custody before relying on lifecycle timing.
- **Publication failure:** retain the local final product and transfer receipt;
  retry the same idempotent command after correcting connectivity. Never make an
  automatic overwrite/delete the recovery mechanism.
- **Public IP change:** request corresponding AWS and LASP allowlist updates.

## End-of-run record

Record the UTC time, operator, source/key/version, ingest receipt, CTDB tree
SHA-256, Git commit and dirty state, environment, power mode, processing run ID,
processing status, product-review decision, copy/publication receipt, storage
free space, and any anomaly. Automation remains blocked until these manual
records and their recovery paths are routine.
