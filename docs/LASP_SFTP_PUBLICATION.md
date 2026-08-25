# LASP SFTP Product Publication

## Status and scope

Outbound SFTP publication from the SOC Jetson was validated manually on
2026-08-25. The test authenticated with a dedicated Jetson key through an SSH
configuration alias, listed the remote public-data tree, uploaded a 54-byte
test file, downloaded it, verified byte-for-byte equality, and deleted both the
remote test object and local temporary copies.

This procedure publishes only finalized, public SunCET products. It does not
publish private CTDB definitions, credentials, processing scratch files, or
incomplete products. The private `suncet_ctdb` tree must remain outside
`suncet_data` and must never be supplied as a publication source.

## Connection configuration

The repository contains no LASP username, hostname, or private-key path. On the
Jetson, configure an SSH alias named `lasp-sfs` in `~/.ssh/config` and protect
that file with mode `0600`. The alias must specify the approved endpoint,
account, port, dedicated identity file, and `IdentitiesOnly yes`.

Validate the host-local configuration independently before publishing:

```shell
sftp lasp-sfs
```

At the prompt, `pwd` and `ls` must succeed without a password. Do not weaken
host-key checking or enable password fallback in an automated context.

## Safe publisher

The publisher is dry-run-only unless `--execute` is supplied. Every source must
resolve below the required `suncet_data` root. A narrower `--local-base` may be
mapped directly to the remote publication root; this is useful when the local
tree has a product-level directory but the public tree starts with mission
year.

For example, to map local `level1/2027/...` products to remote `2027/...`:

```shell
python -m suncet_processing_pipeline.publish_sftp \
  --local-base "$suncet_data/level1" \
  "$suncet_data/level1/2027"
```

Review every printed local-to-remote path and its SHA-256 digest. Then perform
the same explicit publication:

```shell
python -m suncet_processing_pipeline.publish_sftp \
  --local-base "$suncet_data/level1" \
  --execute \
  "$suncet_data/level1/2027"
```

The command may receive one or more explicit files or directories. Never point
it at the entire data root merely for convenience. Select only a reviewed,
final product set.

## Transfer guarantees

For each file, the publisher:

1. Computes the local SHA-256 digest.
2. Refuses to overwrite any remote path with different content.
3. Uploads to a deterministic `.partial.<digest-prefix>` name.
4. Downloads the staged file and verifies its complete SHA-256 digest.
5. Rechecks that the local source did not change during transfer.
6. Renames the verified stage into its final path.
7. Publishes a standard adjacent `.sha256` sidecar through the same staged and
   read-back-verified process.
8. Writes a JSON record below
   `$suncet_data/transfer_logs/lasp_publication/`.

Read-back verification uses temporary space below
`$suncet_data/transfer_staging/lasp_publication/`, keeping large product copies
on the mission data volume rather than the operating-system filesystem.

Transient operations receive bounded retries with exponential backoff. A
second run is idempotent:

- `published` means a new verified file and checksum sidecar were finalized.
- `unchanged` means the remote file and matching checksum sidecar already
  existed and a fresh read-back matched the local digest, so no upload occurred.
- `adopted` means a pre-existing remote file was downloaded, matched the local
  digest, and received a checksum sidecar.

A checksum or content conflict stops publication. Resolve it through product
versioning and operator review; do not delete or overwrite the remote object as
an automatic retry strategy.

Individual SFTP operations have a one-hour timeout by default. Use
`--command-timeout SECONDS` for an intentionally larger product after reviewing
the expected transfer time; the connection-establishment timeout remains short.

## Recovery and operational limits

- A failed upload may leave only a deterministic `.partial.*` object. A retry
  reuses that name and re-verifies the complete contents before finalization.
- The publisher never deletes a finalized local product.
- The publisher never deletes a finalized remote product.
- Transfer logs contain paths, sizes, hashes, statuses, and errors, but no SSH
  keys or passwords.
- Publication remains a manual operation until representative mission products,
  naming policy, alerting, and recovery procedures have been reviewed.
- The selected SFTP receiver path is independent of the raw-data AWS custody
  and retention workflow.
