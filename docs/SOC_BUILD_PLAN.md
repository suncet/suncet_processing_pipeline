# SunCET SOC Jetson Build and Operations Plan

Last updated: 2026-08-26

## Purpose

This document tracks the build of `suncet-soc`, an NVIDIA Jetson AGX Orin used as
the SunCET Science Operations Center processing node. The initial operating model
is manual processing after a satellite pass. Automated ingest and processing will
be added after the pipeline and its data sources are stable.

## Current system

- Hostname: `suncet-soc`
- Administrator account: `james`
- Operating system: Ubuntu 24.04.4 LTS, ARM64
- NVIDIA platform: L4T R39.2.1, Jetson AGX Orin
- Current storage: Ubuntu and the minimal runtime on internal eMMC; SunCET data
  on a Samsung 990 PRO 2 TB NVMe mounted at `/srv/suncet`
- Current network: `JHUAPL-Staff` Wi-Fi using DHCP
- Remote administration: SSH through the APL VPN
- Repository: <https://github.com/suncet/suncet_processing_pipeline>

Remote access was validated on 2026-08-20 from a Mac connected to a phone
hotspot and the APL VPN. `ssh james@suncet-soc` succeeded, and routing to the
Jetson used a VPN `utun` interface. A static IP is therefore not required at
present. The Jetson must remain powered, connected to Staff Wi-Fi, and reachable
under its hostname.

## Decisions

- Work under the `james` account, using passwordless SSH key authentication.
- Use Git `main` for the small development team; record the Git revision for
  every processing run.
- Use Miniforge/Mamba and the repository environment definitions.
- Support both ARM and Intel hosts. Python 3.12 and the current Python 3.14
  release are tested; the environment is not tied to the Mac's ARM architecture.
- Keep CUDA and JetPack components at the operating-system level rather than in
  the portable Python environment.
- Use `imagecodecs` for JPEG-LS decompression.
- Store mission-length telemetry in DuckDB, with one table per APID so each
  packet stream retains its own sampling cadence. Images remain separate FITS
  products; use ASOF joins for ad hoc cross-APID trending.
- Treat versioned CSV exports below `$suncet_data/metadata` as the current
  development definitions, and copy checksum-guarded snapshots into each named
  processing run. A run never follows later edits to the live Google Sheet.
- Keep the operating system, boot support, SSH access, source checkout, and a
  small reproducible production environment on eMMC. Use NVMe for SunCET data,
  products, processing scratch space, large caches, optional development
  environments, and container storage. This is the intended permanent layout,
  not merely a migration stage.
- Preserve the local `suncet_data` directory organization on the Jetson. It is
  rooted at `/srv/suncet/data` on the NVMe so the pipeline does not depend on
  the physical storage location.
- Keep private CTDB content outside `suncet_data`, which is publicly exposed and
  synchronized on development systems. Define the separate host-local
  `suncet_ctdb` root on every system. On the Jetson it will be a local directory,
  not a Box dependency, and must never be nested inside the public data root.
- Use ARM64 `rclone` with explicit one-way `copy` operations for the initial
  Dropbox integration. Dropbox is a replication/publication path, not the
  processing queue, sole backup, or authority for in-progress files. Do not use
  the unsupported native Dropbox Linux client or begin with bidirectional sync.
- Treat the LASP-owned AWS delivery buckets as operational landing areas, not
  long-term archives. Configure AWS-native S3 replication into a LASP-owned raw
  archive bucket using the `DEEP_ARCHIVE` destination storage class, followed
  by lifecycle expiration of the replicated delivery copy after a documented
  retention interval. LASP and the APL SOC independently pull the ordinary
  delivery copy; neither downloader archives or deletes it.
- Use outbound SFTP from the Jetson to the existing LASP public server for
  product publication. The port-22 path, key authentication, read/write/delete
  permissions, and byte-for-byte round trip were validated on 2026-08-25. Keep
  publication manual and checksum-verified until the product policy is mature.
  Do not host a public service on the Jetson.
- Begin with manual data transfer and processing. Do not add unattended jobs
  until the manual workflow is reliable and observable.
- Do not expose SSH directly to the public Internet. Use the APL VPN or another
  APL-approved management path.

## Security and APL network disposition

IT advised that this ARM64 Jetson should use an InfoSec policy exception to
"Extend APLNIS Connectivity for EOL / Unsupported / Retired OS" if APLNIS
connectivity is required. The current architecture does not require the Jetson
itself to join APLNIS: it remains on `JHUAPL-Staff`, while administrators reach
it from an APL-managed Mac through the APL VPN. Offsite VPN/SSH access has been
validated.

The selected posture is therefore:

- Proceed with the current NVIDIA Ubuntu/L4T installation on Staff Wi-Fi.
- Do not reimage with the APL Golden Image or install LCMS/BigFix.
- Do not request the APLNIS unsupported-platform exception unless APLNIS
  connectivity later becomes an operational requirement.
- Do not connect the Jetson to APLNIS Ethernet without first obtaining the
  required approval.
- Keep the system manually patched, restrict remote administration to SSH keys
  over APL-approved network paths, and do not expose public inbound services.
- Revisit the network/security design before storing controlled data, adding
  mission credentials, accepting inbound production services, or publishing
  products from the Jetson.

This interpretation is based on IT's response in `INC0541924`: the exception is
for extending APLNIS connectivity to the unsupported platform. A short ticket
reply confirming that no exception is needed while the Jetson stays off APLNIS
would provide a useful written record, but it does not block reversible
user-space environment work.

## Roadmap and status

### 0. Put development under reproducible version control — complete

- Important local pipeline work was committed and pushed to `main`.
- GitHub Actions passes on Python 3.12 and Python 3.14.
- Portable environment modernization is in commit `95b56bc`.
- Processing-run provenance is in commit `a24cd1d`.

### 1. Record a read-only Jetson baseline — complete

The baseline captured OS, kernel, L4T, storage, memory, CPU, power mode,
temperature, time synchronization, NVIDIA driver/CUDA compatibility, installed
development tools, containers, and LCMS/BigFix status. Important findings:

- The NVIDIA driver reports CUDA 13.2 compatibility, but the complete CUDA
  toolkit and `nvcc` are not installed.
- Docker, containerd, and NVIDIA Container Toolkit are installed.
- Miniforge/Mamba is not installed.
- eMMC has adequate short-term capacity but is not the intended production data
  volume.
- Time synchronization is active through `systemd-timesyncd`.
- The 2026-08-26 maintenance pass installed 34 standard Ubuntu updates,
  including security updates. Seven ordinary packages remained deferred by
  Ubuntu phased updates; `dpkg --audit` was clean and no reboot was required.
  Do not run the currently suggested `apt autoremove`, because its candidate
  list includes Jetson boot-support packages. Three pre-existing DHCP/DNS
  services remain failed and should be reviewed before changing networking;
  they did not affect Staff Wi-Fi, VPN/SSH, SFTP, or AWS connectivity.

### 2. Make the Python environment portable and Jetson-ready — complete

Completed in the repository:

- `environment.yml` defines the production/runtime environment.
- `environment-dev.yml` contains development, test, and notebook tools.
- `environment-radio.yml` isolates optional GNU Radio dependencies.
- `conda-lock.yml` locks tested packages for `linux-aarch64`, `linux-64`,
  `osx-arm64`, and `osx-64`.
- Lock solves succeeded for all four target platforms.
- The test suite passes under Python 3.12 and 3.14.
- Lossless 16-bit JPEG-LS round-trip behavior was validated with `imagecodecs`.
- Managed paths now use one required absolute `suncet_data` root; workstation
  paths and repository-relative processing storage have been removed from code.
- Private CTDBs use the separate required `suncet_ctdb` root. The pipeline
  rejects overlapping public-data and CTDB roots, and provenance manifests hide
  the host-specific private root.
- Regression tests scan Python, configuration, and notebook sources for
  workstation paths or direct legacy environment lookups.
- The staged Level 0.5 implementation is now the canonical
  `suncet_processing_pipeline.make_level0_5` module. The superseded class-based
  implementation and its dead batch helper were removed; their history remains
  available in Git.

Completed on the Jetson and refreshed on 2026-08-26:

- Cloned `main` at commit `f61c75f9ccbccea984765a98ed7db24637e2a2ae`
  into `/home/james/src/suncet_processing_pipeline`; the checkout was clean at
  the validation gate.
- Checksum-verified and installed ARM64 Miniforge 26.3.2-3 under
  `/home/james/miniforge3`. Conda base activation is disabled by default.
- Created the production `suncet` environment from the repository's exact
  `linux-aarch64` lock and installed the checkout in editable mode without
  resolving its dependencies again.
- Verified native ARM64 Python 3.14.7, `imagecodecs` 2026.8.16, the complete
  science-package import set, and `pip check` with no broken requirements.
- Created a cloned `suncet-test` validation environment with only pytest and
  coverage tooling added. After refreshing both environments from the current
  lock, all 116 repository tests passed in 5.80 seconds. The runtime includes
  DuckDB 1.5.5; both environments pass `pip check`.
- Promoted the refreshed Python 3.14.7 environments to the conventional
  `suncet` and `suncet-test` names. The earlier environments remain temporarily
  available under dated `prelock` rollback names; remove them only after a
  reasonable operational soak period.
- Verified lossless 16-bit JPEG-LS encode/decode with representative image data.
  An incompressible random-image encoder stress test also passed when given an
  explicit output buffer. The default encoder buffer can be too small for that
  artificial worst case; operational SunCET code consumes JPEG-LS by decoding,
  so this is recorded as a test-tool behavior rather than a processing blocker.
- Initially created `/home/james/suncet_ctdb` with mode `700` and deliberately
  left `suncet_data` undefined until permanent storage was ready. After the
  verified NVMe migration in Step 7, the redundant eMMC CTDB tree was removed.
- Miniforge, both environments, and the shared package cache use about 4.7 GB;
  approximately 40 GB remains free on eMMC.

The representative Level 0.5 comparison is recorded in Roadmap Step 6.

### 3. Improve processing-run reproducibility — complete

The pipeline writes an atomic JSON manifest for each processing run under the
data directory's `processing_manifests` folder. It records timing, success or
failure, sanitized arguments and configuration, Git state, environment and
package details, input and output checksums, and exception details. Successful
and intentionally failed runs have been tested.

Named processing runs additionally contain exact FITS and NetCDF/Zarr metadata
definition snapshots, their version, and a checksum manifest. Readers verify the
snapshot before use, so a modified historical definition fails visibly.

### 4. Choose the initial APL network posture — complete

Keep the Jetson on Staff Wi-Fi and off APLNIS. IT directed unsupported systems
that need APLNIS connectivity to the InfoSec exception process; this build does
not currently need that extension. Reopen this decision before moving to
Ethernet/APLNIS or expanding the node's data and service exposure.

### 4.1 Configure LASP AWS raw-data custody — replication, retention, and SOC read active

This work is independent of the Jetson NVMe installation and can proceed first.

- S3 Versioning was enabled on the LASP-owned X-band and UHF delivery buckets
  and the shared raw archive bucket on 2026-08-21. Exact AWS resource names are
  kept out of this public repository.
- Same-account live replication is enabled for new objects from both delivery
  buckets to the raw archive bucket, preserving the object key and selecting
  `DEEP_ARCHIVE` as the replica storage class. Source delete markers are not
  replicated, so source cleanup cannot delete the raw archive copy. LASP UHF
  uploads use the `uhf/` key prefix so they remain collision-free and visibly
  distinct in the shared archive.
- Initially retain each ordinary delivery object for 30 days so LASP and the
  Jetson can pull independently without a race. Revisit the interval after
  measuring actual volume and operational recovery needs.
- Matching source-bucket lifecycle rules were enabled on both delivery buckets
  on 2026-08-21 to expire current delivery versions after 30 days, permanently
  expire noncurrent versions after an additional 7-day recovery interval,
  remove expired delete markers, and abort incomplete multipart uploads after
  7 days.
- Monitor replication failures and inventory `PENDING`, `COMPLETED`, and
  `FAILED` states. Lifecycle must remain the deletion authority; neither the
  LASP downloader nor the Jetson may delete delivery objects.
- Restrict the Jetson AWS identity to the minimum list/read permissions needed
  for ingest. Record object key, version, size, and a content checksum in the SOC
  ingest ledger or processing manifest.
- A dedicated non-console SOC identity was created on 2026-08-26. Its policy is
  restricted to locating/listing the two delivery buckets and reading their
  objects and versions from the approved SOC public IPv4 address. Simulation
  and live tests confirmed that archive access, writes, deletes, and reads from
  another source address are denied. The initial manual workflow uses one
  mode-`0600` access key; migrate to temporary external-workload credentials
  before unattended production operation.
- The existing LASP software IAM principal was authorized on 2026-08-21 to
  locate and list the UHF delivery bucket and to upload, verify, and manage
  multipart uploads under `uhf/`. It has no permission to delete UHF delivery
  objects; lifecycle remains the deletion authority. No additional long-lived
  access key was created.
- The same LASP principal was reduced to list/read-only access on the X-band
  delivery bucket on 2026-08-21. Upload, deletion, and multipart-write actions
  are denied so the legacy script will fail visibly until its archive-copy and
  source-deletion responsibilities are removed. Its historical direct-write
  permissions on the archive remain separate cleanup debt; remove them after
  confirming that no retained LASP workflow needs them.
- Simplify the LASP local-ingest script so it only downloads and verifies new
  delivery files. AWS replication and lifecycle are now the archive and source
  cleanup authorities.
- Live replication was validated independently for the X-band and UHF paths on
  2026-08-21: each source reached `COMPLETED` and each destination reported
  `REPLICA` in `DEEP_ARCHIVE`. Lifecycle activation was verified through the S3
  API; confirm the first time-based expirations after the 30-day retention
  windows elapse.

### 5. Establish the short-term data layout and manual ingest — in progress

- The permanent NVMe-backed data root is active at `/srv/suncet/data`, exported
  as `suncet_data`, and initialized with the portable directory structure.
- The repository now provides a host-independent, dry-run-first AWS ingest
  command. Resource names and credentials remain in mode-`0600` host-local
  files. Each executed pull stages on NVMe, verifies reported size, computes
  SHA-256, atomically finalizes content, refuses conflicts, and writes a JSON
  receipt below `transfer_logs/aws_ingest`. It never changes the source object.
- A 29-byte real X-band delivery object was pulled successfully on 2026-08-26.
  A second execution reported `ALREADY_PRESENT`, left no partial file, and wrote
  a second idempotency receipt. Both delivery sources can be listed from the
  Jetson with the least-privilege identity.
- The remaining work in this step is public-data synchronization, the LASP
  public-server download path, and operational disk/retention thresholds.
- Install the ARM64 build of `rclone` and configure a Dropbox remote using a
  dedicated identity that can access only SunCET public data, if that account
  arrangement is available. Avoid placing a token for the owner's full personal
  Dropbox account on the Jetson. Store the rclone configuration with permissions
  restricted to `james`.
- Use explicit one-way `rclone copy` commands rather than `sync` or `bisync`:
  copy reference inputs from Dropbox to the Jetson, and copy finalized products
  and processing manifests from the Jetson to Dropbox. Because `copy` does not
  delete destination files, remote or local deletion will not be propagated by
  the initial workflow.
- Maintain a reviewed rclone filter file that includes only intended public
  subtrees. Exclude scratch space, caches, temporary/partial products, package
  stores, and all private material. `suncet_ctdb` is outside `suncet_data` and
  must never be an rclone source or destination.
- Write products outside the exported Dropbox subtree while they are incomplete,
  then move completed products atomically into their publication location.
- Begin with manual `--dry-run` and logged copy operations. Verify representative
  transfers in both directions before considering a systemd timer. Any later
  move to `sync` or `bisync` requires a separate review of ownership, conflicts,
  deletion propagation, and recovery behavior.
- Document a manual, resumable pull from
  <https://lasp.colorado.edu/data/store/suncet/>.
- Verify downloaded files with server metadata or checksums where available.
- Define retention thresholds so the eMMC cannot be filled by an ingest.
- LASP public-server write-back is authorized and manually validated as
  described below.

### 5.1 Publish finalized products to LASP — manual transport validated

- LASP whitelisted the Jetson's stable APL Staff Wi-Fi egress address. Outbound
  TCP/22 connectivity and dedicated-key SFTP authentication were verified on
  2026-08-25.
- The host-local SSH alias `lasp-sfs` keeps the endpoint, username, port, and
  identity path outside the public repository.
- A controlled test listed the remote public-data tree, uploaded and downloaded
  a 54-byte file, passed byte-for-byte `cmp`, and deleted all test artifacts.
- The repository's
  [`publish_sftp` workflow](LASP_SFTP_PUBLICATION.md) is dry-run-first and maps
  only explicit paths below `suncet_data`. It stages remotely, performs full
  SHA-256 read-back verification, atomically finalizes files, writes checksum
  sidecars and JSON transfer logs, retries bounded transient failures, skips
  matching content idempotently, and refuses conflicting content.
- Before routine publication, define the approved local-to-public directory
  mapping, product naming/versioning rules, release authority, and representative
  product validation. Do not schedule unattended publication yet.

### 6. Validate the end-to-end manual processing workflow — in progress

- A clean Python 3.14 ARM64 Mac run and a clean Python 3.14 ARM64 Jetson run
  processed the same checksum-verified 61,457,952-byte representative X-band
  file at repository commit `a7b2de2` on 2026-08-26.
- Both recovered 8,822 packets without decode failures and produced the same
  59-file non-provenance inventory, including ten CSIE image products. Forty-one
  binary and image artifacts matched SHA-256 exactly. A field-by-field
  comparison of the remaining CSVs found 1,431,630 identical non-path cells and
  zero data-value differences; the only differences were expected absolute
  working-directory strings.
- The first comparison exposed a stale private CTDB copy on the Jetson through
  32 beacon-cell differences. A checksum-based, non-destructive CTDB refresh
  resolved every value difference. Formal CTDB snapshot versioning and a
  repeatable refresh/verification procedure are therefore operational
  requirements, not optional housekeeping.
- Processing through later science levels remains dependent on approved
  calibration files.
- Measure elapsed time, peak memory, disk growth, and failure behavior.
- Write a concise operator runbook covering input discovery, processing,
  product review, retry, and recovery.

### 7. Add NVMe storage and establish the permanent storage split — complete

- Installed the Samsung 990 PRO 2 TB M.2 2280 NVMe. Its initial SMART report
  showed zero critical warnings, media errors, unsafe shutdowns, and endurance
  use, with 100% available spare.
- Created one GPT partition containing an ext4 filesystem labeled
  `SUNCET_NVME`. It is mounted by UUID at `/srv/suncet` with `noatime`, `nofail`,
  and a bounded systemd device timeout so an absent NVMe does not prevent
  recovery boot. The periodic `fstrim` timer is enabled and active.
- Set `suncet_data=/srv/suncet/data` and
  `suncet_ctdb=/srv/suncet/ctdb`. The paths are siblings: the mode-`700` private
  CTDB tree is not nested inside the public data tree.
- Copied 1,611 CTDB files totaling 185,914,112 bytes from eMMC to the NVMe with
  metadata preservation. Checksum comparisons before and after reboot reported
  no differences. A final comparison again passed before the redundant
  `/home/james/suncet_ctdb` source was deleted on 2026-08-25; the Jetson now has
  one CTDB tree at `/srv/suncet/ctdb`.
- Initialized the canonical `suncet_data` directory tree and downloaded the
  validated `v1.0.2dev` FITS and NetCDF/Zarr metadata exports. Mission-approved
  calibration FITS assets are not present yet.
- Verified the mount identity and fstab syntax, shell and Conda path behavior,
  pipeline path resolution, directory ownership, and available 1.8 TiB data
  capacity. A controlled reboot on 2026-08-25 automatically restored the
  correct UUID-backed mount and environment paths. The mount unit, checksum
  comparison, and a SHA-256 write/read/delete test all passed after boot.
- Require the mount before any future processing service can start.
- Keep the OS, SSH recovery path, repository, and minimal production runtime on
  eMMC. Direct high-volume and write-heavy storage to NVMe.
- Review Mamba package caches, optional environments, and Docker's data root
  before they grow; configure large stores directly on NVMe rather than copying
  installed environments afterward.
- Define backup, retention, filesystem monitoring, and recovery procedures.
- Keep code and environment reconstruction possible from Git and lock files.

### 8. Install and validate GPU compute support — pending operational need

- Inventory NVIDIA packages before making changes.
- Select the JetPack/CUDA packages compatible with L4T R39.2.1.
- Install only the required system components and validate with a small CUDA
  workload.
- Benchmark CPU and GPU implementations of PSF deconvolution when the
  GPU-enabled implementation is ready.
- Record Jetson power mode, clocks, temperature, throughput, memory, and energy
  per product for future flight-hardware suitability studies.

### 8.1 Resolve retained metadata documentation debt — pending

- Obtain the authoritative internal variable names, FITS keywords, units, and
  data types for the Dual-SPS Solar Angle Error X/Y fields, then update the live
  Google Sheet and versioned exports.
- Fill the currently blank provenance/source cells for the older FITS metadata
  rows. These blanks are documentation debt, not permission to infer sources in
  production processing.
- Inventory the mission-approved dark, flat, vignette, PSF, bad-pixel, and
  stray-light calibration products; define versioning and validity intervals
  before Level 1/2 production use.

### 9. Automate post-pass operations — future

After manual operations are dependable:

- Detect newly available LASP or Leaf Space inputs without duplicating work.
- Stage downloads atomically and validate completeness before processing.
- Run processing under a dedicated service or timer with an explicit locked
  environment and data root.
- Preserve manifests and logs; alert on failed or stale runs.
- Make retries idempotent and prevent concurrent processing of the same input.
- Coordinate maintenance and reboot windows with mission operations.

### 10. Establish product-publication policy — transport complete, policy pending

Product write-back to LASP is authorized and its manual transport is validated
under Step 5.1. Remaining work is operational policy: approved product levels,
directory and filename conventions, release authority, revision handling,
public documentation, alerting, and recovery. Public hosting on APL
infrastructure remains outside the initial SOC build.

### 11. Establish the public SatNOGS mission presence — in progress

Register SunCET and its UHF transmitter before launch, validate reception with
flight-representative RF data, publish a decoder for only the globally broadcast
CCSDS APID 1 beacon, build the public telemetry dashboard, and coordinate
post-launch identification. The NORAD catalog number is expected only after
launch and does not block the pre-launch work. This workstream is tracked in the
[SunCET SatNOGS onboarding plan](SATNOGS_ONBOARDING_PLAN.md).

The APID 1 public-field review is complete: 112 fields are approved for public
decoding and 24 remain opaque. The first generated bare-CCSDS Kaitai decoder
pass and a synthetic public test vector now compile and validate successfully.
Flight software has confirmed the literal AX.25 header, CCSDS encapsulation,
CRC-16/X-25 coverage, and FCS byte order. RF receiver-path integration and APID
1 packet length remain the technical follow-ups before upstream submission.
Fine time is empirically resolved as integer milliseconds and implemented in
the pipeline and public decoder artifacts.

## Immediate next action

Add AWS replication-failure monitoring, define CTDB snapshot/refresh checks, and
complete the manual public-data synchronization and LASP download procedures in
Roadmap Step 5. Then measure resource use for the representative processing run
and write the operator runbook in Step 6. Define and review the LASP product
mapping before using the verified Step 5.1 publisher on mission products. No
unattended ingest or publication should begin before those gates pass.

## Definition of an initial operational SOC

The initial build is ready for routine manual use when:

- VPN/SSH administration works from an offsite network.
- The Jetson remains off APLNIS unless the required exception is approved.
- The locked ARM64 environment installs reproducibly on the Jetson.
- Unit tests and JPEG-LS validation pass on the Jetson.
- A known dataset produces products consistent with a known-good Mac run.
- Every run emits a complete provenance manifest.
- Data paths, free-space thresholds, and manual recovery steps are documented.
